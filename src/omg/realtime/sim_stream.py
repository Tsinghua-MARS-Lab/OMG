from __future__ import annotations

import json
import os
import platform
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SimStreamConfig:
    bind: str
    fps: float = 20.0
    width: int = 640
    height: int = 360
    camera_view: str = "iso"
    follow_mode: str = "xy"
    camera_distance: float = 4.5
    camera_elevation: float = -18.0
    kinematics_path: str | Path = "assets/robots/g1/g1_kinematics.json"
    urdf_path: str | Path = "assets/robots/g1/g1_29dof.urdf"
    scene_preset: str = "studio"


def parse_host_port(bind: str) -> tuple[str, int]:
    text = str(bind).strip()
    if text.startswith("http://"):
        text = text[len("http://") :]
    if "/" in text:
        text = text.split("/", 1)[0]
    if text.count(":") != 1:
        raise ValueError(f"Expected HOST:PORT bind address, got {bind!r}")
    host, port_text = text.rsplit(":", 1)
    if host == "":
        host = "127.0.0.1"
    port = int(port_text)
    if port <= 0 or port > 65535:
        raise ValueError(f"Invalid port in bind address: {bind!r}")
    return host, port


class MujocoSimStream:
    def __init__(self, config: SimStreamConfig) -> None:
        os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL") or ("glfw" if platform.system() == "Darwin" else "egl")
        self.config = config
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._jpeg: bytes | None = None
        self._qpos_36: np.ndarray | None = None
        self._overlay_lines: list[str] = []
        self._frame_index = -1
        self._render_count = 0
        self._last_render = 0.0
        self._last_update_wall_time = 0.0
        self._closed = False

        import cv2
        import mujoco
        import torch

        from omg.render.mujoco import _make_camera, _set_data_qpos, build_mjcf, parse_urdf
        from omg.robots.g1.kinematics import G1Kinematics
        from omg.tracking.holomotion.video import camera_azimuth, draw_overlay, yaw_degrees_from_wxyz

        self._cv2 = cv2
        self._mujoco = mujoco
        self._torch = torch
        self._make_camera = _make_camera
        self._set_data_qpos = _set_data_qpos
        self._camera_azimuth = camera_azimuth
        self._draw_overlay = draw_overlay
        self._yaw_degrees_from_wxyz = yaw_degrees_from_wxyz

        self._kinematics = G1Kinematics(config.kinematics_path)
        urdf_data = parse_urdf(config.urdf_path)
        xml_string = build_mjcf(
            urdf_data,
            offscreen_width=int(config.width),
            offscreen_height=int(config.height),
            scene_preset=str(config.scene_preset),
        )
        self._model = mujoco.MjModel.from_xml_string(xml_string)
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=int(config.height), width=int(config.width))
        self._joint_qposadr = {}
        for joint_name in self._kinematics.joint_order:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint {joint_name} missing in MuJoCo model")
            self._joint_qposadr[joint_name] = int(self._model.jnt_qposadr[joint_id])

    def update(self, qpos_36: np.ndarray, *, frame_index: int, overlay_lines: Sequence[str] | None = None) -> None:
        min_interval = 1.0 / max(float(self.config.fps), 1e-6)
        now = time.perf_counter()
        with self._lock:
            if self._closed:
                return
            self._qpos_36 = np.asarray(qpos_36, dtype=np.float32).reshape(36).copy()
            self._overlay_lines = [str(line) for line in overlay_lines or ()]
            self._frame_index = int(frame_index)
            self._last_update_wall_time = time.time()
            if now - self._last_render < min_interval:
                return
            self._last_render = now
        frame = self._render_frame(qpos_36, overlay_lines=overlay_lines)
        ok, encoded = self._cv2.imencode(".jpg", frame[..., ::-1], [int(self._cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            raise RuntimeError("Failed to encode MuJoCo frame as JPEG")
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._frame_index = int(frame_index)
            self._render_count += 1
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._renderer.close()

    def wait_jpeg(self, previous_render_count: int, timeout: float = 2.0) -> tuple[bytes | None, int, int]:
        deadline = time.perf_counter() + float(timeout)
        with self._condition:
            while not self._closed and self._render_count <= previous_render_count:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)
            return self._jpeg, self._render_count, self._frame_index

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "frame_index": int(self._frame_index),
                "render_count": int(self._render_count),
                "fps": float(self.config.fps),
                "width": int(self.config.width),
                "height": int(self.config.height),
                "updated_time": float(self._last_update_wall_time),
                "has_qpos": self._qpos_36 is not None,
            }

    def state(self) -> dict[str, object]:
        with self._lock:
            qpos = None if self._qpos_36 is None else self._qpos_36.astype(float).tolist()
            return {
                "frame_index": int(self._frame_index),
                "render_count": int(self._render_count),
                "fps": float(self.config.fps),
                "width": int(self.config.width),
                "height": int(self.config.height),
                "updated_time": float(self._last_update_wall_time),
                "overlay_lines": list(self._overlay_lines),
                "qpos_36": qpos,
            }

    def _render_frame(self, qpos_36: np.ndarray, *, overlay_lines: Sequence[str] | None) -> np.ndarray:
        qpos = np.asarray(qpos_36, dtype=np.float32).reshape(36)
        self._set_data_qpos(self._data, qpos, self._joint_qposadr, self._kinematics.joint_name_to_qpos_index)
        self._mujoco.mj_forward(self._model, self._data)
        root = qpos[:3].astype(np.float64, copy=True)
        lookat = root.copy()
        if str(self.config.follow_mode) in {"xy", "heading"}:
            lookat[2] += 0.35
        elif str(self.config.follow_mode) == "xyz":
            lookat[2] += 0.35
        elif str(self.config.follow_mode) != "none":
            raise ValueError(f"Unsupported follow_mode: {self.config.follow_mode}")
        azimuth = self._camera_azimuth(str(self.config.camera_view))
        if str(self.config.follow_mode) == "heading":
            azimuth = self._yaw_degrees_from_wxyz(qpos[3:7]) + azimuth
        camera = self._make_camera(
            self._model,
            lookat=lookat,
            distance=float(self.config.camera_distance),
            azimuth=float(azimuth),
            elevation=float(self.config.camera_elevation),
        )
        self._renderer.update_scene(self._data, camera=camera)
        return self._draw_overlay(self._renderer.render(), overlay_lines)


class SimStreamServer:
    def __init__(self, config: SimStreamConfig) -> None:
        self.stream = MujocoSimStream(config)
        self.config = config
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        host, port = parse_host_port(self.config.bind)
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/video.mjpg"):
                    self._send_mjpeg()
                    return
                if self.path.startswith("/status"):
                    self._send_json(outer.stream.status())
                    return
                if self.path.startswith("/state.json"):
                    self._send_json(outer.stream.state())
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def log_message(self, fmt: str, *args: object) -> None:
                print(f"[sim-stream] {self.address_string()} - {fmt % args}", flush=True)

            def _send_json(self, payload: dict[str, object]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_mjpeg(self) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=omg")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                render_count = -1
                while True:
                    jpeg, render_count, _frame_index = outer.stream.wait_jpeg(render_count)
                    if jpeg is None:
                        continue
                    try:
                        self.wfile.write(b"--omg\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return

        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[sim-stream] serving http://{host}:{port}/video.mjpg", flush=True)

    def update(self, qpos_36: np.ndarray, *, frame_index: int, overlay_lines: Sequence[str] | None = None) -> None:
        self.stream.update(qpos_36, frame_index=frame_index, overlay_lines=overlay_lines)

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.stream.close()
