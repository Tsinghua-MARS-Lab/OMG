from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


_CAMERA_AZIMUTHS = {
    "back": 0.0,
    "side": 90.0,
    "iso": 135.0,
    "front": 180.0,
}


def camera_azimuth(camera_view: str) -> float:
    try:
        return _CAMERA_AZIMUTHS[camera_view]
    except KeyError as exc:
        raise ValueError(f"Unsupported camera_view: {camera_view}") from exc


def yaw_degrees_from_wxyz(quat_wxyz: np.ndarray) -> float:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Invalid root quaternion for camera heading: {quat_wxyz}")
    w, x, y, z = quat / norm
    return float(np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))))


def _format_overlay_lines(lines: Sequence[str] | None) -> list[str]:
    if lines is None:
        return []
    return [str(line) for line in lines if str(line) != ""]


def draw_overlay(frame_rgb: np.ndarray, lines: Sequence[str] | None) -> np.ndarray:
    overlay_lines = _format_overlay_lines(lines)
    if not overlay_lines:
        return frame_rgb
    import cv2

    frame = np.ascontiguousarray(frame_rgb)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    line_height = 21
    pad_x = 12
    pad_y = 10
    max_width = 0
    for line in overlay_lines:
        (width, _), _ = cv2.getTextSize(line, font, scale, thickness)
        max_width = max(max_width, width)
    rect_w = min(frame.shape[1] - 2 * pad_x, max_width + 2 * pad_x)
    rect_h = len(overlay_lines) * line_height + 2 * pad_y
    panel = frame.copy()
    cv2.rectangle(panel, (8, 8), (8 + rect_w, 8 + rect_h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0.0, dst=frame)
    for idx, line in enumerate(overlay_lines):
        y = 8 + pad_y + 15 + idx * line_height
        cv2.putText(frame, line, (8 + pad_x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def open_video_writer(
    enabled: bool,
    model: Any,
    path: str | Path | None,
    width: int,
    height: int,
    fps: float,
    *,
    camera_view: str = "side",
    camera_distance: float = 4.0,
    camera_elevation: float = -20.0,
):
    if not enabled:
        return None, None, None
    import imageio.v2 as imageio
    import mujoco

    video_path = Path(path or "outputs/holomotion_rollouts/tracker_only.mp4").expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = float(camera_distance)
    camera.azimuth = camera_azimuth(camera_view)
    camera.elevation = float(camera_elevation)
    writer = imageio.get_writer(str(video_path), fps=float(fps), codec="libx264", ffmpeg_log_level="warning")
    return renderer, camera, writer


def append_video_frame(
    renderer: Any,
    camera: Any,
    writer: Any,
    data: Any,
    *,
    track_pos: np.ndarray | None = None,
    camera_view: str = "side",
    follow_mode: str = "xy",
    camera_elevation: float = -20.0,
    overlay_lines: Sequence[str] | None = None,
) -> None:
    if writer is None:
        return
    camera.azimuth = camera_azimuth(camera_view)
    camera.elevation = float(camera_elevation)
    if track_pos is not None and follow_mode != "none":
        pos = np.asarray(track_pos, dtype=np.float64).reshape(3)
        if follow_mode in {"xy", "heading"}:
            camera.lookat[0] = pos[0]
            camera.lookat[1] = pos[1]
            camera.lookat[2] = pos[2] + 0.35
        elif follow_mode == "xyz":
            camera.lookat[:] = pos
            camera.lookat[2] += 0.35
        else:
            raise ValueError(f"Unsupported follow_mode: {follow_mode}")
    if follow_mode == "heading":
        camera.azimuth = yaw_degrees_from_wxyz(np.asarray(data.qpos[3:7])) + camera_azimuth(camera_view)
    renderer.update_scene(data, camera=camera)
    writer.append_data(draw_overlay(renderer.render(), overlay_lines))


def close_video(renderer: Any, writer: Any) -> None:
    if writer is not None:
        writer.close()
    if renderer is not None:
        renderer.close()
