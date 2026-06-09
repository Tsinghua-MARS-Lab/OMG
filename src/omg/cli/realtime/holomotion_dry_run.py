from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from omg.realtime.motion_buffer import ExecutedHistoryBuffer
from omg.realtime.orin_client import RealtimeOrinBufferClient, RealtimeOrinBufferClientConfig
from omg.realtime.status_log import append_jsonl
from omg.tracking.holomotion.reference import body_angvel_from_quats, finite_difference, resample_qpos

HEADER_SIZE = 1280
DEFAULT_TOPIC = b"obs65"


def _load_seed_motion(path: str | Path, fps: float | None) -> tuple[np.ndarray, float]:
    seed_path = Path(path).expanduser()
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed motion not found: {seed_path}")
    loaded_fps = None
    if seed_path.suffix == ".npy":
        qpos = np.load(seed_path)
    elif seed_path.suffix == ".npz":
        with np.load(seed_path, allow_pickle=False) as data:
            for key in ("qpos_36", "pred_qpos_36", "qpos"):
                if key in data:
                    qpos = np.asarray(data[key])
                    break
            else:
                raise KeyError(f"No qpos_36, pred_qpos_36, or qpos key found in {seed_path}")
            if "fps" in data:
                loaded_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    else:
        raise ValueError(f"Unsupported seed motion extension: {seed_path.suffix}")
    resolved_fps = float(fps) if fps is not None else loaded_fps
    if resolved_fps is None:
        raise ValueError(f"--seed-fps is required when seed file does not contain fps: {seed_path}")
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim == 3 and qpos.shape[0] == 1:
        qpos = qpos[0]
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected seed qpos shape (T,36), got {qpos.shape}")
    return qpos.astype(np.float32, copy=False), resolved_fps


def _qpos_to_latest_obs(qpos_36: np.ndarray, *, fps: float) -> np.ndarray:
    qpos = np.asarray(qpos_36, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos_36 shape (T,36), got {qpos.shape}")
    dof_pos = qpos[:, 7:36]
    dof_vel = finite_difference(dof_pos, fps)
    latest_obs = np.concatenate([dof_pos, dof_vel, qpos[:, :3], qpos[:, 3:7]], axis=1)
    if latest_obs.shape[1] != 65:
        raise RuntimeError(f"latest_obs must have dim 65, got {latest_obs.shape}")
    return latest_obs.astype(np.float32, copy=False)


def _pack_latest_obs(topic: bytes, latest_obs: np.ndarray, *, frame_index: int) -> bytes:
    obs = np.asarray(latest_obs, dtype=np.float32).reshape(65)
    timestamp = np.asarray([time.time()], dtype=np.float64)
    frame = np.asarray([int(frame_index)], dtype=np.int64)
    fields = [
        {"name": "latest_obs", "dtype": "f32", "shape": [65]},
        {"name": "timestamp_realtime", "dtype": "f64", "shape": [1]},
        {"name": "frame_index", "dtype": "i64", "shape": [1]},
    ]
    header = json.dumps({"fields": fields}, separators=(",", ":")).encode("utf-8")
    if len(header) > HEADER_SIZE:
        raise RuntimeError(f"ZMQ latest_obs header exceeds {HEADER_SIZE} bytes")
    payload = header + b"\x00" * (HEADER_SIZE - len(header))
    payload += obs.astype("<f4", copy=False).tobytes()
    payload += timestamp.astype("<f8", copy=False).tobytes()
    payload += frame.astype("<i8", copy=False).tobytes()
    return topic + payload


class _LatestObsPublisher:
    def __init__(self, bind: str | None, *, topic: str) -> None:
        self.bind = bind
        self.topic = topic.encode("utf-8")
        self._ctx = None
        self._socket = None
        if bind is None:
            return
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Publishing HoloMotion latest_obs requires pyzmq") from exc
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.bind(bind)

    def publish(self, latest_obs: np.ndarray, *, frame_index: int) -> None:
        if self._socket is None:
            return
        self._socket.send(_pack_latest_obs(self.topic, latest_obs, frame_index=frame_index))

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
        if self._ctx is not None:
            self._ctx.term()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run the Orin-side realtime HoloMotion reference buffer without publishing robot actions."
    )
    parser.add_argument("--connect", default="tcp://127.0.0.1:5555")
    parser.add_argument("--seed-motion", required=True)
    parser.add_argument("--seed-fps", type=float, default=None)
    parser.add_argument("--history-fps", type=float, default=30.0)
    parser.add_argument("--history-frames", type=int, default=10)
    parser.add_argument("--tracker-fps", type=float, default=50.0)
    parser.add_argument("--num-frames", type=int, default=180)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run until interrupted instead of stopping after --num-frames. Partial output is saved on Ctrl-C.",
    )
    parser.add_argument("--replan-remaining-frames", type=int, default=40)
    parser.add_argument(
        "--condition-sequence",
        required=True,
        help=(
            "Per-replan condition sequence sent to the planner, e.g. "
            "'text[5]: walk forward | text: turn right | audio: /path/song.wav'."
        ),
    )
    parser.add_argument(
        "--condition-audio-step-frames",
        type=int,
        default=None,
        help="Audio feature frames advanced per repeated condition chunk. Defaults to generated frames per request.",
    )
    parser.add_argument(
        "--audio-type",
        choices=["audio", "feature"],
        default="audio",
        help=(
            "Condition-sequence audio mode. audio slices wav at replan time; "
            "feature precomputes current35 features from wav at planner startup and slices the feature timeline."
        ),
    )
    parser.add_argument("--audio-feature-type", choices=["current35"], default="current35")
    parser.add_argument("--audio-fps", type=float, default=30.0)
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--publish-bind", default=None, help="Optional HoloMotion obs65 PUB bind URI, e.g. tcp://*:6001.")
    parser.add_argument("--publish-topic", default=DEFAULT_TOPIC.decode("utf-8"))
    parser.add_argument("--status-jsonl", default=None, help="Optional live bridge status JSONL path.")
    parser.add_argument("--sim-stream-bind", default=None, help="Optional MuJoCo MJPEG stream bind address, e.g. 127.0.0.1:7870.")
    parser.add_argument("--sim-stream-fps", type=float, default=20.0)
    parser.add_argument("--sim-stream-width", type=int, default=640)
    parser.add_argument("--sim-stream-height", type=int, default=360)
    parser.add_argument("--sim-camera-view", default="iso", choices=["back", "side", "iso", "front"])
    parser.add_argument("--sim-follow-mode", default="xy", choices=["none", "xy", "xyz", "heading"])
    parser.add_argument("--sim-camera-distance", type=float, default=4.5)
    parser.add_argument("--sim-camera-elevation", type=float, default=-18.0)
    parser.add_argument("--sleep", action="store_true", help="Sleep at tracker fps while dry-running.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.continuous and args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.replan_remaining_frames < 0:
        raise ValueError("--replan-remaining-frames must be non-negative")
    condition_sequence = str(args.condition_sequence).strip()
    if not condition_sequence:
        raise ValueError("--condition-sequence must be non-empty")
    seed_qpos, seed_fps = _load_seed_motion(args.seed_motion, args.seed_fps)
    history_qpos = resample_qpos(seed_qpos, source_fps=seed_fps, target_fps=args.history_fps)
    history = ExecutedHistoryBuffer(target_fps=args.history_fps, max_frames=args.history_frames)
    history.append(history_qpos[-args.history_frames :], fps=args.history_fps)

    client = RealtimeOrinBufferClient(
        RealtimeOrinBufferClientConfig(
            connect=args.connect,
            tracker_fps=float(args.tracker_fps),
            request_timeout_ms=int(args.timeout_ms),
        )
    )
    publisher = _LatestObsPublisher(args.publish_bind, topic=args.publish_topic)
    sim_stream = None
    if args.sim_stream_bind is not None:
        from omg.realtime.sim_stream import SimStreamConfig, SimStreamServer

        sim_stream = SimStreamServer(
            SimStreamConfig(
                bind=str(args.sim_stream_bind),
                fps=float(args.sim_stream_fps),
                width=int(args.sim_stream_width),
                height=int(args.sim_stream_height),
                camera_view=str(args.sim_camera_view),
                follow_mode=str(args.sim_follow_mode),
                camera_distance=float(args.sim_camera_distance),
                camera_elevation=float(args.sim_camera_elevation),
            )
        )
        sim_stream.start()
    executed_frames: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    cursor = 0
    replan_index = 0
    condition_session_id = uuid.uuid4().hex

    def request_and_append(current_cursor: int) -> int:
        nonlocal replan_index
        condition_index = int(replan_index)
        replan_index += 1
        started = time.perf_counter()
        response = client.request_plan(
            tracker_frame=current_cursor,
            qpos_36_history=history.history(args.history_frames),
            history_fps=float(args.history_fps),
            metadata={
                "dry_run": True,
                "condition_sequence": condition_sequence,
                "condition_index": condition_index,
                "condition_session_id": condition_session_id,
                "condition_source": "condition_sequence",
                "audio_fps": float(args.audio_fps),
                "tracker_fps": float(args.tracker_fps),
                "audio_type": str(args.audio_type),
                "audio_feature_type": str(args.audio_feature_type),
                "condition_audio_step_frames": args.condition_audio_step_frames,
            },
        )
        latency = time.perf_counter() - started
        elapsed_frames = int(math.floor(latency * float(args.tracker_fps) + 1e-9))
        bridge_frames = min(elapsed_frames, max(0, client.buffer.remaining(current_cursor)))
        if bridge_frames > 0:
            execute_frames(current_cursor, bridge_frames)
            current_cursor += bridge_frames
        client.append_response(response, current_tracker_frame=current_cursor)
        transport = response.metadata.get("realtime_transport", {})
        event = {
            "kind": "replan",
            "plan_id": int(response.plan_id),
            "request_tracker_frame": int(response.request_tracker_frame),
            "append_tracker_frame": int(current_cursor),
            "latency_seconds": float(latency),
            "elapsed_tracker_frames": int(elapsed_frames),
            "bridge_frames": int(bridge_frames),
            "buffer_frames": int(client.buffer.frames),
            "remaining_after_append": int(client.buffer.remaining(current_cursor)),
            "timing_ms": response.metadata.get("timing_ms", {}),
            "transport_timing": transport,
            "prompt": response.prompt,
            "condition_sequence": condition_sequence,
            "condition_index": condition_index,
            "response_condition": response.metadata.get("realtime_condition"),
        }
        events.append(event)
        append_jsonl(args.status_jsonl, event)
        print(
            f"[dry-run replan {response.plan_id:04d}] request={response.request_tracker_frame:05d} "
            f"append={current_cursor:05d} condition={condition_index:04d} prompt={response.prompt!r} "
            f"latency={latency * 1000.0:.3f}ms "
            f"server={float(transport.get('server_plan_ms', response.planning_latency_seconds * 1000.0)):.3f}ms "
            f"net_queue={float(transport.get('client_network_queue_total_estimate_ms', 0.0)):.3f}ms "
            f"recv_decode={float(transport.get('client_recv_decode_ms', 0.0)):.3f}ms "
            f"elapsed={elapsed_frames} buffer={client.buffer.frames}",
            flush=True,
        )
        return current_cursor

    def execute_frames(start_cursor: int, frames: int) -> None:
        chunk = client.buffer.slice(start_cursor, frames)
        latest_obs = _qpos_to_latest_obs(chunk, fps=float(args.tracker_fps))
        for local_idx, obs in enumerate(latest_obs):
            frame_index = start_cursor + local_idx
            publisher.publish(obs, frame_index=frame_index)
            if sim_stream is not None:
                sim_stream.update(
                    chunk[local_idx],
                    frame_index=frame_index,
                    overlay_lines=[
                        f"condition: {condition_sequence}",
                        f"tracker frame: {frame_index}",
                        f"buffer remaining: {client.buffer.remaining(frame_index)}",
                    ],
                )
            if args.sleep:
                time.sleep(1.0 / float(args.tracker_fps))
        executed_frames.append(chunk)
        history.append(chunk, fps=float(args.tracker_fps))

    interrupted = False
    try:
        cursor = request_and_append(0)
        while args.continuous or cursor < int(args.num_frames):
            remaining_total = math.inf if args.continuous else int(args.num_frames) - cursor
            remaining_buffer = client.buffer.remaining(cursor)
            if remaining_buffer <= 0:
                raise RuntimeError(f"Motion buffer underrun at tracker frame {cursor}")
            if remaining_buffer <= int(args.replan_remaining_frames):
                cursor = request_and_append(cursor)
                remaining_total = math.inf if args.continuous else int(args.num_frames) - cursor
                remaining_buffer = client.buffer.remaining(cursor)
                if remaining_total <= 0:
                    break
            step_capacity = max(1, remaining_buffer - int(args.replan_remaining_frames))
            step_frames = step_capacity if args.continuous else min(int(remaining_total), step_capacity)
            if remaining_total <= remaining_buffer:
                step_frames = int(remaining_total)
            execute_frames(cursor, step_frames)
            cursor += step_frames
    except KeyboardInterrupt:
        interrupted = True
        print(f"interrupted_at_frame={cursor}; saving partial dry-run output", flush=True)
    finally:
        publisher.close()
        if sim_stream is not None:
            sim_stream.close()
        client.close()

    executed = (
        np.concatenate(executed_frames, axis=0).astype(np.float32, copy=False)
        if executed_frames
        else np.zeros((0, 36), dtype=np.float32)
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        executed_qpos_36=executed,
        latest_obs=_qpos_to_latest_obs(executed, fps=float(args.tracker_fps)) if executed.shape[0] else np.zeros((0, 65), dtype=np.float32),
        fps=np.asarray([float(args.tracker_fps)], dtype=np.float32),
        events=np.asarray([json.dumps(events, sort_keys=True)]),
        condition_sequence=np.asarray([condition_sequence], dtype=np.str_),
        continuous=np.asarray([bool(args.continuous)]),
        interrupted=np.asarray([bool(interrupted)]),
    )
    print(f"dry_run_output={output.resolve()} frames={executed.shape[0]} replans={len(events)}")


if __name__ == "__main__":
    main()
