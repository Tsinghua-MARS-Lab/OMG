from __future__ import annotations

import argparse
import bisect
import json
import math
import struct
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from omg.cli.realtime.holomotion_dry_run import _LatestObsPublisher, _qpos_to_latest_obs
from omg.realtime.orin_client import RealtimeOrinBufferClient, RealtimeOrinBufferClientConfig
from omg.realtime.status_log import append_jsonl


@dataclass(frozen=True)
class LowStateSample:
    timestamp: float
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    root_quat_wxyz: np.ndarray
    remote_key_mask: int


@dataclass(frozen=True)
class PendingReplan:
    request_tracker_frame: int
    history_source: str
    condition_sequence: str
    condition_index: int
    condition_source: str
    started: float


def _load_condition_sequence(args: argparse.Namespace) -> str:
    sequence = str(args.condition_sequence).strip()
    if not sequence:
        raise ValueError("--condition-sequence must be non-empty")
    return sequence


def _default_joint_angles(cfg: dict[str, Any], dof_names: list[str], *, config_path: Path) -> np.ndarray:
    raw = cfg.get("default_joint_angles")
    if raw is None:
        raw = cfg.get("default_position")
    if raw is None:
        raise ValueError(f"{config_path} must define default_joint_angles or default_position for realtime hold qpos")
    if isinstance(raw, dict):
        missing = [name for name in dof_names if name not in raw]
        if missing:
            raise ValueError(f"Default joint angle config is missing joints: {missing}")
        return np.asarray([float(raw[name]) for name in dof_names], dtype=np.float32)
    values = np.asarray(raw, dtype=np.float32).reshape(-1)
    if values.shape[0] != len(dof_names):
        raise ValueError(
            f"Default joint angle config must have {len(dof_names)} values, got {values.shape[0]}"
        )
    return values.astype(np.float32, copy=False)


def _load_holomotion_mapping(config_path: str | Path, *, default_root_height: float) -> tuple[list[str], list[int], str, np.ndarray]:
    path = Path(config_path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dof_names = list(cfg.get("complete_dof_order") or [])
    mapping = dict(cfg.get("dof2motor_idx_mapping") or {})
    if len(dof_names) != 29:
        raise ValueError(f"Expected 29 complete_dof_order entries in {path}, got {len(dof_names)}")
    missing = [name for name in dof_names if name not in mapping]
    if missing:
        raise ValueError(f"Missing dof2motor_idx_mapping entries in {path}: {missing}")
    indices = [int(mapping[name]) for name in dof_names]
    if len(set(indices)) != len(indices):
        raise ValueError(f"dof2motor_idx_mapping contains duplicate indices for {path}")
    topic = str(cfg.get("lowstate_topic") or "/lowstate")
    default_qpos = np.zeros((36,), dtype=np.float32)
    default_qpos[2] = float(default_root_height)
    default_qpos[3] = 1.0
    default_qpos[7:36] = _default_joint_angles(cfg, dof_names, config_path=path)
    return dof_names, indices, topic, default_qpos


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(out))
    if not np.isfinite(norm) or norm <= 1e-6:
        raise ValueError(f"Invalid root quaternion: {quat}")
    return (out / norm).astype(np.float32, copy=False)


def _nearest_root_position(
    reference: np.ndarray,
    *,
    frame_index: int,
    default_qpos: np.ndarray,
) -> np.ndarray:
    if reference.shape[0] > 0:
        idx = min(max(int(frame_index), 0), reference.shape[0] - 1)
        return reference[idx, :3].astype(np.float32, copy=True)
    return np.asarray(default_qpos[:3], dtype=np.float32).copy()


class LowStateHistory:
    def __init__(self, *, topic: str, motor_indices: list[int], max_seconds: float = 10.0) -> None:
        self.topic = topic
        self.motor_indices = list(motor_indices)
        self.max_seconds = float(max_seconds)
        self._samples: deque[LowStateSample] = deque()
        self._lock = threading.Lock()
        self._node = None
        self._spin_thread: threading.Thread | None = None
        self._rclpy = None
        self._last_remote_key_mask = 0
        self._remote_key_rising_edges = 0

    def start(self) -> None:
        import rclpy
        from rclpy.node import Node
        from unitree_hg.msg import LowState

        if not rclpy.ok():
            rclpy.init(args=None)

        class _LowStateNode(Node):
            pass

        node = _LowStateNode("omg_real_bridge")
        node.create_subscription(LowState, self.topic, self._callback, 50)
        self._rclpy = rclpy
        self._node = node
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        self._spin_thread.start()

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
            self._spin_thread = None

    def _callback(self, msg: Any) -> None:
        motor_state = msg.motor_state
        dof_pos = np.asarray([float(motor_state[idx].q) for idx in self.motor_indices], dtype=np.float32)
        dof_vel = np.asarray([float(motor_state[idx].dq) for idx in self.motor_indices], dtype=np.float32)
        root_quat = _normalize_quat_wxyz(np.asarray(msg.imu_state.quaternion, dtype=np.float32))
        remote_key_mask = _remote_key_mask(getattr(msg, "wireless_remote", []))
        sample = LowStateSample(
            timestamp=time.perf_counter(),
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            root_quat_wxyz=root_quat,
            remote_key_mask=remote_key_mask,
        )
        with self._lock:
            self._samples.append(sample)
            self._remote_key_rising_edges |= remote_key_mask & ~self._last_remote_key_mask
            self._last_remote_key_mask = remote_key_mask
            cutoff = sample.timestamp - self.max_seconds
            while self._samples and self._samples[0].timestamp < cutoff:
                self._samples.popleft()

    def latest_age_seconds(self) -> float | None:
        with self._lock:
            if not self._samples:
                return None
            return time.perf_counter() - self._samples[-1].timestamp

    def consume_remote_key_rising_edge(self, key_bit: int) -> bool:
        bit = 1 << int(key_bit)
        with self._lock:
            pressed = bool(self._remote_key_rising_edges & bit)
            self._remote_key_rising_edges &= ~bit
        return pressed

    def latest_hold_qpos_36(self, *, default_qpos_36: np.ndarray) -> np.ndarray:
        with self._lock:
            sample = self._samples[-1] if self._samples else None
        if sample is None:
            return default_qpos_36.reshape(1, 36).astype(np.float32, copy=True)
        qpos = default_qpos_36.reshape(1, 36).astype(np.float32, copy=True)
        qpos[0, 3:7] = sample.root_quat_wxyz
        qpos[0, 7:36] = sample.dof_pos
        return qpos

    def history_qpos_36(
        self,
        *,
        frames: int,
        fps: float,
        tracker_fps: float,
        current_tracker_frame: int,
        reference_qpos_36: np.ndarray,
        default_qpos_36: np.ndarray,
    ) -> np.ndarray | None:
        frames = int(frames)
        fps = float(fps)
        with self._lock:
            samples = list(self._samples)
        if len(samples) < 2:
            return None

        latest_time = samples[-1].timestamp
        target_times = np.asarray(
            [latest_time - (frames - 1 - i) / fps for i in range(frames)],
            dtype=np.float64,
        )
        if samples[0].timestamp > target_times[0]:
            return None

        sample_times = [sample.timestamp for sample in samples]
        qpos = np.zeros((frames, 36), dtype=np.float32)
        for out_idx, target_time in enumerate(target_times):
            sample = samples[_nearest_index(sample_times, float(target_time))]
            frame_offset = int(round((latest_time - sample.timestamp) * float(tracker_fps)))
            root_frame = int(current_tracker_frame) - frame_offset
            qpos[out_idx, :3] = _nearest_root_position(
                reference_qpos_36,
                frame_index=root_frame,
                default_qpos=default_qpos_36,
            )
            qpos[out_idx, 3:7] = sample.root_quat_wxyz
            qpos[out_idx, 7:36] = sample.dof_pos
        return qpos

    def latest_qpos_36(
        self,
        *,
        current_tracker_frame: int,
        reference_qpos_36: np.ndarray,
        default_qpos_36: np.ndarray,
    ) -> np.ndarray | None:
        with self._lock:
            if not self._samples:
                return None
            sample = self._samples[-1]
        qpos = np.zeros((36,), dtype=np.float32)
        qpos[:3] = _nearest_root_position(
            reference_qpos_36,
            frame_index=int(current_tracker_frame),
            default_qpos=default_qpos_36,
        )
        qpos[3:7] = sample.root_quat_wxyz
        qpos[7:36] = sample.dof_pos
        return qpos


def _nearest_index(sorted_values: list[float], value: float) -> int:
    idx = bisect.bisect_left(sorted_values, value)
    if idx <= 0:
        return 0
    if idx >= len(sorted_values):
        return len(sorted_values) - 1
    before = sorted_values[idx - 1]
    after = sorted_values[idx]
    return idx - 1 if abs(value - before) <= abs(after - value) else idx


def _remote_key_mask(remote: Any) -> int:
    data = bytes(int(x) & 0xFF for x in remote)
    if len(data) < 4:
        return 0
    return int(struct.unpack("<H", data[2:4])[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the G1 realtime bridge using true lowstate joint/IMU history for diffusion replanning "
            "while publishing HoloMotion obs65 reference packets."
        )
    )
    parser.add_argument("--connect", default="tcp://127.0.0.1:5555")
    parser.add_argument(
        "--default-root-height",
        type=float,
        default=0.76,
        help="Root z used for the local hold qpos before any planner reference exists.",
    )
    parser.add_argument("--history-fps", type=float, default=30.0)
    parser.add_argument("--history-frames", type=int, default=10)
    parser.add_argument("--tracker-fps", type=float, default=50.0)
    parser.add_argument("--continuous", action="store_true", help="Run until interrupted.")
    parser.add_argument("--num-frames", type=int, default=300)
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
    parser.add_argument("--holomotion-config", required=True)
    parser.add_argument("--lowstate-topic", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--publish-bind", default=None, help="HoloMotion obs65 PUB bind URI, e.g. tcp://*:6001.")
    parser.add_argument("--publish-topic", default="obs65")
    parser.add_argument("--status-jsonl", default=None, help="Optional live bridge status JSONL path.")
    parser.add_argument("--sleep", action="store_true", help="Sleep at tracker fps while publishing obs65.")
    parser.add_argument("--status-interval", type=float, default=2.0)
    parser.add_argument(
        "--activation-mode",
        default="remote-b",
        choices=["remote-b", "none"],
        help="Gate planner rollout until the Unitree remote B button is pressed. Use 'none' to start immediately.",
    )
    parser.add_argument("--activation-key-bit", type=int, default=9, help="Unitree wireless_remote key bit used by remote-b activation.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.continuous and args.num_frames <= 0:
        raise ValueError("--num-frames must be positive unless --continuous is set")
    if args.history_frames <= 0:
        raise ValueError("--history-frames must be positive")
    if args.replan_remaining_frames < 0:
        raise ValueError("--replan-remaining-frames must be non-negative")
    condition_sequence = _load_condition_sequence(args)

    _dof_names, motor_indices, config_topic, default_qpos = _load_holomotion_mapping(
        args.holomotion_config,
        default_root_height=float(args.default_root_height),
    )
    lowstate_topic = str(args.lowstate_topic or config_topic)

    lowstate = LowStateHistory(topic=lowstate_topic, motor_indices=motor_indices)
    lowstate.start()
    client = RealtimeOrinBufferClient(
        RealtimeOrinBufferClientConfig(
            connect=args.connect,
            tracker_fps=float(args.tracker_fps),
            request_timeout_ms=int(args.timeout_ms),
        )
    )
    publisher = _LatestObsPublisher(args.publish_bind, topic=args.publish_topic)
    reference_frames: list[np.ndarray] = []
    real_frames: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    cursor = 0
    replan_index = 0
    condition_session_id = uuid.uuid4().hex
    interrupted = False
    last_status = 0.0

    def current_history(current_cursor: int) -> tuple[np.ndarray, str]:
        real_history = lowstate.history_qpos_36(
            frames=int(args.history_frames),
            fps=float(args.history_fps),
            tracker_fps=float(args.tracker_fps),
            current_tracker_frame=current_cursor,
            reference_qpos_36=client.buffer.qpos_36,
            default_qpos_36=default_qpos,
        )
        if real_history is not None:
            return real_history, "lowstate"
        age = lowstate.latest_age_seconds()
        age_text = "none" if age is None else f"{age * 1000.0:.1f}ms"
        raise RuntimeError(
            "Lowstate history is required for active realtime replanning, but no complete history window is available "
            f"(history_frames={int(args.history_frames)}, history_fps={float(args.history_fps)}, lowstate_age={age_text})."
        )

    def lowstate_history_ready(current_cursor: int) -> bool:
        return (
            lowstate.history_qpos_36(
                frames=int(args.history_frames),
                fps=float(args.history_fps),
                tracker_fps=float(args.tracker_fps),
                current_tracker_frame=current_cursor,
                reference_qpos_36=client.buffer.qpos_36,
                default_qpos_36=default_qpos,
            )
            is not None
        )

    def maybe_status(current_cursor: int) -> None:
        nonlocal last_status
        now = time.perf_counter()
        if now - last_status < float(args.status_interval):
            return
        last_status = now
        age = lowstate.latest_age_seconds()
        age_text = "none" if age is None else f"{age * 1000.0:.1f}ms"
        print(
            f"[real-bridge status] frame={current_cursor:05d} buffer={client.buffer.remaining(current_cursor):04d} "
            f"lowstate_age={age_text}",
            flush=True,
        )
        append_jsonl(
            args.status_jsonl,
            {
                "kind": "status",
                "tracker_frame": int(current_cursor),
                "buffer_remaining": int(client.buffer.remaining(current_cursor)),
                "lowstate_age_seconds": None if age is None else float(age),
            },
        )

    def publish_hold(frame_index: int) -> None:
        hold_qpos = lowstate.latest_hold_qpos_36(default_qpos_36=default_qpos)
        publisher.publish(_qpos_to_latest_obs(hold_qpos, fps=float(args.tracker_fps))[0], frame_index=frame_index)

    def wait_for_activation() -> None:
        if args.activation_mode == "none":
            return
        print(
            f"[real-bridge activation] waiting for remote key bit {int(args.activation_key_bit)}; "
            "holding latest lowstate reference",
            flush=True,
        )
        append_jsonl(
            args.status_jsonl,
            {
                "kind": "activation_wait",
                "activation_mode": str(args.activation_mode),
                "activation_key_bit": int(args.activation_key_bit),
            },
        )
        frame_index = 0
        activation_requested = False
        while True:
            if not activation_requested and lowstate.consume_remote_key_rising_edge(int(args.activation_key_bit)):
                activation_requested = True
                print(
                    f"[real-bridge activation] remote key received at hold_frame={frame_index}; "
                    "waiting for complete lowstate history",
                    flush=True,
                )
                append_jsonl(
                    args.status_jsonl,
                    {
                        "kind": "activation_requested",
                        "activation_mode": str(args.activation_mode),
                        "activation_key_bit": int(args.activation_key_bit),
                        "hold_frame": int(frame_index),
                        "lowstate_age_seconds": lowstate.latest_age_seconds(),
                    },
                )
            if activation_requested and lowstate_history_ready(0):
                print(f"[real-bridge activation] activated at hold_frame={frame_index}", flush=True)
                append_jsonl(
                    args.status_jsonl,
                    {
                        "kind": "activation",
                        "activation_mode": str(args.activation_mode),
                        "activation_key_bit": int(args.activation_key_bit),
                        "hold_frame": int(frame_index),
                        "lowstate_age_seconds": lowstate.latest_age_seconds(),
                    },
                )
                return
            publish_hold(frame_index)
            maybe_status(0)
            frame_index += 1
            time.sleep(1.0 / float(args.tracker_fps))

    def execute_frames(start_cursor: int, frames: int) -> None:
        chunk = client.buffer.slice(start_cursor, frames)
        latest_obs = _qpos_to_latest_obs(chunk, fps=float(args.tracker_fps))
        sampled_real: list[np.ndarray] = []
        for local_idx, obs in enumerate(latest_obs):
            frame_index = start_cursor + local_idx
            publisher.publish(obs, frame_index=frame_index)
            real_qpos = lowstate.latest_qpos_36(
                current_tracker_frame=frame_index,
                reference_qpos_36=client.buffer.qpos_36,
                default_qpos_36=default_qpos,
            )
            if real_qpos is not None:
                sampled_real.append(real_qpos)
            maybe_status(frame_index)
            if args.sleep:
                time.sleep(1.0 / float(args.tracker_fps))
        reference_frames.append(chunk)
        if sampled_real:
            real_frames.append(np.stack(sampled_real, axis=0).astype(np.float32, copy=False))

    def begin_replan(current_cursor: int) -> PendingReplan:
        nonlocal replan_index
        history, history_source = current_history(current_cursor)
        sequence = condition_sequence
        condition_index = int(replan_index)
        condition_source = "condition_sequence"
        started = time.perf_counter()
        client.begin_request(
            tracker_frame=current_cursor,
            qpos_36_history=history,
            history_fps=float(args.history_fps),
            metadata={
                "bridge": "holomotion_real",
                "history_source": history_source,
                "condition_sequence": sequence,
                "condition_index": int(condition_index),
                "condition_session_id": condition_session_id,
                "condition_source": condition_source,
                "audio_fps": float(args.audio_fps),
                "tracker_fps": float(args.tracker_fps),
                "audio_type": str(args.audio_type),
                "audio_feature_type": str(args.audio_feature_type),
                "condition_audio_step_frames": args.condition_audio_step_frames,
            },
        )
        replan_index += 1
        return PendingReplan(
            request_tracker_frame=int(current_cursor),
            history_source=history_source,
            condition_sequence=sequence,
            condition_index=int(condition_index),
            condition_source=condition_source,
            started=started,
        )

    def poll_replan(pending: PendingReplan, current_cursor: int) -> tuple[int, bool]:
        response = client.poll_response(timeout_ms=0)
        if response is None:
            return current_cursor, False
        latency = time.perf_counter() - pending.started
        elapsed_frames = int(current_cursor) - int(pending.request_tracker_frame)
        client.append_response(response, current_tracker_frame=current_cursor)
        transport = response.metadata.get("realtime_transport", {})
        event = {
            "kind": "replan",
            "plan_id": int(response.plan_id),
            "request_tracker_frame": int(response.request_tracker_frame),
            "append_tracker_frame": int(current_cursor),
            "history_source": pending.history_source,
            "latency_seconds": float(latency),
            "elapsed_tracker_frames": int(elapsed_frames),
            "buffer_frames": int(client.buffer.frames),
            "remaining_after_append": int(client.buffer.remaining(current_cursor)),
            "lowstate_age_seconds": lowstate.latest_age_seconds(),
            "timing_ms": response.metadata.get("timing_ms", {}),
            "transport_timing": transport,
            "prompt": response.prompt,
            "condition_sequence": pending.condition_sequence,
            "condition_index": int(pending.condition_index),
            "condition_source": pending.condition_source,
            "response_condition": response.metadata.get("realtime_condition"),
        }
        events.append(event)
        append_jsonl(args.status_jsonl, event)
        print(
            f"[real-bridge replan {response.plan_id:04d}] request={response.request_tracker_frame:05d} "
            f"append={current_cursor:05d} history={pending.history_source} "
            f"condition={pending.condition_source}:{pending.condition_index:04d} prompt={response.prompt!r} "
            f"latency={latency * 1000.0:.3f}ms "
            f"server={float(transport.get('server_plan_ms', response.planning_latency_seconds * 1000.0)):.3f}ms "
            f"elapsed={elapsed_frames} buffer={client.buffer.frames}",
            flush=True,
        )
        return current_cursor, True

    try:
        wait_for_activation()
        pending: PendingReplan | None = begin_replan(0)
        hold_frame = 0
        while pending is not None and client.buffer.remaining(cursor) <= 0:
            cursor, completed = poll_replan(pending, cursor)
            if completed:
                pending = None
                break
            publish_hold(hold_frame)
            maybe_status(cursor)
            hold_frame += 1
            if args.sleep:
                time.sleep(1.0 / float(args.tracker_fps))
            else:
                time.sleep(0.001)
        while args.continuous or cursor < int(args.num_frames):
            remaining_total = math.inf if args.continuous else int(args.num_frames) - cursor
            if pending is not None:
                cursor, completed = poll_replan(pending, cursor)
                if completed:
                    pending = None
                    remaining_total = math.inf if args.continuous else int(args.num_frames) - cursor
            remaining_buffer = client.buffer.remaining(cursor)
            if remaining_buffer <= 0:
                raise RuntimeError(f"Motion buffer underrun at tracker frame {cursor}")
            if pending is None and remaining_buffer <= int(args.replan_remaining_frames):
                pending = begin_replan(cursor)
            if remaining_total <= 0:
                break
            step_frames = 1 if args.continuous else min(1, int(remaining_total))
            if remaining_total <= remaining_buffer:
                step_frames = min(step_frames, int(remaining_total))
            execute_frames(cursor, step_frames)
            cursor += step_frames
    except KeyboardInterrupt:
        interrupted = True
        print(f"interrupted_at_frame={cursor}; saving partial real-bridge output", flush=True)
    finally:
        publisher.close()
        client.close()
        lowstate.close()

    reference = (
        np.concatenate(reference_frames, axis=0).astype(np.float32, copy=False)
        if reference_frames
        else np.zeros((0, 36), dtype=np.float32)
    )
    real = (
        np.concatenate(real_frames, axis=0).astype(np.float32, copy=False)
        if real_frames
        else np.zeros((0, 36), dtype=np.float32)
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        reference_qpos_36=reference,
        real_qpos_36=real,
        reference_latest_obs=(
            _qpos_to_latest_obs(reference, fps=float(args.tracker_fps))
            if reference.shape[0]
            else np.zeros((0, 65), dtype=np.float32)
        ),
        fps=np.asarray([float(args.tracker_fps)], dtype=np.float32),
        events=np.asarray([json.dumps(events, sort_keys=True)]),
        condition_sequence=np.asarray([condition_sequence], dtype=np.str_),
        continuous=np.asarray([bool(args.continuous)]),
        interrupted=np.asarray([bool(interrupted)]),
    )
    print(
        f"real_bridge_output={output.resolve()} reference_frames={reference.shape[0]} "
        f"real_frames={real.shape[0]} replans={len(events)}"
    )


if __name__ == "__main__":
    main()
