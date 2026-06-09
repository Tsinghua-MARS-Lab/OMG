from __future__ import annotations

import io
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

MESSAGE_VERSION = 1
KIND_ROBOT_STATE_REQUEST = "robot_state_request"
KIND_MOTION_PLAN_CHUNK = "motion_plan_chunk"
QPOS_DIM = 36


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coerce_qpos_36(value: Any, *, name: str) -> np.ndarray:
    qpos = np.asarray(value, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != QPOS_DIM:
        raise ValueError(f"{name} must have shape (T,{QPOS_DIM}), got {qpos.shape}")
    if qpos.shape[0] <= 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(qpos).all():
        raise ValueError(f"{name} contains non-finite values")
    return qpos.astype(np.float32, copy=False)


def _coerce_fps(value: float, *, name: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return fps


def _coerce_non_negative_int(value: int, *, name: str) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return out


def _request_id(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        return uuid.uuid4().hex
    return str(value)


def encode_message(header: Mapping[str, Any], arrays: Mapping[str, np.ndarray] | None = None) -> tuple[bytes, bytes]:
    payload = io.BytesIO()
    np.savez_compressed(payload, **{str(key): np.asarray(value) for key, value in (arrays or {}).items()})
    header_bytes = json.dumps(_jsonable(dict(header)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return header_bytes, payload.getvalue()


def decode_message(header_bytes: bytes, payload_bytes: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    header = json.loads(header_bytes.decode("utf-8"))
    version = int(header.get("version", -1))
    if version != MESSAGE_VERSION:
        raise ValueError(f"Unsupported realtime protocol version {version}; expected {MESSAGE_VERSION}")
    with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    return header, arrays


@dataclass(frozen=True)
class RobotStateRequest:
    qpos_36_history: np.ndarray
    history_fps: float
    tracker_frame: int
    buffer_remaining_frames: int = 0
    request_id: str | None = None
    last_plan_id: int | None = None
    prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "qpos_36_history", _coerce_qpos_36(self.qpos_36_history, name="qpos_36_history"))
        object.__setattr__(self, "history_fps", _coerce_fps(self.history_fps, name="history_fps"))
        object.__setattr__(self, "tracker_frame", _coerce_non_negative_int(self.tracker_frame, name="tracker_frame"))
        object.__setattr__(
            self,
            "buffer_remaining_frames",
            _coerce_non_negative_int(self.buffer_remaining_frames, name="buffer_remaining_frames"),
        )
        object.__setattr__(self, "request_id", _request_id(self.request_id))
        if self.last_plan_id is not None:
            object.__setattr__(self, "last_plan_id", int(self.last_plan_id))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_message(self) -> tuple[bytes, bytes]:
        header = {
            "version": MESSAGE_VERSION,
            "kind": KIND_ROBOT_STATE_REQUEST,
            "created_time": time.time(),
            "request_id": self.request_id,
            "history_fps": self.history_fps,
            "tracker_frame": self.tracker_frame,
            "buffer_remaining_frames": self.buffer_remaining_frames,
            "last_plan_id": self.last_plan_id,
            "prompt": self.prompt,
            "metadata": self.metadata,
        }
        arrays = {"qpos_36_history": self.qpos_36_history}
        return encode_message(header, arrays)

    @classmethod
    def from_message(cls, header: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> "RobotStateRequest":
        if header.get("kind") != KIND_ROBOT_STATE_REQUEST:
            raise ValueError(f"Expected message kind {KIND_ROBOT_STATE_REQUEST}, got {header.get('kind')}")
        if "qpos_36_history" not in arrays:
            raise KeyError("robot_state_request payload is missing qpos_36_history")
        return cls(
            qpos_36_history=arrays["qpos_36_history"],
            history_fps=float(header["history_fps"]),
            tracker_frame=int(header["tracker_frame"]),
            buffer_remaining_frames=int(header.get("buffer_remaining_frames", 0)),
            request_id=str(header["request_id"]),
            last_plan_id=header.get("last_plan_id"),
            prompt=header.get("prompt"),
            metadata=dict(header.get("metadata") or {}),
        )


@dataclass(frozen=True)
class MotionPlanChunk:
    qpos_36: np.ndarray
    fps: float
    request_id: str
    plan_id: int
    request_tracker_frame: int
    motion_features: np.ndarray | None = None
    planning_latency_seconds: float = 0.0
    prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "qpos_36", _coerce_qpos_36(self.qpos_36, name="qpos_36"))
        object.__setattr__(self, "fps", _coerce_fps(self.fps, name="fps"))
        object.__setattr__(self, "request_id", _request_id(self.request_id))
        object.__setattr__(self, "plan_id", int(self.plan_id))
        if self.plan_id < 0:
            raise ValueError(f"plan_id must be non-negative, got {self.plan_id}")
        object.__setattr__(
            self,
            "request_tracker_frame",
            _coerce_non_negative_int(self.request_tracker_frame, name="request_tracker_frame"),
        )
        latency = float(self.planning_latency_seconds)
        if not np.isfinite(latency) or latency < 0.0:
            raise ValueError(f"planning_latency_seconds must be non-negative and finite, got {latency}")
        object.__setattr__(self, "planning_latency_seconds", latency)
        if self.motion_features is not None:
            features = np.asarray(self.motion_features, dtype=np.float32)
            if features.ndim != 2 or features.shape[0] != self.qpos_36.shape[0]:
                raise ValueError(
                    "motion_features must have shape (T,F) matching qpos_36 frames, "
                    f"got {features.shape} for qpos {self.qpos_36.shape}"
                )
            if not np.isfinite(features).all():
                raise ValueError("motion_features contains non-finite values")
            object.__setattr__(self, "motion_features", features.astype(np.float32, copy=False))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_message(self) -> tuple[bytes, bytes]:
        arrays = {"qpos_36": self.qpos_36}
        if self.motion_features is not None:
            arrays["motion_features"] = self.motion_features
        header = {
            "version": MESSAGE_VERSION,
            "kind": KIND_MOTION_PLAN_CHUNK,
            "created_time": time.time(),
            "request_id": self.request_id,
            "plan_id": self.plan_id,
            "request_tracker_frame": self.request_tracker_frame,
            "fps": self.fps,
            "planning_latency_seconds": self.planning_latency_seconds,
            "prompt": self.prompt,
            "metadata": self.metadata,
        }
        return encode_message(header, arrays)

    @classmethod
    def from_message(cls, header: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> "MotionPlanChunk":
        if header.get("kind") != KIND_MOTION_PLAN_CHUNK:
            raise ValueError(f"Expected message kind {KIND_MOTION_PLAN_CHUNK}, got {header.get('kind')}")
        if "qpos_36" not in arrays:
            raise KeyError("motion_plan_chunk payload is missing qpos_36")
        return cls(
            qpos_36=arrays["qpos_36"],
            motion_features=arrays.get("motion_features"),
            fps=float(header["fps"]),
            request_id=str(header["request_id"]),
            plan_id=int(header["plan_id"]),
            request_tracker_frame=int(header["request_tracker_frame"]),
            planning_latency_seconds=float(header.get("planning_latency_seconds", 0.0)),
            prompt=header.get("prompt"),
            metadata=dict(header.get("metadata") or {}),
        )
