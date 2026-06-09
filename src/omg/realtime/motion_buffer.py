from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omg.realtime.protocol import QPOS_DIM
from omg.tracking.holomotion.reference import resample_qpos


def _coerce_qpos_36(value: Any, *, name: str, allow_empty: bool = False) -> np.ndarray:
    qpos = np.asarray(value, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != QPOS_DIM:
        raise ValueError(f"{name} must have shape (T,{QPOS_DIM}), got {qpos.shape}")
    if not allow_empty and qpos.shape[0] <= 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(qpos).all():
        raise ValueError(f"{name} contains non-finite values")
    return qpos.astype(np.float32, copy=False)


def _coerce_fps(value: float, *, name: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return fps


@dataclass(frozen=True)
class PlanSegment:
    plan_id: int
    start: int
    end: int
    source_fps: float
    source_frames: int
    skipped_tracker_frames: int = 0
    request_tracker_frame: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": int(self.plan_id),
            "start": int(self.start),
            "end": int(self.end),
            "source_fps": float(self.source_fps),
            "source_frames": int(self.source_frames),
            "skipped_tracker_frames": int(self.skipped_tracker_frames),
            "request_tracker_frame": self.request_tracker_frame,
            "metadata": dict(self.metadata),
        }


class ReferenceMotionBuffer:
    def __init__(self, *, target_fps: float) -> None:
        self.target_fps = _coerce_fps(target_fps, name="target_fps")
        self._qpos_36 = np.zeros((0, QPOS_DIM), dtype=np.float32)
        self._segments: list[PlanSegment] = []

    @property
    def qpos_36(self) -> np.ndarray:
        return self._qpos_36

    @property
    def frames(self) -> int:
        return int(self._qpos_36.shape[0])

    def append_plan(
        self,
        *,
        plan_id: int,
        qpos_36: np.ndarray,
        source_fps: float,
        skip_frames: int = 0,
        request_tracker_frame: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PlanSegment:
        qpos = _coerce_qpos_36(qpos_36, name="qpos_36")
        source_fps = _coerce_fps(source_fps, name="source_fps")
        skip = int(skip_frames)
        if skip < 0:
            raise ValueError(f"skip_frames must be non-negative, got {skip_frames}")
        reference = resample_qpos(qpos, source_fps=source_fps, target_fps=self.target_fps)
        if skip >= reference.shape[0]:
            raise RuntimeError(
                "Plan chunk has no remaining tracker frames after activation skip: "
                f"skip={skip}, horizon={reference.shape[0]}"
            )
        reference = reference[skip:]
        start = self.frames
        self._qpos_36 = np.concatenate([self._qpos_36, reference.astype(np.float32, copy=False)], axis=0)
        end = self.frames
        segment = PlanSegment(
            plan_id=int(plan_id),
            start=start,
            end=end,
            source_fps=source_fps,
            source_frames=int(qpos.shape[0]),
            skipped_tracker_frames=skip,
            request_tracker_frame=None if request_tracker_frame is None else int(request_tracker_frame),
            metadata=dict(metadata or {}),
        )
        self._segments.append(segment)
        return segment

    def clip_future(self, cursor: int) -> None:
        cursor = int(cursor)
        if cursor < 0 or cursor > self.frames:
            raise ValueError(f"cursor must be in [0, {self.frames}], got {cursor}")
        self._qpos_36 = self._qpos_36[:cursor].astype(np.float32, copy=False)
        clipped: list[PlanSegment] = []
        for segment in self._segments:
            if segment.start >= cursor:
                continue
            clipped.append(
                PlanSegment(
                    plan_id=segment.plan_id,
                    start=segment.start,
                    end=min(segment.end, cursor),
                    source_fps=segment.source_fps,
                    source_frames=segment.source_frames,
                    skipped_tracker_frames=segment.skipped_tracker_frames,
                    request_tracker_frame=segment.request_tracker_frame,
                    metadata=dict(segment.metadata),
                )
            )
        self._segments = [segment for segment in clipped if segment.end > segment.start]

    def remaining(self, cursor: int) -> int:
        cursor = int(cursor)
        if cursor < 0 or cursor > self.frames:
            raise ValueError(f"cursor must be in [0, {self.frames}], got {cursor}")
        return self.frames - cursor

    def slice(self, cursor: int, frames: int) -> np.ndarray:
        cursor = int(cursor)
        frames = int(frames)
        if frames < 0:
            raise ValueError(f"frames must be non-negative, got {frames}")
        end = cursor + frames
        if cursor < 0 or end > self.frames:
            raise ValueError(f"Requested buffer slice [{cursor}, {end}) outside [0, {self.frames})")
        return self._qpos_36[cursor:end].astype(np.float32, copy=True)

    def segment_for(self, cursor: int) -> PlanSegment | None:
        cursor = int(cursor)
        for segment in self._segments:
            if segment.start <= cursor < segment.end:
                return segment
        return None

    def segments_as_dicts(self) -> list[dict[str, Any]]:
        return [segment.as_dict() for segment in self._segments]


class ExecutedHistoryBuffer:
    def __init__(self, *, target_fps: float, max_frames: int) -> None:
        self.target_fps = _coerce_fps(target_fps, name="target_fps")
        self.max_frames = int(max_frames)
        if self.max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        self._qpos_36 = np.zeros((0, QPOS_DIM), dtype=np.float32)

    @property
    def qpos_36(self) -> np.ndarray:
        return self._qpos_36

    @property
    def frames(self) -> int:
        return int(self._qpos_36.shape[0])

    def append(self, qpos_36: np.ndarray, *, fps: float) -> None:
        qpos = _coerce_qpos_36(qpos_36, name="qpos_36")
        resampled = resample_qpos(qpos, source_fps=_coerce_fps(fps, name="fps"), target_fps=self.target_fps)
        updated = np.concatenate([self._qpos_36, resampled.astype(np.float32, copy=False)], axis=0)
        self._qpos_36 = updated[-self.max_frames :].astype(np.float32, copy=False)

    def history(self, frames: int | None = None) -> np.ndarray:
        count = self.max_frames if frames is None else int(frames)
        if count <= 0:
            raise ValueError(f"frames must be positive, got {frames}")
        if self.frames < count:
            raise RuntimeError(f"Executed history has {self.frames} frames, requires {count}")
        return self._qpos_36[-count:].astype(np.float32, copy=True)
