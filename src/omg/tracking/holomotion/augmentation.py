from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PushEvent:
    start_frame: int
    duration_frames: int
    force_xyz: tuple[float, float, float]
    body_name: str = "pelvis"

    @property
    def end_frame(self) -> int:
        return int(self.start_frame) + int(self.duration_frames)

    def active_at(self, frame_idx: int) -> bool:
        return int(self.start_frame) <= int(frame_idx) < self.end_frame

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": int(self.start_frame),
            "duration_frames": int(self.duration_frames),
            "force_xyz": [float(v) for v in self.force_xyz],
            "body_name": str(self.body_name),
        }


def push_events_to_dicts(events: list[PushEvent] | None) -> list[dict[str, Any]]:
    return [event.to_dict() for event in events or []]


def make_random_push_events(
    *,
    num_frames: int,
    count: int,
    duration_frames: int,
    force_min: float,
    force_max: float,
    rng: np.random.Generator,
    body_name: str = "pelvis",
) -> list[PushEvent]:
    if count <= 0:
        return []
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if duration_frames <= 0:
        raise ValueError(f"duration_frames must be positive, got {duration_frames}")
    if force_min < 0.0 or force_max < 0.0 or force_min > force_max:
        raise ValueError(f"Invalid push force range: [{force_min}, {force_max}]")
    max_start = max(0, int(num_frames) - int(duration_frames))
    starts = rng.integers(0, max_start + 1, size=int(count))
    events: list[PushEvent] = []
    for start in starts.tolist():
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        magnitude = float(rng.uniform(float(force_min), float(force_max)))
        force_xyz = (magnitude * float(np.cos(angle)), magnitude * float(np.sin(angle)), 0.0)
        events.append(
            PushEvent(
                start_frame=int(start),
                duration_frames=int(duration_frames),
                force_xyz=force_xyz,
                body_name=str(body_name),
            )
        )
    return events
