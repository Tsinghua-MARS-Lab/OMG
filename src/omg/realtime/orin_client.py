from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from omg.realtime.motion_buffer import ReferenceMotionBuffer
from omg.realtime.protocol import MotionPlanChunk, RobotStateRequest
from omg.realtime.transport import ZmqPlanClient


@dataclass(frozen=True)
class RealtimeOrinBufferClientConfig:
    connect: str
    tracker_fps: float = 50.0
    request_timeout_ms: int = 30000


class RealtimeOrinBufferClient:
    """Planner-side buffer helper for a HoloMotion deployment loop.

    This class does not send robot commands. The deployment loop owns HoloMotion
    policy execution and uses this helper to request future motion chunks and
    splice returned plans into a tracker-fps reference buffer.
    """

    def __init__(self, config: RealtimeOrinBufferClientConfig) -> None:
        self.config = config
        self.buffer = ReferenceMotionBuffer(target_fps=config.tracker_fps)
        self.transport = ZmqPlanClient(config.connect)
        self.last_plan_id: int | None = None

    def request_plan(
        self,
        *,
        tracker_frame: int,
        qpos_36_history: np.ndarray,
        history_fps: float,
        prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MotionPlanChunk:
        request = RobotStateRequest(
            qpos_36_history=qpos_36_history,
            history_fps=history_fps,
            tracker_frame=int(tracker_frame),
            buffer_remaining_frames=self.buffer.remaining(int(tracker_frame)),
            last_plan_id=self.last_plan_id,
            prompt=prompt,
            metadata=dict(metadata or {}),
        )
        return self.transport.request_plan(request, timeout_ms=self.config.request_timeout_ms)

    def begin_request(
        self,
        *,
        tracker_frame: int,
        qpos_36_history: np.ndarray,
        history_fps: float,
        prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RobotStateRequest:
        request = RobotStateRequest(
            qpos_36_history=qpos_36_history,
            history_fps=history_fps,
            tracker_frame=int(tracker_frame),
            buffer_remaining_frames=self.buffer.remaining(int(tracker_frame)),
            last_plan_id=self.last_plan_id,
            prompt=prompt,
            metadata=dict(metadata or {}),
        )
        self.transport.begin_request(request)
        return request

    def poll_response(self, *, timeout_ms: int = 0) -> MotionPlanChunk | None:
        return self.transport.poll_plan(timeout_ms=timeout_ms)

    def append_response(self, response: MotionPlanChunk, *, current_tracker_frame: int) -> None:
        current = int(current_tracker_frame)
        if current < response.request_tracker_frame:
            raise ValueError(
                "current_tracker_frame cannot precede the frame that launched the replan: "
                f"{current} < {response.request_tracker_frame}"
            )
        self.buffer.clip_future(current)
        self.buffer.append_plan(
            plan_id=response.plan_id,
            qpos_36=response.qpos_36,
            source_fps=response.fps,
            skip_frames=current - response.request_tracker_frame,
            request_tracker_frame=response.request_tracker_frame,
            metadata=response.metadata,
        )
        self.last_plan_id = response.plan_id

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> "RealtimeOrinBufferClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
