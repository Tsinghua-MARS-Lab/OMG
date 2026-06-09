from __future__ import annotations

import numpy as np
import pytest

from omg.realtime.protocol import MotionPlanChunk, RobotStateRequest, decode_message


def _qpos(frames: int) -> np.ndarray:
    out = np.zeros((frames, 36), dtype=np.float32)
    out[:, 2] = 0.75
    out[:, 3] = 1.0
    return out


def test_robot_state_request_round_trip() -> None:
    request = RobotStateRequest(
        qpos_36_history=_qpos(10),
        history_fps=30.0,
        tracker_frame=123,
        buffer_remaining_frames=40,
        prompt="walk forward",
        metadata={"source": "unit-test"},
    )
    header, arrays = decode_message(*request.to_message())
    restored = RobotStateRequest.from_message(header, arrays)

    assert restored.request_id == request.request_id
    assert restored.tracker_frame == 123
    assert restored.buffer_remaining_frames == 40
    assert restored.prompt == "walk forward"
    np.testing.assert_allclose(restored.qpos_36_history, request.qpos_36_history)


def test_motion_plan_chunk_round_trip_with_features() -> None:
    plan = MotionPlanChunk(
        qpos_36=_qpos(60),
        motion_features=np.ones((60, 8), dtype=np.float32),
        fps=30.0,
        request_id="req-1",
        plan_id=2,
        request_tracker_frame=50,
        planning_latency_seconds=0.07,
        prompt="walk forward",
        metadata={"timing_ms": {"total_ms": 70.0}},
    )
    header, arrays = decode_message(*plan.to_message())
    restored = MotionPlanChunk.from_message(header, arrays)

    assert restored.request_id == "req-1"
    assert restored.plan_id == 2
    assert restored.request_tracker_frame == 50
    assert restored.planning_latency_seconds == pytest.approx(0.07)
    np.testing.assert_allclose(restored.qpos_36, plan.qpos_36)
    np.testing.assert_allclose(restored.motion_features, plan.motion_features)


def test_rejects_invalid_qpos_shape() -> None:
    with pytest.raises(ValueError, match="qpos_36_history"):
        RobotStateRequest(qpos_36_history=np.zeros((10, 35), dtype=np.float32), history_fps=30.0, tracker_frame=0)
