from __future__ import annotations

import numpy as np
import pytest

from omg.realtime.motion_buffer import ExecutedHistoryBuffer, ReferenceMotionBuffer


def _qpos(frames: int) -> np.ndarray:
    out = np.zeros((frames, 36), dtype=np.float32)
    out[:, 0] = np.arange(frames, dtype=np.float32)
    out[:, 2] = 0.75
    out[:, 3] = 1.0
    return out


def test_reference_buffer_append_clip_and_slice() -> None:
    buffer = ReferenceMotionBuffer(target_fps=50.0)
    segment0 = buffer.append_plan(plan_id=0, qpos_36=_qpos(11), source_fps=50.0)
    segment1 = buffer.append_plan(plan_id=1, qpos_36=_qpos(11), source_fps=50.0, skip_frames=2)

    assert segment0.start == 0
    assert segment0.end == 11
    assert segment1.start == 11
    assert segment1.end == 20
    assert buffer.remaining(5) == 15
    assert buffer.slice(11, 2).shape == (2, 36)
    assert buffer.segment_for(12).plan_id == 1

    buffer.clip_future(12)
    assert buffer.frames == 12
    assert buffer.remaining(12) == 0
    assert buffer.segments_as_dicts()[-1]["end"] == 12


def test_reference_buffer_rejects_consumed_plan() -> None:
    buffer = ReferenceMotionBuffer(target_fps=50.0)
    with pytest.raises(RuntimeError, match="no remaining tracker frames"):
        buffer.append_plan(plan_id=0, qpos_36=_qpos(3), source_fps=50.0, skip_frames=3)


def test_executed_history_buffer_resamples_and_trims() -> None:
    history = ExecutedHistoryBuffer(target_fps=50.0, max_frames=5)
    history.append(_qpos(7), fps=50.0)

    assert history.frames == 5
    np.testing.assert_allclose(history.history(3)[:, 0], np.asarray([4.0, 5.0, 6.0], dtype=np.float32))

    with pytest.raises(RuntimeError, match="requires 6"):
        history.history(6)
