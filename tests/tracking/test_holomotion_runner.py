from types import SimpleNamespace

import numpy as np
import pytest

from omg.tracking.holomotion import runner as runner_module
from omg.tracking.holomotion.runner import HoloMotionRolloutRunner


class _Tracker:
    def run(self, _obs: np.ndarray) -> np.ndarray:
        return np.zeros(29, dtype=np.float32)


def test_run_reference_chunk_stops_after_requested_frame(monkeypatch) -> None:
    runner = object.__new__(HoloMotionRolloutRunner)
    runner.target_fps = 50.0
    runner.control_substeps = 10
    runner.action_clip = 10.0
    runner.initialized = True
    runner.metadata = SimpleNamespace(
        n_fut_frames=1,
        context_length=1,
        obs_schema_version="test",
        joint_names=tuple(str(index) for index in range(29)),
    )
    runner.model = object()
    runner.data = object()
    runner.g1_handles = {}
    runner.holomotion_handles = SimpleNamespace(onnx_to_g1=np.arange(29))
    runner.tracker = _Tracker()
    runner.last_action = np.zeros(29, dtype=np.float32)
    runner._obs_history = {}
    runner.executed_qpos_36 = []
    runner.reference_qpos_36 = []
    runner.actions = []
    runner.plan_cursor = []
    runner.plan_id = []
    runner.writer = None
    runner._clear_external_forces = lambda: None
    runner._apply_push_events = lambda _frame_idx, _events: None

    monkeypatch.setattr(
        runner_module,
        "precompute_reference_features",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        runner_module,
        "build_holomotion_obs",
        lambda **_kwargs: np.zeros(1, dtype=np.float32),
    )
    monkeypatch.setattr(
        runner_module,
        "apply_holomotion_action_pd",
        lambda **_kwargs: np.zeros(29, dtype=np.float32),
    )

    root_heights = iter((1.0, 0.8, 0.6, 0.4, 0.2))

    def extract_qpos(_data: object, _handles: dict) -> np.ndarray:
        qpos = np.zeros(36, dtype=np.float32)
        qpos[2] = next(root_heights)
        qpos[3] = 1.0
        return qpos

    monkeypatch.setattr(runner_module, "extract_g1_qpos", extract_qpos)
    reference = np.zeros((5, 36), dtype=np.float32)
    reference[:, 2] = 1.0
    reference[:, 3] = 1.0

    result = runner.run_reference_chunk(
        reference,
        reference_fps=50.0,
        stop_condition=lambda _cursor, executed, target, _action: bool(
            executed[2] < 0.5 * target[2]
        ),
        record_buffers=False,
    )

    assert result.frames == 4
    assert result.qpos_36.shape == (4, 36)
    assert len(runner.executed_qpos_36) == 4
    assert len(runner.reference_qpos_36) == 0
    assert len(runner.actions) == 0
    assert runner.executed_qpos_36[-1][2] == pytest.approx(0.4)
