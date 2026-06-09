import numpy as np
import pytest

from omg.tracking.holomotion.runtime import (
    HoloMotionTrackerSession,
    infer_holomotion_obs_schema,
    infer_n_fut_frames,
    load_holomotion_metadata,
)


class FakeValue:
    def __init__(self, name, shape, type="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type


class FakeMeta:
    custom_metadata_map = {
        "joint_names": "left_hip_pitch_joint,right_hip_pitch_joint",
        "default_joint_pos": "0 0",
        "action_scale": "1 1",
        "joint_stiffness": "10 10",
        "joint_damping": "1 1",
    }


class FakeSession:
    def __init__(self):
        self.feeds = []

    def get_modelmeta(self):
        return FakeMeta()

    def get_inputs(self):
        return [FakeValue("obs", [1, 132 + 39 * 3])]

    def get_outputs(self):
        return [FakeValue("actions", [1, 2])]

    def run(self, output_names, feed):
        self.feeds.append((output_names, feed))
        return [np.asarray([[0.25, -0.5]], dtype=np.float32)]


def test_infer_n_fut_frames():
    assert infer_n_fut_frames(132 + 39 * 5) == 5
    assert infer_holomotion_obs_schema(
        32 * 132 + 10 * 39,
        context_length=32,
        n_fut_frames=10,
    ) == (32, 10)
    with pytest.raises(ValueError):
        infer_n_fut_frames(133)
    with pytest.raises(ValueError):
        infer_holomotion_obs_schema(32 * 132 + 10 * 39)


def test_load_holomotion_metadata_from_fake_session():
    metadata = load_holomotion_metadata(FakeSession())
    assert metadata.joint_names == ["left_hip_pitch_joint", "right_hip_pitch_joint"]
    assert metadata.n_fut_frames == 3
    assert metadata.context_length == 1
    np.testing.assert_allclose(metadata.action_scale, [1.0, 1.0])


def test_tracker_session_runs_action_output():
    session = FakeSession()
    metadata = load_holomotion_metadata(session)
    tracker = HoloMotionTrackerSession(session, metadata)
    action = tracker.run(np.zeros((1, 132 + 39 * 3), dtype=np.float32))
    np.testing.assert_allclose(action, [0.25, -0.5])
    assert tracker.step_idx == 1
