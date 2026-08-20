import numpy as np
import pytest

from omg.tracking.holomotion.runtime import (
    HoloMotionTrackerSession,
    infer_holomotion_obs_contract,
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
    def __init__(self, obs_schema_version=None):
        self.custom_metadata_map = {
            "joint_names": "left_hip_pitch_joint,right_hip_pitch_joint",
            "default_joint_pos": "0 0",
            "action_scale": "1 1",
            "joint_stiffness": "10 10",
            "joint_damping": "1 1",
        }
        if obs_schema_version is not None:
            self.custom_metadata_map["obs_schema_version"] = obs_schema_version


class FakeSession:
    def __init__(self, obs_dim=132 + 39 * 3, obs_schema_version=None):
        self.feeds = []
        self.obs_dim = obs_dim
        self.meta = FakeMeta(obs_schema_version)

    def get_modelmeta(self):
        return self.meta

    def get_inputs(self):
        return [FakeValue("obs", [1, self.obs_dim])]

    def get_outputs(self):
        return [FakeValue("actions", [1, 2])]

    def run(self, output_names, feed):
        self.feeds.append((output_names, feed))
        return [np.asarray([[0.25, -0.5]], dtype=np.float32)]


def test_infer_n_fut_frames():
    assert infer_n_fut_frames(132 + 39 * 5) == 5
    v1_2, context_length, n_fut_frames = infer_holomotion_obs_contract(132 + 39 * 10)
    assert (v1_2.version, context_length, n_fut_frames) == ("v1_2", 1, 10)
    v1_3, context_length, n_fut_frames = infer_holomotion_obs_contract(134 + 47 * 10)
    assert (v1_3.version, context_length, n_fut_frames) == ("v1_3", 1, 10)
    assert infer_holomotion_obs_schema(134 + 47 * 10) == (1, 10)
    assert infer_holomotion_obs_schema(
        32 * 132 + 10 * 39,
        context_length=32,
        n_fut_frames=10,
    ) == (32, 10)
    with pytest.raises(ValueError):
        infer_n_fut_frames(133)
    with pytest.raises(ValueError):
        infer_holomotion_obs_schema(32 * 132 + 10 * 39)
    with pytest.raises(ValueError, match="Unsupported HoloMotion observation schema version"):
        infer_holomotion_obs_contract(132 + 39 * 10, obs_schema_version="legacy")
    with pytest.raises(ValueError, match="does not match"):
        infer_holomotion_obs_contract(
            134 + 47 * 10,
            context_length=1,
            n_fut_frames=10,
            obs_schema_version="v1_2",
        )


def test_load_holomotion_metadata_from_fake_session():
    metadata = load_holomotion_metadata(FakeSession())
    assert metadata.joint_names == ["left_hip_pitch_joint", "right_hip_pitch_joint"]
    assert metadata.n_fut_frames == 3
    assert metadata.context_length == 1
    assert metadata.obs_schema_version == "v1_2"
    np.testing.assert_allclose(metadata.action_scale, [1.0, 1.0])


def test_load_v1_3_holomotion_metadata_from_fake_session():
    metadata = load_holomotion_metadata(FakeSession(obs_dim=134 + 47 * 10))
    assert metadata.n_fut_frames == 10
    assert metadata.context_length == 1
    assert metadata.obs_schema_version == "v1_3"


def test_metadata_schema_version_must_match_input_dimension():
    with pytest.raises(ValueError, match="does not match"):
        load_holomotion_metadata(
            FakeSession(obs_dim=134 + 47 * 10, obs_schema_version="v1_2")
        )


def test_tracker_session_runs_action_output():
    session = FakeSession()
    metadata = load_holomotion_metadata(session)
    tracker = HoloMotionTrackerSession(session, metadata)
    action = tracker.run(np.zeros((1, 132 + 39 * 3), dtype=np.float32))
    np.testing.assert_allclose(action, [0.25, -0.5])
    assert tracker.step_idx == 1
