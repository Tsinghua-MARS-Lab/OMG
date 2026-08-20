from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omg.tracking.holomotion.reference import (
    build_holomotion_obs,
    current_robot_obs_terms,
    future_indices,
    precompute_reference_features,
    yaw_from_quat_wxyz,
)


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    xyzw = Rotation.from_euler("z", yaw).as_quat().astype(np.float32)
    return xyzw[[3, 0, 1, 2]]


def _fixture():
    frames = 12
    qpos_36 = np.zeros((frames, 36), dtype=np.float32)
    qpos_36[:, 2] = 0.8
    qpos_36[:, 3:7] = np.stack([_yaw_quat_wxyz(0.1 * idx) for idx in range(frames)])
    qpos_36[:, 7:] = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) / 100.0
    ref_features = precompute_reference_features(qpos_36, fps=50.0, onnx_to_g1=np.arange(29))

    data = SimpleNamespace(
        qpos=np.linspace(-0.2, 0.2, 29, dtype=np.float32),
        qvel=np.linspace(0.3, -0.3, 29, dtype=np.float32),
        xquat=np.asarray([_yaw_quat_wxyz(0.25)], dtype=np.float32),
        sensordata=np.asarray([0.4, -0.5, 0.6], dtype=np.float32),
    )
    g1_handles = {
        "joint_qpos_adr": np.arange(29),
        "joint_dof_adr": np.arange(29),
        "pelvis_body_id": 0,
        "pelvis_gyro_adr": 0,
        "pelvis_gyro_dim": 3,
    }
    holomotion_handles = SimpleNamespace(
        onnx_to_g1=np.arange(29),
        default_joint_pos=np.linspace(0.1, -0.1, 29, dtype=np.float32),
    )
    last_action = np.linspace(-1.0, 1.0, 29, dtype=np.float32)
    return ref_features, data, g1_handles, holomotion_handles, last_action


def test_v1_2_observation_preserves_existing_layout():
    ref_features, data, g1_handles, holomotion_handles, last_action = _fixture()
    frame_idx = 1
    n_fut_frames = 10
    obs = build_holomotion_obs(
        ref_features,
        frame_idx,
        n_fut_frames,
        data,
        g1_handles,
        holomotion_handles,
        last_action,
        obs_schema_version="v1_2",
    )

    projected_gravity, root_ang_vel, dof_pos, dof_vel = current_robot_obs_terms(
        data, g1_handles, holomotion_handles
    )
    fut_idx = future_indices(frame_idx, len(ref_features["qpos_g1"]), n_fut_frames)
    expected = np.concatenate(
        [
            ref_features["ref_gravity_projection"][frame_idx],
            ref_features["ref_base_linvel"][frame_idx],
            ref_features["ref_base_angvel"][frame_idx],
            ref_features["ref_dof_pos_onnx"][frame_idx],
            np.asarray([ref_features["ref_root_height"][frame_idx]], dtype=np.float32),
            projected_gravity,
            root_ang_vel,
            dof_pos,
            dof_vel,
            last_action,
            ref_features["ref_dof_pos_onnx"][fut_idx].reshape(-1),
            ref_features["ref_root_height"][fut_idx].reshape(-1),
            ref_features["ref_gravity_projection"][fut_idx].reshape(-1),
            ref_features["ref_base_linvel"][fut_idx].reshape(-1),
            ref_features["ref_base_angvel"][fut_idx].reshape(-1),
        ]
    )[None, :]
    assert obs.shape == (1, 522)
    np.testing.assert_array_equal(obs, expected)

    without_v1_3_feature = dict(ref_features)
    del without_v1_3_feature["ref_root_quat_wxyz"]
    np.testing.assert_array_equal(
        build_holomotion_obs(
            without_v1_3_feature,
            frame_idx,
            n_fut_frames,
            data,
            g1_handles,
            holomotion_handles,
            last_action,
            obs_schema_version="v1_2",
        ),
        expected,
    )


def test_v1_3_observation_adds_heading_contract():
    ref_features, data, g1_handles, holomotion_handles, last_action = _fixture()
    frame_idx = 1
    n_fut_frames = 10
    obs = build_holomotion_obs(
        ref_features,
        frame_idx,
        n_fut_frames,
        data,
        g1_handles,
        holomotion_handles,
        last_action,
        obs_schema_version="v1_3",
    )
    assert obs.shape == (1, 604)

    reference_yaw = yaw_from_quat_wxyz(ref_features["ref_root_quat_wxyz"][frame_idx])
    robot_yaw = yaw_from_quat_wxyz(data.xquat[0])
    np.testing.assert_allclose(
        obs[0, 39:41],
        [np.sin(reference_yaw - robot_yaw), np.cos(reference_yaw - robot_yaw)],
        atol=1.0e-6,
    )

    fut_idx = future_indices(frame_idx, len(ref_features["qpos_g1"]), n_fut_frames)
    future_yaw = yaw_from_quat_wxyz(ref_features["ref_root_quat_wxyz"][fut_idx])
    future_heading_offset = 134 + 390
    np.testing.assert_allclose(
        obs[0, future_heading_offset : future_heading_offset + 20].reshape(10, 2),
        np.stack(
            [np.sin(future_yaw - reference_yaw), np.cos(future_yaw - reference_yaw)],
            axis=-1,
        ),
        atol=1.0e-6,
    )


def test_observation_rejects_unknown_schema_version():
    ref_features, data, g1_handles, holomotion_handles, last_action = _fixture()
    with pytest.raises(ValueError, match="Unsupported HoloMotion observation schema version"):
        build_holomotion_obs(
            ref_features,
            0,
            10,
            data,
            g1_handles,
            holomotion_handles,
            last_action,
            obs_schema_version="legacy",
        )
