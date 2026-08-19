from __future__ import annotations

import mujoco
import numpy as np

from omg.tracking.holomotion.runtime import (
    build_g1_state_handles,
    g1_qvel_from_qpos_pair,
    g1_state_from_qpos_history,
    resolve_robot_xml,
    set_g1_state,
)


def _standing_qpos() -> np.ndarray:
    qpos = np.zeros(36, dtype=np.float32)
    qpos[2] = 0.8
    qpos[3] = 1.0
    return qpos


def test_g1_history_velocity_uses_mujoco_free_joint_and_joint_geometry():
    model = mujoco.MjModel.from_xml_path(str(resolve_robot_xml(None)))
    data = mujoco.MjData(model)
    handles = build_g1_state_handles(model)
    previous = _standing_qpos()
    current = previous.copy()
    current[0] += 0.1
    current[7] += 0.2

    qvel = g1_qvel_from_qpos_pair(
        model,
        handles,
        previous,
        current,
        dt=0.1,
    )
    np.testing.assert_allclose(qvel[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(qvel[1:6], 0.0, atol=1e-6)
    np.testing.assert_allclose(qvel[6], 2.0, atol=1e-6)
    np.testing.assert_allclose(qvel[7:], 0.0, atol=1e-6)

    set_g1_state(model, data, handles, current, qvel)
    np.testing.assert_allclose(data.qvel[:6], qvel[:6], atol=1e-6)
    np.testing.assert_allclose(data.qvel[handles["joint_dof_adr"]], qvel[6:], atol=1e-6)


def test_g1_state_from_history_uses_last_pose_and_finite_difference_velocity():
    model = mujoco.MjModel.from_xml_path(str(resolve_robot_xml(None)))
    handles = build_g1_state_handles(model)
    previous = _standing_qpos()
    current = previous.copy()
    current[0] += 0.1
    current[7] += 0.2

    qpos, qvel = g1_state_from_qpos_history(
        model,
        handles,
        np.stack([previous, current]),
        fps=10.0,
    )

    np.testing.assert_array_equal(qpos, current)
    np.testing.assert_allclose(qvel[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(qvel[6], 2.0, atol=1e-6)
