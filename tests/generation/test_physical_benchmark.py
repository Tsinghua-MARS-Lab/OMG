import numpy as np
import pytest
import torch

from omg.cli.generation.physical_benchmark import load_motion_qpos
from omg.generation.metrics import body_jerk_mean, contact_sliding_speed, foot_ground_error


def test_contact_sliding_speed_measures_contact_interval_xy_speed():
    sole_points = torch.tensor(
        [[[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    valid = torch.ones((1, 3), dtype=torch.bool)
    value = contact_sliding_speed(
        sole_points=sole_points,
        sole_radii=torch.zeros(1),
        valid=valid,
        fps=10.0,
        contact_height_threshold=0.03,
    )
    assert value.item() == pytest.approx(10.0)


def test_contact_sliding_speed_requires_contact_at_both_interval_ends():
    sole_points = torch.tensor(
        [[[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.1]], [[2.0, 0.0, 0.1]]]],
        dtype=torch.float32,
    )
    valid = torch.ones((1, 3), dtype=torch.bool)
    value = contact_sliding_speed(
        sole_points=sole_points,
        sole_radii=torch.zeros(1),
        valid=valid,
        fps=10.0,
        contact_height_threshold=0.03,
    )
    assert value.item() == pytest.approx(0.0)



def test_contact_sliding_speed_excludes_deep_penetration():
    sole_points = torch.tensor(
        [[[[0.0, 0.0, -0.10]], [[1.0, 0.0, -0.10]], [[2.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    valid = torch.ones((1, 3), dtype=torch.bool)
    value = contact_sliding_speed(
        sole_points=sole_points,
        sole_radii=torch.zeros(1),
        valid=valid,
        fps=10.0,
        contact_height_threshold=0.03,
        contact_penetration_tolerance=0.02,
    )
    assert value.item() == pytest.approx(0.0)


def test_foot_ground_error_measures_penetration_and_hover():
    sole_points = torch.tensor(
        [[[[0.0, 0.0, -0.02]], [[0.0, 0.0, 0.01]], [[0.0, 0.0, 0.04]]]],
        dtype=torch.float32,
    )
    valid = torch.ones((1, 3), dtype=torch.bool)
    value = foot_ground_error(
        sole_points=sole_points,
        sole_radii=torch.zeros(1),
        valid=valid,
    )
    assert value.item() == pytest.approx((0.02 + 0.01 + 0.04) / 3.0)


def test_body_jerk_mean_matches_cubic_motion():
    body_pos = torch.zeros((1, 5, 2, 3), dtype=torch.float32)
    t = torch.arange(5, dtype=torch.float32)
    body_pos[..., 0] = t.view(1, 5, 1).pow(3)
    valid = torch.ones((1, 5), dtype=torch.bool)
    value = body_jerk_mean(body_pos_w=body_pos, valid=valid, fps=2.0)
    assert value.item() == pytest.approx(48.0)


def test_load_motion_qpos_prefers_executed_npz_key(tmp_path):
    path = tmp_path / "motion.npz"
    executed = np.ones((4, 36), dtype=np.float32)
    reference = np.zeros((4, 36), dtype=np.float32)
    np.savez(path, executed_qpos_36=executed, reference_qpos_36=reference, fps=np.array([50.0], dtype=np.float32))

    qpos, key, fps = load_motion_qpos(path, qpos_key=None)

    assert key == "executed_qpos_36"
    assert fps == pytest.approx(50.0)
    np.testing.assert_array_equal(qpos, executed)
