"""Human-reference/tracking motion metrics.

Implements:
- g_mpjpe: global mean per-joint position error.
- mpjpe: root-relative mean per-joint position error.
- e_vel / velocity_error: velocity reconstruction error.
- e_acc / acceleration_error: acceleration reconstruction error.
"""

from __future__ import annotations

import numpy as np


def _as_batched_positions(positions: np.ndarray, *, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(positions, dtype=np.float32)
    single = array.ndim == 3
    if single:
        array = array[None]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (T, J, 3) or (B, T, J, 3), got {tuple(array.shape)}")
    return array, single


def _validate_position_pair(pred_positions: np.ndarray, ref_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred, _ = _as_batched_positions(pred_positions, name="pred_positions")
    ref, _ = _as_batched_positions(ref_positions, name="ref_positions")
    if pred.shape != ref.shape:
        raise ValueError(
            f"pred_positions and ref_positions must have the same shape, got {tuple(pred.shape)} and {tuple(ref.shape)}"
        )
    return pred, ref


def _mean_l2_mm(delta: np.ndarray, *, unit_scale: float) -> float:
    error = np.linalg.norm(delta, axis=-1)
    return float(error.mean() * float(unit_scale))


def g_mpjpe(pred_positions: np.ndarray, ref_positions: np.ndarray, *, unit_scale: float = 1000.0) -> float:
    """Global mean per-joint position error in millimeters.

    Inputs must be body/joint positions with shape ``(T, J, 3)`` or
    ``(B, T, J, 3)``. The default ``unit_scale`` assumes positions are stored in
    meters and reports millimeters, matching humanoid tracking papers such as
    PHC/H2O/SONIC.
    """
    pred, ref = _validate_position_pair(pred_positions, ref_positions)
    return _mean_l2_mm(pred - ref, unit_scale=unit_scale)


def mpjpe(
    pred_positions: np.ndarray,
    ref_positions: np.ndarray,
    *,
    root_index: int = 0,
    unit_scale: float = 1000.0,
) -> float:
    """Root-relative mean per-joint position error in millimeters."""
    pred, ref = _validate_position_pair(pred_positions, ref_positions)
    num_joints = int(pred.shape[-2])
    if not 0 <= int(root_index) < num_joints:
        raise ValueError(f"root_index must be in [0, {num_joints}), got {root_index}")
    root_index = int(root_index)
    pred_rel = pred - pred[:, :, root_index : root_index + 1, :]
    ref_rel = ref - ref[:, :, root_index : root_index + 1, :]
    return _mean_l2_mm(pred_rel - ref_rel, unit_scale=unit_scale)


def e_vel(pred_positions: np.ndarray, ref_positions: np.ndarray, *, unit_scale: float = 1000.0) -> float:
    """Mean body/joint velocity error in millimeters per frame."""
    pred, ref = _validate_position_pair(pred_positions, ref_positions)
    if pred.shape[1] < 2:
        raise ValueError("velocity error requires at least 2 frames")
    pred_vel = np.diff(pred, axis=1)
    ref_vel = np.diff(ref, axis=1)
    return _mean_l2_mm(pred_vel - ref_vel, unit_scale=unit_scale)


def e_acc(pred_positions: np.ndarray, ref_positions: np.ndarray, *, unit_scale: float = 1000.0) -> float:
    """Mean body/joint acceleration error in millimeters per frame squared."""
    pred, ref = _validate_position_pair(pred_positions, ref_positions)
    if pred.shape[1] < 3:
        raise ValueError("acceleration error requires at least 3 frames")
    pred_acc = np.diff(pred, n=2, axis=1)
    ref_acc = np.diff(ref, n=2, axis=1)
    return _mean_l2_mm(pred_acc - ref_acc, unit_scale=unit_scale)


velocity_error = e_vel
acceleration_error = e_acc
