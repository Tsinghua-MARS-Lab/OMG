from __future__ import annotations

import torch
import torch.nn.functional as F

from omg.robots.g1.kinematics import G1Kinematics
from omg.utils.rotation_conversions import (
    axis_angle_to_quaternion,
    quaternion_apply,
    quaternion_invert,
    quaternion_to_matrix,
    standardize_quaternion,
)


def heading_inverse_from_root_quat(root_quat_wxyz: torch.Tensor) -> torch.Tensor:
    root_quat = standardize_quaternion(F.normalize(root_quat_wxyz, dim=-1))
    root_rot = quaternion_to_matrix(root_quat)
    yaw = torch.atan2(root_rot[..., 1, 0], root_rot[..., 0, 0])
    axis_angle = torch.zeros(*yaw.shape, 3, device=root_quat.device, dtype=root_quat.dtype)
    axis_angle[..., 2] = yaw
    heading_quat = standardize_quaternion(F.normalize(axis_angle_to_quaternion(axis_angle), dim=-1))
    return quaternion_invert(heading_quat)


def canonical_body_positions(
    body_pos_w: torch.Tensor,
    anchor_root_pos: torch.Tensor,
    anchor_root_quat: torch.Tensor,
) -> torch.Tensor:
    if body_pos_w.shape[-1] != 3:
        raise ValueError(f"Expected body_pos_w last dim 3, got {tuple(body_pos_w.shape)}")
    if anchor_root_pos.shape[-1] != 3:
        raise ValueError(f"Expected anchor_root_pos last dim 3, got {tuple(anchor_root_pos.shape)}")
    if anchor_root_quat.shape[-1] != 4:
        raise ValueError(f"Expected anchor_root_quat last dim 4, got {tuple(anchor_root_quat.shape)}")
    while anchor_root_pos.ndim < body_pos_w.ndim:
        anchor_root_pos = anchor_root_pos.unsqueeze(-2)
    while anchor_root_quat.ndim < body_pos_w.ndim:
        anchor_root_quat = anchor_root_quat.unsqueeze(-2)
    heading_inv = heading_inverse_from_root_quat(anchor_root_quat)
    heading_inv = heading_inv.expand(*body_pos_w.shape[:-1], 4)
    return quaternion_apply(heading_inv, body_pos_w - anchor_root_pos)


def canonical_body_positions_from_qpos(
    qpos_36: torch.Tensor,
    kinematics: G1Kinematics,
    *,
    anchor_frame: int = 0,
) -> torch.Tensor:
    if qpos_36.shape[-1] != 36:
        raise ValueError(f"Expected qpos_36 last dim 36, got {tuple(qpos_36.shape)}")
    if qpos_36.ndim < 2:
        raise ValueError(f"Expected qpos_36 with a frame dimension, got {tuple(qpos_36.shape)}")
    num_frames = qpos_36.shape[-2]
    if not 0 <= int(anchor_frame) < num_frames:
        raise ValueError(f"anchor_frame must be in [0, {num_frames}), got {anchor_frame}")
    anchor_slice = slice(int(anchor_frame), int(anchor_frame) + 1)
    fk = kinematics.forward_kinematics(qpos_36)
    return canonical_body_positions(
        fk["body_pos_w"],
        qpos_36[..., anchor_slice, :3],
        qpos_36[..., anchor_slice, 3:7],
    )


def motion_input_dim(motion_key: str, *, kinematics_path: str = "assets/robots/g1/g1_kinematics.json") -> int:
    if motion_key == "qpos_36":
        return 36
    kinematics = G1Kinematics(kinematics_path=kinematics_path)
    if motion_key == "motion_features":
        return 3 + 4 + kinematics.num_joints + (kinematics.num_bodies - 1) * 3
    if motion_key == "body_link_pos_local":
        return (kinematics.num_bodies - 1) * 3
    if motion_key == "body_pos_local":
        return kinematics.num_bodies * 3
    raise ValueError(
        f"Unsupported evaluator motion_key={motion_key!r}. "
        "Expected one of: qpos_36, motion_features, body_link_pos_local, body_pos_local."
    )
