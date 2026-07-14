from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from omg.robots.g1.kinematics import G1Kinematics
from omg.motion.feature_codec import G1MotionFeatureCodec, MotionComponents
from omg.utils.rotation_conversions import standardize_quaternion


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[3] / path


class G1MotionRepresentation(nn.Module):
    def __init__(
        self,
        stats_path: str | Path = "assets/stats/g1_125d_stats.json",
        kinematics_path: str | Path = "assets/robots/g1/g1_kinematics.json",
        num_prev_states: int = 2,
        canonical_frame_idx: int | None = None,
        feat_dim: int = 123,
        sequence_length: int = 64,
        clip_std_min: float = 1e-6,
        rotation_representation: str = "quat",
        rot6d_gradient_mode: str = "vanilla",
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.stats_path = str(_resolve_repo_path(stats_path))
        self.kinematics = G1Kinematics(kinematics_path=kinematics_path)
        self.codec = G1MotionFeatureCodec(
            self.kinematics,
            num_prev_states=num_prev_states,
            canonical_frame_idx=canonical_frame_idx,
            rotation_representation=rotation_representation,
            rot6d_gradient_mode=rot6d_gradient_mode,
        )
        self.num_prev_states = num_prev_states
        self.rotation_representation = self.codec.rotation_representation
        self.rot6d_gradient_mode = self.codec.rot6d_gradient_mode
        self.canonical_frame_idx = self.codec.canonical_frame_idx
        self.sequence_length = int(sequence_length)
        self.is_motion_representation = True

        with open(self.stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32)
        default_root_pos = torch.tensor(stats["default_root_pos"], dtype=torch.float32)
        default_root_quat = torch.tensor(stats["default_root_quat"], dtype=torch.float32)
        default_joint_dof = torch.tensor(stats["default_joint_dof"], dtype=torch.float32)
        grounded_default_root_pos = self._ground_default_root_pos(
            default_root_pos=default_root_pos,
            default_root_quat=default_root_quat,
            default_joint_dof=default_joint_dof,
        )
        if mean.numel() != feat_dim or std.numel() != feat_dim:
            raise ValueError(
                f"Stats dim mismatch: expected {feat_dim}, got mean={mean.numel()}, std={std.numel()}"
            )
        std = std.clamp_min(clip_std_min)

        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self.register_buffer("default_root_pos_stats", default_root_pos, persistent=False)
        self.register_buffer("default_root_pos", grounded_default_root_pos, persistent=False)
        self.register_buffer("default_root_quat", default_root_quat, persistent=False)
        self.register_buffer("default_joint_dof", default_joint_dof, persistent=False)
        self.obs_indices_dict = None
        self.build_obs_indices_dict()

    def _ground_default_root_pos(
        self,
        default_root_pos: torch.Tensor,
        default_root_quat: torch.Tensor,
        default_joint_dof: torch.Tensor,
    ) -> torch.Tensor:
        qpos_36 = torch.cat(
            [
                default_root_pos.view(1, 1, 3),
                default_root_quat.view(1, 1, 4),
                default_joint_dof.view(1, 1, -1),
            ],
            dim=-1,
        )
        body_state = self.kinematics.forward_kinematics(qpos_36)
        sole_points, sole_radii = self.kinematics.get_sole_proxy_points(
            body_state["body_pos_w"], body_state["body_quat_w"]
        )
        sole_bottom = sole_points[..., 2] - sole_radii.view(1, 1, -1)
        grounded_root_pos = default_root_pos.clone()
        grounded_root_pos[2] = grounded_root_pos[2] - sole_bottom.min()
        return grounded_root_pos

    def build_obs_indices_dict(self):
        self.obs_indices_dict = dict(self.codec.feature_slices)

    def normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x)) / self.std.to(x)

    def denormalize_features(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x) + self.mean.to(x)

    def encode(self, batch: dict) -> torch.Tensor:
        if "motion_features" not in batch:
            raise KeyError("motion_features is required for G1MotionRepresentation.encode")
        return self.normalize_features(batch["motion_features"])

    def decode(self, x_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.denormalize_features(x_norm)
        components = self.codec.split_features(x)
        return {
            "root_pos_local": components.root_pos_local,
            "root_rot_local": components.root_rot_local_quat,
            "root_rot_local_quat": components.root_rot_local_quat,
            "joint_dof": components.joint_dof,
            "body_link_pos_local": components.body_link_pos_local,
        }

    def compose_qpos_36(
        self,
        decode_dict: dict[str, torch.Tensor],
        canon_root_pos: torch.Tensor,
        canon_root_quat: torch.Tensor,
    ) -> torch.Tensor:
        components = MotionComponents(
            root_pos_local=decode_dict["root_pos_local"],
            root_rot_local_quat=decode_dict["root_rot_local_quat"],
            joint_dof=decode_dict["joint_dof"],
            body_link_pos_local=decode_dict["body_link_pos_local"],
        )
        return self.codec.decode_to_world_qpos36(
            components,
            anchor_root_pos=canon_root_pos,
            anchor_root_quat=canon_root_quat,
        )

    def get_default_prev_qpos36(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        root_pos = self.default_root_pos.to(device=device, dtype=dtype).view(1, 1, 3).expand(batch_size, self.num_prev_states, -1)
        root_quat = self.default_root_quat.to(device=device, dtype=dtype).view(1, 1, 4).expand(batch_size, self.num_prev_states, -1)
        joint_dof = self.default_joint_dof.to(device=device, dtype=dtype).view(1, 1, -1).expand(batch_size, self.num_prev_states, -1)
        root_quat = standardize_quaternion(F.normalize(root_quat, dim=-1))
        return torch.cat([root_pos, root_quat, joint_dof], dim=-1)

    def get_motion_dim(self) -> int:
        return self.feat_dim

    def get_obs_indices(self, obs: str):
        return self.obs_indices_dict[obs]
