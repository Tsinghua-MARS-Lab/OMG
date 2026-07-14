from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from omg.robots.g1.kinematics import G1Kinematics
from omg.utils.rotation_conversions import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    rotation_6d_to_matrix_canonical_gradient,
    standardize_quaternion,
)


@dataclass
class MotionComponents:
    root_pos_local: torch.Tensor
    root_rot_local_quat: torch.Tensor
    joint_dof: torch.Tensor
    body_link_pos_local: torch.Tensor


class G1MotionFeatureCodec:
    def __init__(
        self,
        kinematics: G1Kinematics,
        num_prev_states: int = 2,
        canonical_frame_idx: int | None = None,
        rotation_representation: str = "quat",
        rot6d_gradient_mode: str = "vanilla",
    ):
        self.kinematics = kinematics
        self.num_prev_states = int(num_prev_states)
        if canonical_frame_idx is None:
            canonical_frame_idx = self.num_prev_states - 1
        self.canonical_frame_idx = int(canonical_frame_idx)
        if self.canonical_frame_idx < 0 or self.canonical_frame_idx >= self.num_prev_states:
            raise ValueError(
                f"canonical_frame_idx must be in [0, {self.num_prev_states}), got {self.canonical_frame_idx}"
            )
        self.num_body_links = self.kinematics.num_bodies - 1
        self.rotation_representation = self._normalize_rotation_representation(rotation_representation)
        self.rot6d_gradient_mode = self._normalize_rot6d_gradient_mode(rot6d_gradient_mode)
        self.root_rot_dim = {"quat": 4, "rot6d": 6}[self.rotation_representation]
        self.feature_dim = 3 + self.root_rot_dim + self.kinematics.num_joints + self.num_body_links * 3
        joint_start = 3 + self.root_rot_dim
        self.feature_slices = {
            "root_pos_local": (0, 3),
            "root_rot_local": (3, joint_start),
            "root_rot_local_quat": (3, joint_start),
            "joint_dof": (joint_start, joint_start + self.kinematics.num_joints),
            "body_link_pos_local": (
                joint_start + self.kinematics.num_joints,
                self.feature_dim,
            ),
        }

    @staticmethod
    def _normalize_rotation_representation(value: str) -> str:
        key = str(value).strip().lower().replace("-", "_")
        aliases = {
            "quat": "quat",
            "quaternion": "quat",
            "quat4": "quat",
            "rot6d": "rot6d",
            "rotation_6d": "rot6d",
            "6d": "rot6d",
        }
        if key not in aliases:
            raise ValueError(f"Unsupported rotation_representation={value!r}; expected quat or rot6d")
        return aliases[key]

    @staticmethod
    def _normalize_rot6d_gradient_mode(value: str) -> str:
        key = str(value).strip().lower().replace("-", "_")
        aliases = {
            "vanilla": "vanilla",
            "autograd": "vanilla",
            "gram_schmidt": "vanilla",
            "canonical": "canonical",
            "canonical_gradient": "canonical",
            "canonical_section": "canonical",
        }
        if key not in aliases:
            raise ValueError(
                f"Unsupported rot6d_gradient_mode={value!r}; expected vanilla or canonical"
            )
        return aliases[key]

    def _standardize_quat(self, quat: torch.Tensor) -> torch.Tensor:
        return standardize_quaternion(F.normalize(quat, dim=-1))

    def _build_heading_quat(self, root_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        root_quat = F.normalize(root_quat, dim=-1)
        root_quat = standardize_quaternion(root_quat)
        root_rot = quaternion_to_matrix(root_quat)
        yaw = torch.atan2(root_rot[..., 1, 0], root_rot[..., 0, 0])
        axis_angle = torch.zeros(*yaw.shape, 3, device=root_quat.device, dtype=root_quat.dtype)
        axis_angle[..., 2] = yaw
        heading_quat = axis_angle_to_quaternion(axis_angle)
        heading_quat = F.normalize(heading_quat, dim=-1)
        heading_quat = standardize_quaternion(heading_quat)
        heading_quat_inv = quaternion_invert(heading_quat)
        return heading_quat, heading_quat_inv

    def _canonical_anchor(
        self,
        anchor_root_pos: torch.Tensor,
        anchor_root_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if anchor_root_pos.ndim == 2:
            anchor_root_pos = anchor_root_pos.unsqueeze(-2)
        if anchor_root_quat.ndim == 2:
            anchor_root_quat = anchor_root_quat.unsqueeze(-2)
        anchor_root_quat = F.normalize(anchor_root_quat, dim=-1)
        anchor_root_quat = standardize_quaternion(anchor_root_quat)
        heading_quat, heading_quat_inv = self._build_heading_quat(anchor_root_quat)
        return anchor_root_pos, heading_quat, heading_quat_inv

    def rotation_quat_to_features(self, quat: torch.Tensor) -> torch.Tensor:
        quat = self._standardize_quat(quat)
        if self.rotation_representation == "quat":
            return quat
        if self.rotation_representation == "rot6d":
            return matrix_to_rotation_6d(quaternion_to_matrix(quat))
        raise AssertionError(f"Unhandled rotation_representation={self.rotation_representation}")

    def rotation_features_to_quat(self, features: torch.Tensor) -> torch.Tensor:
        if self.rotation_representation == "quat":
            return self._standardize_quat(features)
        if self.rotation_representation == "rot6d":
            if self.rot6d_gradient_mode == "canonical":
                rotation = rotation_6d_to_matrix_canonical_gradient(features)
            else:
                rotation = rotation_6d_to_matrix(features)
            return self._standardize_quat(matrix_to_quaternion(rotation))
        raise AssertionError(f"Unhandled rotation_representation={self.rotation_representation}")

    def assemble_features(self, components: MotionComponents) -> torch.Tensor:
        body_link_flat = components.body_link_pos_local.reshape(*components.body_link_pos_local.shape[:-2], -1)
        return torch.cat(
            [
                components.root_pos_local,
                self.rotation_quat_to_features(components.root_rot_local_quat),
                components.joint_dof,
                body_link_flat,
            ],
            dim=-1,
        )

    def split_features(self, features: torch.Tensor) -> MotionComponents:
        s = self.feature_slices
        body_link_flat = features[..., s["body_link_pos_local"][0] : s["body_link_pos_local"][1]]
        body_link_pos_local = body_link_flat.view(*features.shape[:-1], self.num_body_links, 3)
        root_rot_features = features[..., s["root_rot_local"][0] : s["root_rot_local"][1]]
        root_rot_local_quat = self.rotation_features_to_quat(root_rot_features)
        return MotionComponents(
            root_pos_local=features[..., s["root_pos_local"][0] : s["root_pos_local"][1]],
            root_rot_local_quat=root_rot_local_quat,
            joint_dof=features[..., s["joint_dof"][0] : s["joint_dof"][1]],
            body_link_pos_local=body_link_pos_local,
        )

    def _split_qpos_36(self, qpos_36: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if qpos_36.shape[-1] != 36:
            raise ValueError(f"Expected qpos_36 last dim 36, got {qpos_36.shape}")
        root_pos = qpos_36[..., :3]
        root_quat = F.normalize(qpos_36[..., 3:7], dim=-1)
        root_quat = standardize_quaternion(root_quat)
        joint_dof = qpos_36[..., 7:]
        return root_pos, root_quat, joint_dof

    def canonicalize(
        self,
        qpos_36: torch.Tensor,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor | None,
        anchor_root_pos: torch.Tensor,
        anchor_root_quat: torch.Tensor,
        fps: torch.Tensor | float,
        valid_mask: torch.Tensor | None = None,
    ) -> MotionComponents:
        anchor_root_pos, _, heading_quat_inv = self._canonical_anchor(anchor_root_pos, anchor_root_quat)
        root_pos_w, root_rot_w, joint_dof = self._split_qpos_36(qpos_36)

        heading_quat_seq = heading_quat_inv.expand(*root_pos_w.shape[:-1], 4)
        root_pos_local = quaternion_apply(heading_quat_seq, root_pos_w - anchor_root_pos)
        root_rot_local_quat = quaternion_multiply(heading_quat_seq, root_rot_w)
        root_rot_local_quat = F.normalize(root_rot_local_quat, dim=-1)
        root_rot_local_quat = standardize_quaternion(root_rot_local_quat)

        body_link_pos_w = body_pos_w[..., 1:, :]
        heading_quat_body = heading_quat_inv.unsqueeze(-2).expand(*body_link_pos_w.shape[:-2], self.num_body_links, 4)
        body_link_pos_local = quaternion_apply(
            heading_quat_body,
            body_link_pos_w - anchor_root_pos.unsqueeze(-2),
        )
        return MotionComponents(
            root_pos_local=root_pos_local,
            root_rot_local_quat=root_rot_local_quat,
            joint_dof=joint_dof,
            body_link_pos_local=body_link_pos_local,
        )

    def body_rot_local_matrices(
        self,
        body_quat_w: torch.Tensor,
        anchor_root_quat: torch.Tensor,
    ) -> torch.Tensor:
        _, _, heading_quat_inv = self._canonical_anchor(
            torch.zeros_like(anchor_root_quat[..., :3]),
            anchor_root_quat,
        )
        heading_quat_body = heading_quat_inv.unsqueeze(-2).expand(*body_quat_w.shape[:-2], body_quat_w.shape[-2], 4)
        body_quat_local = quaternion_multiply(heading_quat_body, body_quat_w)
        body_quat_local = F.normalize(body_quat_local, dim=-1)
        body_quat_local = standardize_quaternion(body_quat_local)
        return quaternion_to_matrix(body_quat_local)

    def decode_to_world_qpos36(
        self,
        components: MotionComponents,
        anchor_root_pos: torch.Tensor,
        anchor_root_quat: torch.Tensor,
    ) -> torch.Tensor:
        anchor_root_pos, heading_quat, _ = self._canonical_anchor(anchor_root_pos, anchor_root_quat)
        heading_quat_seq = heading_quat.expand(*components.root_pos_local.shape[:-1], 4)
        root_pos_world = quaternion_apply(heading_quat_seq, components.root_pos_local) + anchor_root_pos
        root_rot_local_quat = F.normalize(components.root_rot_local_quat, dim=-1)
        root_rot_local_quat = standardize_quaternion(root_rot_local_quat)
        root_rot_world_quat = quaternion_multiply(heading_quat_seq, root_rot_local_quat)
        root_rot_world_quat = F.normalize(root_rot_world_quat, dim=-1)
        root_rot_world_quat = standardize_quaternion(root_rot_world_quat)
        return torch.cat([root_pos_world, root_rot_world_quat, components.joint_dof], dim=-1)

    def prev_state_features_from_history(
        self,
        prev_qpos_36: torch.Tensor,
        prev_body_pos_w: torch.Tensor,
        prev_body_quat_w: torch.Tensor | None,
        fps: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_root_pos = prev_qpos_36[..., self.canonical_frame_idx : self.canonical_frame_idx + 1, :3]
        anchor_root_quat = prev_qpos_36[..., self.canonical_frame_idx : self.canonical_frame_idx + 1, 3:7]
        anchor_root_quat = F.normalize(anchor_root_quat, dim=-1)
        anchor_root_quat = standardize_quaternion(anchor_root_quat)
        comps = self.canonicalize(
            prev_qpos_36,
            prev_body_pos_w,
            prev_body_quat_w,
            anchor_root_pos=anchor_root_pos,
            anchor_root_quat=anchor_root_quat,
            fps=fps,
            valid_mask=torch.ones(prev_qpos_36.shape[:-1], dtype=torch.bool, device=prev_qpos_36.device),
        )
        return self.assemble_features(comps), anchor_root_pos, anchor_root_quat
