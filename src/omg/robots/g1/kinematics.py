from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.nn.functional as F

from omg.utils.rotation_conversions import (
    axis_angle_to_matrix,
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    standardize_quaternion,
)


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[4] / path


def _resolve_asset_path(base_path: Path, asset_path: str | Path) -> Path:
    path = Path(asset_path)
    if path.is_absolute():
        return path
    relative_to_base = (base_path.parent / path).resolve()
    if relative_to_base.exists():
        return relative_to_base
    return _resolve_repo_path(path)


def _view_for_batch(x: torch.Tensor, batch_ndim: int) -> torch.Tensor:
    return x.view(*([1] * batch_ndim), *x.shape)


def _axis_aligned_rotation_matrix(
    angle: torch.Tensor,
    axis_dim: int,
    axis_sign: float,
) -> torch.Tensor:
    angle = angle.squeeze(-1) * axis_sign
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    rot = torch.empty(*angle.shape, 3, 3, device=angle.device, dtype=angle.dtype)

    if axis_dim == 0:
        rot[..., 0, 0] = one
        rot[..., 0, 1] = zero
        rot[..., 0, 2] = zero
        rot[..., 1, 0] = zero
        rot[..., 1, 1] = cos
        rot[..., 1, 2] = -sin
        rot[..., 2, 0] = zero
        rot[..., 2, 1] = sin
        rot[..., 2, 2] = cos
    elif axis_dim == 1:
        rot[..., 0, 0] = cos
        rot[..., 0, 1] = zero
        rot[..., 0, 2] = sin
        rot[..., 1, 0] = zero
        rot[..., 1, 1] = one
        rot[..., 1, 2] = zero
        rot[..., 2, 0] = -sin
        rot[..., 2, 1] = zero
        rot[..., 2, 2] = cos
    elif axis_dim == 2:
        rot[..., 0, 0] = cos
        rot[..., 0, 1] = -sin
        rot[..., 0, 2] = zero
        rot[..., 1, 0] = sin
        rot[..., 1, 1] = cos
        rot[..., 1, 2] = zero
        rot[..., 2, 0] = zero
        rot[..., 2, 1] = zero
        rot[..., 2, 2] = one
    else:
        raise ValueError(f"axis_dim must be 0, 1, or 2, got {axis_dim}")
    return rot


def _cardinal_axis_dim_sign(axis: list[float]) -> tuple[int, float] | None:
    abs_axis = [abs(float(v)) for v in axis]
    axis_dim = max(range(3), key=lambda idx: abs_axis[idx])
    if abs(abs_axis[axis_dim] - 1.0) > 1e-6:
        return None
    if any(abs_axis[idx] > 1e-6 for idx in range(3) if idx != axis_dim):
        return None
    return axis_dim, 1.0 if float(axis[axis_dim]) >= 0.0 else -1.0


class G1Kinematics(nn.Module):
    def __init__(self, kinematics_path: str | Path = "assets/robots/g1/g1_kinematics.json"):
        super().__init__()

        kinematics_path = _resolve_repo_path(kinematics_path)
        with kinematics_path.open("r", encoding="utf-8") as f:
            spec = json.load(f)

        self.kinematics_path = str(kinematics_path)
        self.robot_name = spec["robot_name"]
        self.root_link = spec["root_link"]
        self.body_order = tuple(spec["body_order"])
        self.joint_order = tuple(spec["joint_order"])
        self.body_name_to_index = dict(spec["body_name_to_index"])
        self.joint_name_to_qpos_index = dict(spec["joint_name_to_qpos_index"])
        urdf_path = _resolve_asset_path(kinematics_path, spec.get("source_urdf", "g1_29dof.urdf"))
        self.urdf_path = str(urdf_path)
        self._parent_body_indices_py = tuple(int(v) for v in spec["parent_body_indices"])
        self._child_body_indices_py = tuple(int(v) for v in spec["child_body_indices"])
        axis_dim_signs = [_cardinal_axis_dim_sign(axis) for axis in spec["joint_axes"]]
        self._joint_axes_are_cardinal = all(item is not None for item in axis_dim_signs)
        self._joint_axis_dims_py = tuple(-1 if item is None else item[0] for item in axis_dim_signs)
        self._joint_axis_signs_py = tuple(1.0 if item is None else item[1] for item in axis_dim_signs)

        self.register_buffer(
            "parent_body_indices",
            torch.tensor(spec["parent_body_indices"], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "child_body_indices",
            torch.tensor(spec["child_body_indices"], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "joint_axes",
            torch.tensor(spec["joint_axes"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "joint_origin_xyz",
            torch.tensor(spec["joint_origin_xyz"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "joint_origin_rpy",
            torch.tensor(spec["joint_origin_rpy"], dtype=torch.float32),
            persistent=False,
        )

        # URDF origin rpy is fixed-axis roll-pitch-yaw:
        # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        joint_origin_rot = euler_angles_to_matrix(
            self.joint_origin_rpy[..., [2, 1, 0]], convention="ZYX"
        )
        self.register_buffer("joint_origin_rot", joint_origin_rot, persistent=False)

        joint_lower_limits, joint_upper_limits = self._load_joint_limits(urdf_path)
        self.register_buffer("joint_lower_limits", joint_lower_limits, persistent=False)
        self.register_buffer("joint_upper_limits", joint_upper_limits, persistent=False)

        sole_proxy_body_indices, sole_proxy_local_positions, sole_proxy_radii, sole_proxy_foot_ids = (
            self._load_sole_proxy_spec(urdf_path)
        )
        self.register_buffer("sole_proxy_body_indices", sole_proxy_body_indices, persistent=False)
        self.register_buffer("sole_proxy_local_positions", sole_proxy_local_positions, persistent=False)
        self.register_buffer("sole_proxy_radii", sole_proxy_radii, persistent=False)
        self.register_buffer("sole_proxy_foot_ids", sole_proxy_foot_ids, persistent=False)

    def _load_joint_limits(self, urdf_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        root = ET.parse(urdf_path).getroot()
        lower_limits = []
        upper_limits = []
        for joint_name in self.joint_order:
            joint_node = root.find(f"./joint[@name='{joint_name}']")
            if joint_node is None:
                raise ValueError(f"Missing joint `{joint_name}` in URDF `{urdf_path}`")
            limit_node = joint_node.find("limit")
            if limit_node is None:
                lower_limits.append(float("-inf"))
                upper_limits.append(float("inf"))
                continue
            lower_limits.append(float(limit_node.get("lower", "-inf")))
            upper_limits.append(float(limit_node.get("upper", "inf")))
        return (
            torch.tensor(lower_limits, dtype=torch.float32),
            torch.tensor(upper_limits, dtype=torch.float32),
        )

    def _load_sole_proxy_spec(
        self, urdf_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        root = ET.parse(urdf_path).getroot()
        foot_links = (
            ("left_ankle_roll_link", 0),
            ("right_ankle_roll_link", 1),
        )
        body_indices: list[int] = []
        local_positions: list[list[float]] = []
        radii: list[float] = []
        foot_ids: list[int] = []
        for link_name, foot_id in foot_links:
            link_node = root.find(f"./link[@name='{link_name}']")
            if link_node is None:
                raise ValueError(f"Missing link `{link_name}` in URDF `{urdf_path}`")
            body_idx = self.body_name_to_index[link_name]
            for collision in link_node.findall("collision"):
                geometry = collision.find("geometry")
                sphere = None if geometry is None else geometry.find("sphere")
                if sphere is None:
                    continue
                origin = collision.find("origin")
                xyz_str = "0 0 0" if origin is None else origin.get("xyz", "0 0 0")
                xyz = [float(v) for v in xyz_str.split()]
                radius = float(sphere.get("radius"))
                body_indices.append(body_idx)
                local_positions.append(xyz)
                radii.append(radius)
                foot_ids.append(foot_id)
        if not body_indices:
            raise ValueError(f"No sole proxy spheres found in URDF `{urdf_path}`")
        return (
            torch.tensor(body_indices, dtype=torch.long),
            torch.tensor(local_positions, dtype=torch.float32),
            torch.tensor(radii, dtype=torch.float32),
            torch.tensor(foot_ids, dtype=torch.long),
        )

    @property
    def num_joints(self) -> int:
        return len(self.joint_order)

    @property
    def num_bodies(self) -> int:
        return len(self.body_order)

    @staticmethod
    def body_quat_wxyz_to_matrix(body_quat_wxyz: torch.Tensor) -> torch.Tensor:
        body_quat_wxyz = F.normalize(body_quat_wxyz, dim=-1)
        return quaternion_to_matrix(body_quat_wxyz)

    @staticmethod
    def matrix_to_body_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
        quat = matrix_to_quaternion(matrix)
        quat = F.normalize(quat, dim=-1)
        return standardize_quaternion(quat)

    def get_sole_proxy_points(
        self,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        body_dim = body_pos_w.ndim - 2
        proxy_body_indices = self.sole_proxy_body_indices.to(device=body_pos_w.device)
        proxy_offsets = self.sole_proxy_local_positions.to(
            device=body_pos_w.device,
            dtype=body_pos_w.dtype,
        )
        proxy_radii = self.sole_proxy_radii.to(device=body_pos_w.device, dtype=body_pos_w.dtype)

        proxy_body_pos = body_pos_w.index_select(dim=body_dim, index=proxy_body_indices)
        proxy_body_quat = body_quat_w.index_select(dim=body_dim, index=proxy_body_indices)
        proxy_body_rot = quaternion_to_matrix(proxy_body_quat)
        proxy_offsets = proxy_offsets.view(*([1] * (body_pos_w.ndim - 2)), *proxy_offsets.shape)
        proxy_points = proxy_body_pos + torch.matmul(
            proxy_body_rot, proxy_offsets.unsqueeze(-1)
        ).squeeze(-1)
        return proxy_points, proxy_radii

    def clamp_joint_positions(self, joint_pos: torch.Tensor) -> torch.Tensor:
        lower = self.joint_lower_limits.to(device=joint_pos.device, dtype=joint_pos.dtype)
        upper = self.joint_upper_limits.to(device=joint_pos.device, dtype=joint_pos.dtype)
        return joint_pos.clamp(min=lower, max=upper)

    def _split_qpos(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if qpos.shape[-1] != 36:
            raise ValueError(f"Expected qpos_36 last dim 36, got {qpos.shape}")
        root_pos = qpos[..., :3]
        root_quat = F.normalize(qpos[..., 3:7], dim=-1)
        root_quat = standardize_quaternion(root_quat)
        root_rot = quaternion_to_matrix(root_quat)
        joint_pos = qpos[..., 7:]
        return root_pos, root_quat, root_rot, joint_pos

    def _joint_rotation_matrix(
        self,
        joint_pos: torch.Tensor,
        joint_idx: int,
        batch_ndim: int,
        joint_axes: torch.Tensor,
    ) -> torch.Tensor:
        angle = joint_pos[..., joint_idx : joint_idx + 1]
        if self._joint_axes_are_cardinal:
            return _axis_aligned_rotation_matrix(
                angle,
                self._joint_axis_dims_py[joint_idx],
                self._joint_axis_signs_py[joint_idx],
            )
        axis = joint_axes[joint_idx]
        axis_view = _view_for_batch(axis, batch_ndim)
        return axis_angle_to_matrix(axis_view * angle)

    def _forward_body_pos_rot(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos, _, root_rot, joint_pos = self._split_qpos(qpos)
        batch_ndim = len(qpos.shape[:-1])
        device = qpos.device
        dtype = qpos.dtype
        joint_axes = self.joint_axes.to(device=device, dtype=dtype)
        joint_origin_rot = self.joint_origin_rot.to(device=device, dtype=dtype)
        joint_origin_xyz = self.joint_origin_xyz.to(device=device, dtype=dtype)

        body_pos_by_index: list[torch.Tensor | None] = [None] * self.num_bodies
        body_rot_by_index: list[torch.Tensor | None] = [None] * self.num_bodies
        body_pos_by_index[0] = root_pos
        body_rot_by_index[0] = root_rot

        for joint_idx in range(self.num_joints):
            parent_idx = self._parent_body_indices_py[joint_idx]
            child_idx = self._child_body_indices_py[joint_idx]

            parent_pos = body_pos_by_index[parent_idx]
            parent_rot = body_rot_by_index[parent_idx]
            if parent_pos is None or parent_rot is None:
                raise RuntimeError(
                    f"Kinematic chain is not topologically ordered for joint {joint_idx}: "
                    f"parent body {parent_idx} is unresolved"
                )

            joint_rot = self._joint_rotation_matrix(joint_pos, joint_idx, batch_ndim, joint_axes)
            origin_rot = _view_for_batch(joint_origin_rot[joint_idx], batch_ndim)
            origin_xyz = _view_for_batch(joint_origin_xyz[joint_idx], batch_ndim).unsqueeze(-1)

            child_pos = parent_pos + (parent_rot @ origin_xyz).squeeze(-1)
            child_rot = parent_rot @ (origin_rot @ joint_rot)
            body_pos_by_index[child_idx] = child_pos
            body_rot_by_index[child_idx] = child_rot

        if any(x is None for x in body_pos_by_index) or any(x is None for x in body_rot_by_index):
            raise RuntimeError("Forward kinematics did not resolve all body states")
        body_pos = torch.stack([x for x in body_pos_by_index if x is not None], dim=-2)
        body_rot = torch.stack([x for x in body_rot_by_index if x is not None], dim=-3)
        return body_pos, body_rot

    def forward_body_positions(self, qpos: torch.Tensor) -> torch.Tensor:
        body_pos, _ = self._forward_body_pos_rot(qpos)
        return body_pos

    def forward_kinematics_full(self, qpos: torch.Tensor) -> dict[str, torch.Tensor]:
        root_pos, _, root_rot, joint_pos = self._split_qpos(qpos)

        batch_shape = qpos.shape[:-1]
        device = qpos.device
        dtype = qpos.dtype
        batch_ndim = len(batch_shape)

        body_pos_by_index: list[torch.Tensor | None] = [None] * self.num_bodies
        body_rot_by_index: list[torch.Tensor | None] = [None] * self.num_bodies
        joint_pos_by_index: list[torch.Tensor] = []
        joint_axis_by_index: list[torch.Tensor] = []
        body_pos_by_index[0] = root_pos
        body_rot_by_index[0] = root_rot

        for joint_idx in range(self.num_joints):
            parent_idx = int(self.parent_body_indices[joint_idx].item())
            child_idx = int(self.child_body_indices[joint_idx].item())

            parent_pos = body_pos_by_index[parent_idx]
            parent_rot = body_rot_by_index[parent_idx]
            if parent_pos is None or parent_rot is None:
                raise RuntimeError(
                    f"Kinematic chain is not topologically ordered for joint {joint_idx}: "
                    f"parent body {parent_idx} is unresolved"
                )

            axis = self.joint_axes[joint_idx].to(device=device, dtype=dtype)
            angle = joint_pos[..., joint_idx : joint_idx + 1]
            axis_view = _view_for_batch(axis, batch_ndim)
            axis_angle = axis_view * angle
            joint_rot = axis_angle_to_matrix(axis_angle)

            origin_rot = self.joint_origin_rot[joint_idx].to(device=device, dtype=dtype)
            origin_rot = _view_for_batch(origin_rot, batch_ndim)
            local_rot = origin_rot @ joint_rot

            origin_xyz = self.joint_origin_xyz[joint_idx].to(device=device, dtype=dtype)
            origin_xyz = _view_for_batch(origin_xyz, batch_ndim).unsqueeze(-1)

            child_pos = parent_pos + (parent_rot @ origin_xyz).squeeze(-1)
            child_rot = parent_rot @ local_rot

            joint_pos_by_index.append(child_pos)
            joint_axis_parent = (origin_rot @ axis_view.unsqueeze(-1)).squeeze(-1)
            joint_axis_world = torch.matmul(parent_rot, joint_axis_parent.unsqueeze(-1)).squeeze(-1)
            joint_axis_by_index.append(F.normalize(joint_axis_world, dim=-1))

            body_pos_by_index[child_idx] = child_pos
            body_rot_by_index[child_idx] = child_rot

        if any(x is None for x in body_pos_by_index) or any(x is None for x in body_rot_by_index):
            raise RuntimeError("Forward kinematics did not resolve all body states")

        body_pos = torch.stack([x for x in body_pos_by_index if x is not None], dim=-2)
        body_rot = torch.stack([x for x in body_rot_by_index if x is not None], dim=-3)
        joint_pos_w = torch.stack(joint_pos_by_index, dim=-2)
        joint_axis_w = torch.stack(joint_axis_by_index, dim=-2)
        body_quat = self.matrix_to_body_quat_wxyz(body_rot)
        return {
            "body_pos_w": body_pos,
            "body_rot_w": body_rot,
            "body_quat_w": body_quat,
            "joint_pos_w": joint_pos_w,
            "joint_axis_w": joint_axis_w,
        }

    def forward_kinematics(self, qpos: torch.Tensor) -> dict[str, torch.Tensor]:
        body_pos, body_rot = self._forward_body_pos_rot(qpos)
        return {
            "body_pos_w": body_pos,
            "body_quat_w": self.matrix_to_body_quat_wxyz(body_rot),
        }
