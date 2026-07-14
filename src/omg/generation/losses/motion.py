from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from omg.generation.metrics import physical_motion_metrics
from omg.utils.rotation_conversions import (
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)


def _masked_element_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average every valid scalar while keeping samples equally weighted.

    Motion terms have different event shapes: root positions contain three
    coordinates, joint losses contain one value per joint, and FK losses add a
    body dimension.  Counting only valid frames makes those trailing dimensions
    implicit multipliers on the objective.  That is inconsistent with the
    diffusion objective, which averages its feature dimension explicitly.
    """
    mask = mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    denominator = mask.flatten(1).sum(dim=1)
    per_sample = (value * mask).flatten(1).sum(dim=1) / denominator.clamp_min(1.0)
    has_values = denominator > 0
    if not has_values.any():
        return value.new_zeros(())
    return per_sample[has_values].mean()



def _quat_geodesic_sq(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    pred_matrix = quaternion_to_matrix(pred)
    target_matrix = quaternion_to_matrix(target)
    return _rotation_geodesic_sq(pred_matrix, target_matrix)


def _rotation_geodesic_sq(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = pred.transpose(-1, -2) @ target
    trace = relative.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).pow(2)


def _quat_velocity(quat: torch.Tensor, fps: torch.Tensor) -> torch.Tensor:
    delta = quaternion_multiply(quat[:, 1:], quaternion_invert(quat[:, :-1]))
    axis_angle = quaternion_to_axis_angle(delta)
    while fps.ndim < axis_angle.ndim:
        fps = fps.unsqueeze(-1)
    return axis_angle * fps



class MotionLoss(nn.Module):
    def __init__(
        self,
        simple_root_pos: float = 1.0,
        simple_root_rot: float = 1.0,
        simple_joint_dof: float = 0.5,
        simple_body_pos: float = 0.5,
        body_pos_consistency: float = 1.0,
        vel_root_pos: float = 0.5,
        vel_root_rot: float = 0.02,
        vel_joint_dof: float = 0.01,
        fk_body_pos: float = 0.5,
        fk_body_rot: float = 0.1,
        body_vel: float = 0.03,
        terrain_penetration: float = 15.0,
        contact_height: float = 5.0,
        contact_velocity: float = 0.2,
        seam_root_pos: float = 1.0,
        seam_root_rot: float = 1.0,
        seam_joint_dof: float = 0.5,
        seam_body_pos: float = 0.5,
        contact_height_threshold: float = 0.03,
        contact_velocity_threshold: float = 0.5,
        loss_term_clip: float | None = None,
        loss_term_clips: Mapping[str, float] | None = None,
    ):
        super().__init__()
        self.weights = {
            "simple_root_pos": float(simple_root_pos),
            "simple_root_rot": float(simple_root_rot),
            "simple_joint_dof": float(simple_joint_dof),
            "simple_body_pos": float(simple_body_pos),
            "body_pos_consistency": float(body_pos_consistency),
            "vel_root_pos": float(vel_root_pos),
            "vel_root_rot": float(vel_root_rot),
            "vel_joint_dof": float(vel_joint_dof),
            "fk_body_pos": float(fk_body_pos),
            "fk_body_rot": float(fk_body_rot),
            "body_vel": float(body_vel),
            "terrain_penetration": float(terrain_penetration),
            "contact_height": float(contact_height),
            "contact_velocity": float(contact_velocity),
            "seam_root_pos": float(seam_root_pos),
            "seam_root_rot": float(seam_root_rot),
            "seam_joint_dof": float(seam_joint_dof),
            "seam_body_pos": float(seam_body_pos),
        }
        self.contact_height_threshold = float(contact_height_threshold)
        self.contact_velocity_threshold = float(contact_velocity_threshold)
        self.loss_term_clip = None if loss_term_clip is None else float(loss_term_clip)
        self.loss_term_clips = self._normalize_term_clips(loss_term_clips)

    def _normalize_term_clips(self, clips: Mapping[str, float] | None) -> dict[str, float]:
        if clips is None:
            return {}
        out = {}
        for key, value in dict(clips).items():
            name = str(key)
            if name.endswith("_loss"):
                name = name[: -len("_loss")]
            if name not in self.weights:
                raise ValueError(f"Unknown loss term clip {key!r}; expected one of {sorted(self.weights)}")
            out[name] = float(value)
        return out

    def _clip_value_for_term(self, name: str) -> float | None:
        value = self.loss_term_clips.get(name, self.loss_term_clip)
        if value is None or value <= 0.0:
            return None
        return float(value)

    def _apply_loss_term_clips(self, terms: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, bool]]:
        active: dict[str, bool] = {}
        clipped = dict(terms)
        for name in self.weights:
            key = f"{name}_loss"
            cap = self._clip_value_for_term(name)
            value = terms[key]
            is_active = False
            if cap is not None:
                is_active = bool((value.detach() > cap).cpu().item())
                clipped[key] = value.clamp(max=cap)
            active[name] = is_active
        return clipped, active

    def _decode_fk(self, pred, batch: dict, representation, valid: torch.Tensor) -> tuple[torch.Tensor, dict, object]:
        canon_root_pos = batch["canon_root_pos"].to(valid.device)
        canon_root_quat = batch["canon_root_quat"].to(valid.device)
        pred_qpos = representation.codec.decode_to_world_qpos36(
            pred,
            anchor_root_pos=canon_root_pos,
            anchor_root_quat=canon_root_quat,
        )
        pred_fk = representation.kinematics.forward_kinematics(pred_qpos)
        pred_fk_local = representation.codec.canonicalize(
            pred_qpos,
            pred_fk["body_pos_w"],
            pred_fk["body_quat_w"],
            anchor_root_pos=canon_root_pos,
            anchor_root_quat=canon_root_quat,
            fps=batch["fps"].to(valid.device),
            valid_mask=valid,
        )
        return pred_qpos, pred_fk, pred_fk_local

    def forward(self, pred_features: torch.Tensor, target_features: torch.Tensor, batch: dict, representation) -> dict[str, torch.Tensor]:
        valid = batch["mask"]["valid"].to(pred_features.device).bool()
        fps = batch["fps"].to(pred_features.device, dtype=pred_features.dtype)
        pred = representation.codec.split_features(pred_features)
        gt = representation.codec.split_features(target_features.to(pred_features.device))
        zero = pred_features.new_zeros(())
        terms = {
            "simple_root_pos_loss": _masked_element_mean((pred.root_pos_local - gt.root_pos_local).pow(2), valid),
            "simple_root_rot_loss": _masked_element_mean(_quat_geodesic_sq(pred.root_rot_local_quat, gt.root_rot_local_quat), valid),
            "simple_joint_dof_loss": _masked_element_mean((pred.joint_dof - gt.joint_dof).pow(2), valid),
            "simple_body_pos_loss": _masked_element_mean((pred.body_link_pos_local - gt.body_link_pos_local).pow(2), valid[:, :, None]),
            "body_pos_consistency_loss": zero,
            "vel_root_pos_loss": zero,
            "vel_root_rot_loss": zero,
            "vel_joint_dof_loss": zero,
            "fk_body_pos_loss": zero,
            "fk_body_rot_loss": zero,
            "body_vel_loss": zero,
            "terrain_penetration_loss": zero,
            "contact_height_loss": zero,
            "contact_velocity_loss": zero,
            "seam_root_pos_loss": zero,
            "seam_root_rot_loss": zero,
            "seam_joint_dof_loss": zero,
            "seam_body_pos_loss": zero,
        }

        if pred_features.shape[1] > 1:
            pair_mask = valid[:, 1:] & valid[:, :-1]
            terms["vel_root_pos_loss"] = _masked_element_mean(
                (torch.diff(pred.root_pos_local, dim=1) - torch.diff(gt.root_pos_local, dim=1)).pow(2),
                pair_mask,
            )
            terms["vel_root_rot_loss"] = _masked_element_mean(
                (_quat_velocity(pred.root_rot_local_quat, fps) - _quat_velocity(gt.root_rot_local_quat, fps)).pow(2),
                pair_mask,
            )
            terms["vel_joint_dof_loss"] = _masked_element_mean(
                (torch.diff(pred.joint_dof, dim=1) - torch.diff(gt.joint_dof, dim=1)).pow(2),
                pair_mask,
            )

        needs_fk = any(
            self.weights[name] > 0.0
            for name in (
                "body_pos_consistency",
                "fk_body_pos",
                "fk_body_rot",
                "body_vel",
                "terrain_penetration",
                "contact_height",
                "contact_velocity",
            )
        )
        pred_qpos = None
        pred_fk = None
        pred_fk_local = None
        if needs_fk:
            pred_qpos, pred_fk, pred_fk_local = self._decode_fk(pred, batch, representation, valid)
            terms["body_pos_consistency_loss"] = _masked_element_mean(
                (pred.body_link_pos_local - pred_fk_local.body_link_pos_local).pow(2),
                valid[:, :, None],
            )
            terms["fk_body_pos_loss"] = _masked_element_mean(
                (pred_fk_local.body_link_pos_local - gt.body_link_pos_local).pow(2),
                valid[:, :, None],
            )

            if "body_quat_w" in batch:
                gt_body_rot_local = representation.codec.body_rot_local_matrices(
                    batch["body_quat_w"].to(pred_features.device),
                    batch["canon_root_quat"].to(pred_features.device),
                )
                pred_body_rot_local = representation.codec.body_rot_local_matrices(
                    pred_fk["body_quat_w"],
                    batch["canon_root_quat"].to(pred_features.device),
                )
                terms["fk_body_rot_loss"] = _masked_element_mean(
                    _rotation_geodesic_sq(pred_body_rot_local, gt_body_rot_local),
                    valid[:, :, None],
                )

            if pred_features.shape[1] > 1:
                pair_mask = valid[:, 1:] & valid[:, :-1]
                terms["body_vel_loss"] = _masked_element_mean(
                    (torch.diff(pred_fk_local.body_link_pos_local, dim=1) - torch.diff(gt.body_link_pos_local, dim=1)).pow(2),
                    pair_mask[:, :, None],
                )

            sole_points, sole_radii = representation.kinematics.get_sole_proxy_points(
                pred_fk["body_pos_w"],
                pred_fk["body_quat_w"],
            )
            sole_bottom = sole_points[..., 2] - sole_radii.view(1, 1, -1)
            terms["terrain_penetration_loss"] = _masked_element_mean(torch.relu(-sole_bottom).pow(2), valid[:, :, None])
            if "body_pos_w" in batch and "body_quat_w" in batch:
                gt_sole, gt_radii = representation.kinematics.get_sole_proxy_points(
                    batch["body_pos_w"].to(pred_features.device),
                    batch["body_quat_w"].to(pred_features.device),
                )
                gt_bottom = gt_sole[..., 2] - gt_radii.view(1, 1, -1)
                gt_vel = torch.zeros_like(gt_sole[..., :2])
                if gt_sole.shape[1] > 1:
                    gt_vel[:, 1:] = torch.diff(gt_sole[..., :2], dim=1) * fps.view(-1, 1, 1, 1)
                contact_mask = (
                    valid[:, :, None]
                    & (gt_bottom <= self.contact_height_threshold)
                    & (gt_vel.norm(dim=-1) <= self.contact_velocity_threshold)
                )
                terms["contact_height_loss"] = _masked_element_mean(sole_bottom.pow(2), contact_mask)
                if pred_features.shape[1] > 1:
                    pred_vel = torch.diff(sole_points[..., :2], dim=1) * fps.view(-1, 1, 1, 1)
                    terms["contact_velocity_loss"] = _masked_element_mean(
                        pred_vel.pow(2).sum(dim=-1),
                        contact_mask[:, 1:] & contact_mask[:, :-1],
                    )

        history = batch.get("history_features", batch.get("prev_state_features"))
        if history is not None:
            prev = representation.codec.split_features(history.to(pred_features.device)[:, -1])
            first_valid = valid[:, :1]
            terms["seam_root_pos_loss"] = _masked_element_mean(
                (pred.root_pos_local[:, :1] - prev.root_pos_local[:, None]).pow(2),
                first_valid,
            )
            terms["seam_root_rot_loss"] = _masked_element_mean(
                _quat_geodesic_sq(pred.root_rot_local_quat[:, :1], prev.root_rot_local_quat[:, None]),
                first_valid,
            )
            terms["seam_joint_dof_loss"] = _masked_element_mean(
                (pred.joint_dof[:, :1] - prev.joint_dof[:, None]).pow(2),
                first_valid,
            )
            terms["seam_body_pos_loss"] = _masked_element_mean(
                (pred.body_link_pos_local[:, :1] - prev.body_link_pos_local[:, None]).pow(2),
                first_valid[:, :, None],
            )

        total_unclipped = pred_features.new_zeros(())
        for name, weight in self.weights.items():
            total_unclipped = total_unclipped + float(weight) * terms[f"{name}_loss"]
        terms, clip_active = self._apply_loss_term_clips(terms)

        total = pred_features.new_zeros(())
        for name, weight in self.weights.items():
            total = total + float(weight) * terms[f"{name}_loss"]
        terms["motion_loss_unclipped"] = total_unclipped
        terms["loss_term_clip_active_count"] = pred_features.new_tensor(
            sum(1 for value in clip_active.values() if value),
            dtype=pred_features.dtype,
        )
        terms["motion_loss"] = total

        if pred_qpos is None or pred_fk is None:
            pred_qpos, pred_fk, _ = self._decode_fk(pred, batch, representation, valid)
        terms.update(
            physical_motion_metrics(
                pred_features=pred_features,
                target_features=target_features,
                batch=batch,
                representation=representation,
                valid=valid,
                pred=pred,
                pred_qpos=pred_qpos,
                pred_fk=pred_fk,
                contact_height_threshold=self.contact_height_threshold,
            )
        )
        return terms
