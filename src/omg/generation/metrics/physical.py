from __future__ import annotations

import torch
import torch.nn.functional as F

from omg.utils.rotation_conversions import quaternion_to_matrix


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=value.device, dtype=value.dtype)
    return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1.0)


def _first_valid(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    index = valid.float().argmax(dim=1)
    return x[torch.arange(x.shape[0], device=x.device), index]


def _last_valid(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    lengths = valid.long().sum(dim=1).clamp_min(1)
    index = lengths - 1
    return x[torch.arange(x.shape[0], device=x.device), index]


def _fps_vector(fps: torch.Tensor | float, *, batch_size: int, like: torch.Tensor) -> torch.Tensor:
    fps_tensor = torch.as_tensor(fps, device=like.device, dtype=like.dtype)
    if fps_tensor.ndim == 0 or fps_tensor.numel() == 1:
        return fps_tensor.reshape(1).expand(batch_size)
    fps_tensor = fps_tensor.reshape(-1)
    if fps_tensor.numel() != batch_size:
        raise ValueError(f"fps must be scalar or have one value per batch item, got {tuple(fps_tensor.shape)}")
    return fps_tensor


def _contact_interval_stats(
    *,
    sole_points: torch.Tensor,
    sole_radii: torch.Tensor,
    valid: torch.Tensor,
    fps: torch.Tensor | float,
    contact_height_threshold: float,
    contact_penetration_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sole_points.ndim != 4 or sole_points.shape[-1] != 3:
        raise ValueError(f"sole_points must have shape (B, T, P, 3), got {tuple(sole_points.shape)}")
    if valid.shape != sole_points.shape[:2]:
        raise ValueError(f"valid must have shape {tuple(sole_points.shape[:2])}, got {tuple(valid.shape)}")
    if sole_points.shape[1] < 2:
        raise ValueError("contact interval metrics require at least 2 frames")

    sole_radii = sole_radii.to(device=sole_points.device, dtype=sole_points.dtype)
    sole_bottom = sole_points[..., 2] - sole_radii.view(1, 1, -1)
    fps_per_batch = _fps_vector(fps, batch_size=sole_points.shape[0], like=sole_points)
    sole_speed = torch.diff(sole_points[..., :2], dim=1).norm(dim=-1) * fps_per_batch.view(-1, 1, 1)
    contact = (sole_bottom >= -float(contact_penetration_tolerance)) & (sole_bottom <= float(contact_height_threshold))
    contact_interval = contact[:, 1:] & contact[:, :-1] & valid[:, 1:, None] & valid[:, :-1, None]
    valid_interval = valid[:, 1:, None] & valid[:, :-1, None]
    valid_interval = valid_interval.expand_as(contact_interval)
    return sole_speed, contact_interval, valid_interval


def contact_sliding_speed(
    *,
    sole_points: torch.Tensor,
    sole_radii: torch.Tensor,
    valid: torch.Tensor,
    fps: torch.Tensor | float,
    contact_height_threshold: float,
    contact_penetration_tolerance: float = 0.02,
    sole_foot_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean horizontal foot speed during contact intervals, in m/s.

    Contact is defined per foot: a foot is in contact on a frame when any of
    its sole proxy points is within the ground band. Sliding is measured on
    consecutive contact frames for that foot, using the maximum horizontal
    proxy-point speed on the contacting foot.
    """
    if sole_points.ndim != 4 or sole_points.shape[-1] != 3:
        raise ValueError(f"sole_points must have shape (B, T, P, 3), got {tuple(sole_points.shape)}")
    if valid.shape != sole_points.shape[:2]:
        raise ValueError(f"valid must have shape {tuple(sole_points.shape[:2])}, got {tuple(valid.shape)}")
    if sole_points.shape[1] < 2:
        raise ValueError("contact sliding requires at least 2 frames")

    sole_radii = sole_radii.to(device=sole_points.device, dtype=sole_points.dtype)
    sole_bottom = sole_points[..., 2] - sole_radii.view(1, 1, -1)
    fps_per_batch = _fps_vector(fps, batch_size=sole_points.shape[0], like=sole_points)
    point_speed = torch.diff(sole_points[..., :2], dim=1).norm(dim=-1) * fps_per_batch.view(-1, 1, 1)
    point_contact = (sole_bottom >= -float(contact_penetration_tolerance)) & (
        sole_bottom <= float(contact_height_threshold)
    )

    if sole_foot_ids is None:
        contact_frame = point_contact.any(dim=-1)
        contact_interval = contact_frame[:, 1:] & contact_frame[:, :-1] & valid[:, 1:] & valid[:, :-1]
        interval_speed = point_speed.max(dim=-1).values
        return _masked_mean(interval_speed, contact_interval)

    sole_foot_ids = sole_foot_ids.to(device=sole_points.device)
    speeds = []
    masks = []
    for foot_id in torch.unique(sole_foot_ids, sorted=True):
        foot_mask = sole_foot_ids == foot_id
        foot_contact = point_contact[..., foot_mask].any(dim=-1)
        foot_interval = foot_contact[:, 1:] & foot_contact[:, :-1] & valid[:, 1:] & valid[:, :-1]
        foot_speed = point_speed[..., foot_mask].max(dim=-1).values
        speeds.append(foot_speed)
        masks.append(foot_interval)
    return _masked_mean(torch.stack(speeds, dim=-1), torch.stack(masks, dim=-1))


def foot_ground_error(
    *,
    sole_points: torch.Tensor,
    sole_radii: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute signed distance from the lowest sole point to the ground plane, in meters."""
    if sole_points.ndim != 4 or sole_points.shape[-1] != 3:
        raise ValueError(f"sole_points must have shape (B, T, P, 3), got {tuple(sole_points.shape)}")
    if valid.shape != sole_points.shape[:2]:
        raise ValueError(f"valid must have shape {tuple(sole_points.shape[:2])}, got {tuple(valid.shape)}")
    sole_radii = sole_radii.to(device=sole_points.device, dtype=sole_points.dtype)
    sole_bottom = sole_points[..., 2] - sole_radii.view(1, 1, -1)
    lowest_sole_bottom = sole_bottom.min(dim=-1).values
    return _masked_mean(lowest_sole_bottom.abs(), valid)


def body_jerk_mean(
    *,
    body_pos_w: torch.Tensor,
    valid: torch.Tensor,
    fps: torch.Tensor | float,
) -> torch.Tensor:
    """Mean third finite difference magnitude of body positions, in m/s^3."""
    if body_pos_w.ndim != 4 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape (B, T, J, 3), got {tuple(body_pos_w.shape)}")
    if valid.shape != body_pos_w.shape[:2]:
        raise ValueError(f"valid must have shape {tuple(body_pos_w.shape[:2])}, got {tuple(valid.shape)}")
    if body_pos_w.shape[1] < 4:
        raise ValueError("body_jerk_mean requires at least 4 frames")

    fps_per_batch = _fps_vector(fps, batch_size=body_pos_w.shape[0], like=body_pos_w)
    jerk = body_pos_w[:, 3:] - 3.0 * body_pos_w[:, 2:-1] + 3.0 * body_pos_w[:, 1:-2] - body_pos_w[:, :-3]
    jerk = jerk.norm(dim=-1) * fps_per_batch.pow(3).view(-1, 1, 1)
    valid_jerk = valid[:, 3:] & valid[:, 2:-1] & valid[:, 1:-2] & valid[:, :-3]
    return _masked_mean(jerk, valid_jerk)


def physical_qpos_metrics(
    *,
    qpos_36: torch.Tensor,
    representation,
    fps: torch.Tensor | float,
    contact_height_threshold: float,
    contact_penetration_tolerance: float = 0.02,
    valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if qpos_36.ndim == 2:
        qpos_36 = qpos_36.unsqueeze(0)
    if qpos_36.ndim != 3 or qpos_36.shape[-1] != 36:
        raise ValueError(f"qpos_36 must have shape (T, 36) or (B, T, 36), got {tuple(qpos_36.shape)}")
    if valid is None:
        valid = torch.ones(qpos_36.shape[:2], dtype=torch.bool, device=qpos_36.device)
    else:
        valid = valid.to(device=qpos_36.device, dtype=torch.bool)

    fk = representation.kinematics.forward_kinematics(qpos_36)
    sole_points, sole_radii = representation.kinematics.get_sole_proxy_points(
        fk["body_pos_w"],
        fk["body_quat_w"],
    )
    return {
        "foot_ground_error": foot_ground_error(
            sole_points=sole_points,
            sole_radii=sole_radii,
            valid=valid,
        ),
        "contact_sliding_speed": contact_sliding_speed(
            sole_points=sole_points,
            sole_radii=sole_radii,
            valid=valid,
            fps=fps,
            contact_height_threshold=contact_height_threshold,
            contact_penetration_tolerance=contact_penetration_tolerance,
            sole_foot_ids=getattr(representation.kinematics, "sole_proxy_foot_ids", None),
        ),
        "body_jerk_mean": body_jerk_mean(body_pos_w=fk["body_pos_w"], valid=valid, fps=fps),
    }


def physical_motion_metrics(
    *,
    pred_features: torch.Tensor,
    target_features: torch.Tensor,
    batch: dict,
    representation,
    valid: torch.Tensor,
    pred,
    pred_qpos: torch.Tensor,
    pred_fk: dict,
    contact_height_threshold: float,
    contact_penetration_tolerance: float = 0.02,
) -> dict[str, torch.Tensor]:
    target_features = target_features.to(pred_features.device)
    out = {
        "feature_rmse": _masked_mean((pred_features - target_features).pow(2), valid).sqrt(),
        "forward_displacement": _last_valid(pred.root_pos_local, valid)[:, 0].sub(_first_valid(pred.root_pos_local, valid)[:, 0]).mean(),
    }

    gt_qpos = batch.get("qpos_36")
    if gt_qpos is not None:
        gt_qpos = gt_qpos.to(pred_features.device)
        out["qpos_root_pos_rmse"] = _masked_mean((pred_qpos[..., :3] - gt_qpos[..., :3]).pow(2), valid).sqrt()

    if "body_pos_w" in batch:
        gt_body_pos = batch["body_pos_w"].to(pred_features.device)
        out["body_pos_rmse"] = _masked_mean((pred_fk["body_pos_w"] - gt_body_pos).pow(2), valid[:, :, None]).sqrt()

    sole_points, sole_radii = representation.kinematics.get_sole_proxy_points(
        pred_fk["body_pos_w"],
        pred_fk["body_quat_w"],
    )
    sole_bottom = sole_points[..., 2] - sole_radii.view(1, 1, -1)
    out["sole_penetration_mean"] = _masked_mean(torch.relu(-sole_bottom), valid[:, :, None])
    out["sole_clearance_mean"] = _masked_mean(sole_bottom, valid[:, :, None])
    out["foot_ground_error"] = _masked_mean(sole_bottom.min(dim=-1).values.abs(), valid)

    if sole_points.shape[1] > 1:
        fps = batch["fps"].to(pred_features.device, dtype=pred_features.dtype)
        out["contact_sliding_speed"] = contact_sliding_speed(
            sole_points=sole_points,
            sole_radii=sole_radii,
            valid=valid,
            fps=fps,
            contact_height_threshold=contact_height_threshold,
            contact_penetration_tolerance=contact_penetration_tolerance,
            sole_foot_ids=getattr(representation.kinematics, "sole_proxy_foot_ids", None),
        )

        root_quat = pred_qpos[..., 3:7]
        root_rot = quaternion_to_matrix(root_quat)
        heading = F.normalize(root_rot[..., :2, 0], dim=-1)
        dot = (heading[:, 1:] * heading[:, :-1]).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        valid_pair = valid[:, 1:] & valid[:, :-1]
        out["max_heading_step_deg"] = torch.rad2deg(torch.acos(dot).masked_fill(~valid_pair, 0.0)).max()

    if pred_qpos.shape[1] > 3:
        fps = batch["fps"].to(pred_features.device, dtype=pred_features.dtype)
        out["body_jerk_mean"] = body_jerk_mean(body_pos_w=pred_fk["body_pos_w"], valid=valid, fps=fps)

    history = batch.get("history_features", batch.get("prev_state_features"))
    if history is not None:
        prev = representation.codec.split_features(history.to(pred_features.device)[:, -1])
        out["seam_root_pos_error"] = (pred.root_pos_local[:, 0] - prev.root_pos_local).norm(dim=-1).mean()
        out["seam_joint_dof_error"] = (pred.joint_dof[:, 0] - prev.joint_dof).norm(dim=-1).mean()
        out["seam_body_pos_error"] = (pred.body_link_pos_local[:, 0] - prev.body_link_pos_local).norm(dim=-1).mean()

    return out
