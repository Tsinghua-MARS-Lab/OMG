from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import yaml
from tqdm import tqdm


MetricDict = Dict[str, Union[float, Dict[str, float]]]


@dataclass(frozen=True)
class H2HMeta:
    body_map: tuple[tuple[int, int], ...]
    ee_map: dict[str, tuple[int, int]]
    foot_ids: tuple[int, ...]
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    human_root_id: int
    robot_root_id: int
    heading_body_map: tuple[tuple[int, int], tuple[int, int]] | None = None


def _as_float_array(value: Any, *, name: str, ndim: int | None = None, shape_tail: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dims, got shape {array.shape}")
    if shape_tail is not None and array.shape[-len(shape_tail) :] != shape_tail:
        raise ValueError(f"{name} must end with shape {shape_tail}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_index_pair_list(value: Any, *, name: str) -> tuple[tuple[int, int], ...]:
    pairs = []
    for idx, pair in enumerate(value or []):
        if len(pair) != 2:
            raise ValueError(f"{name}[{idx}] must contain exactly two indices")
        pairs.append((int(pair[0]), int(pair[1])))
    if not pairs:
        raise ValueError(f"{name} must not be empty")
    return tuple(pairs)


def _as_ee_map(value: Any) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for key, pair in dict(value or {}).items():
        if len(pair) != 2:
            raise ValueError(f"ee_map[{key!r}] must contain exactly two indices")
        result[str(key)] = (int(pair[0]), int(pair[1]))
    if not result:
        raise ValueError("ee_map must not be empty")
    return result


def load_meta(path: str | Path) -> H2HMeta:
    meta_path = Path(path).expanduser().resolve()
    with meta_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    body_map = _as_index_pair_list(payload.get("body_map"), name="body_map")
    first_human, first_robot = body_map[0]
    heading_body_map = None
    if payload.get("heading_body_map") is not None:
        heading_pairs = _as_index_pair_list(payload["heading_body_map"], name="heading_body_map")
        if len(heading_pairs) != 2:
            raise ValueError("heading_body_map must contain exactly two [human_id, robot_id] pairs")
        heading_body_map = (heading_pairs[0], heading_pairs[1])

    return H2HMeta(
        body_map=body_map,
        ee_map=_as_ee_map(payload.get("ee_map")),
        foot_ids=tuple(int(value) for value in payload.get("foot_ids", [])),
        joint_lower=_as_float_array(payload.get("joint_lower"), name="joint_lower", ndim=1),
        joint_upper=_as_float_array(payload.get("joint_upper"), name="joint_upper", ndim=1),
        human_root_id=int(payload.get("human_root_id", first_human)),
        robot_root_id=int(payload.get("robot_root_id", first_robot)),
        heading_body_map=heading_body_map,
    )


def _validate_body_indices(human_body_pos: np.ndarray, robot_body_pos: np.ndarray, pairs: Sequence[tuple[int, int]]) -> None:
    human_bodies = int(human_body_pos.shape[1])
    robot_bodies = int(robot_body_pos.shape[1])
    for human_idx, robot_idx in pairs:
        if not 0 <= int(human_idx) < human_bodies:
            raise IndexError(f"human body index {human_idx} is outside [0, {human_bodies})")
        if not 0 <= int(robot_idx) < robot_bodies:
            raise IndexError(f"robot body index {robot_idx} is outside [0, {robot_bodies})")


def _matched_positions(
    human_body_pos: np.ndarray,
    robot_body_pos: np.ndarray,
    body_map: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    _validate_body_indices(human_body_pos, robot_body_pos, body_map)
    human_ids = [int(pair[0]) for pair in body_map]
    robot_ids = [int(pair[1]) for pair in body_map]
    return human_body_pos[:, human_ids], robot_body_pos[:, robot_ids]


def _normalize_xy(vector: np.ndarray, *, name: str) -> np.ndarray:
    xy = np.asarray(vector[..., :2], dtype=np.float32)
    norm = np.linalg.norm(xy, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError(f"{name} has near-zero xy heading vector")
    return xy / norm


def _yaw_from_quaternion_wxyz(quat: np.ndarray, *, name: str) -> np.ndarray:
    q = _as_float_array(quat, name=name, ndim=2)
    if q.shape[1] != 4:
        raise ValueError(f"{name} quaternion must have shape (T, 4), got {q.shape}")
    # H2H benchmark root quaternions are MuJoCo free-joint qpos[3:7] in wxyz order.
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)


def _yaw_from_6d(rot6d: np.ndarray, *, name: str) -> np.ndarray:
    rot = _as_float_array(rot6d, name=name, ndim=2)
    if rot.shape[1] != 6:
        raise ValueError(f"{name} 6D rotation must have shape (T, 6), got {rot.shape}")
    forward = rot[:, 0:3]
    xy = _normalize_xy(forward, name=name)
    return np.arctan2(xy[:, 1], xy[:, 0]).astype(np.float32)


def _yaw_from_rotation(root_rot: np.ndarray, *, name: str) -> np.ndarray:
    if root_rot.shape[-1] == 4:
        return _yaw_from_quaternion_wxyz(root_rot, name=name)
    if root_rot.shape[-1] == 6:
        return _yaw_from_6d(root_rot, name=name)
    raise ValueError(f"{name} must be quaternion (T,4) or 6D rotation (T,6), got {root_rot.shape}")


def _yaw_from_body_axis(body_pos: np.ndarray, first_id: int, second_id: int, *, name: str) -> np.ndarray:
    axis = body_pos[:, int(second_id)] - body_pos[:, int(first_id)]
    xy = _normalize_xy(axis, name=name)
    return np.arctan2(xy[:, 1], xy[:, 0]).astype(np.float32)


def _rotation_z(yaw: np.ndarray) -> np.ndarray:
    cos = np.cos(yaw)
    sin = np.sin(yaw)
    rot = np.zeros((yaw.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = cos
    rot[:, 0, 1] = -sin
    rot[:, 1, 0] = sin
    rot[:, 1, 1] = cos
    rot[:, 2, 2] = 1.0
    return rot


def _root_positions(human_body_pos: np.ndarray, robot: dict[str, np.ndarray], meta: H2HMeta) -> tuple[np.ndarray, np.ndarray]:
    human_root = human_body_pos[:, int(meta.human_root_id)]
    robot_root = _as_float_array(robot["root_pos"], name="robot_root_pos", ndim=2, shape_tail=(3,))
    if robot_root.shape[0] != human_body_pos.shape[0]:
        raise ValueError(f"robot_root_pos frame count {robot_root.shape[0]} != human frame count {human_body_pos.shape[0]}")
    return human_root, robot_root


def _heading_yaws(human_body_pos: np.ndarray, robot_body_pos: np.ndarray, robot: dict[str, np.ndarray], meta: H2HMeta) -> tuple[np.ndarray, np.ndarray, str]:
    if meta.heading_body_map is not None:
        first, second = meta.heading_body_map
        human_yaw = _yaw_from_body_axis(human_body_pos, first[0], second[0], name="human heading_body_map")
        robot_yaw = _yaw_from_body_axis(robot_body_pos, first[1], second[1], name="robot heading_body_map")
        return human_yaw, robot_yaw, "heading_body_map"
    if "human_root_rot" in robot:
        human_yaw = _yaw_from_rotation(robot["human_root_rot"], name="human_root_rot")
        robot_yaw = _yaw_from_rotation(robot["root_rot"], name="robot_root_rot")
        return human_yaw, robot_yaw, "root_rot"
    zeros = np.zeros((human_body_pos.shape[0],), dtype=np.float32)
    return zeros, zeros, "translation_only"


def _aligned_robot_positions(
    human_body_pos: np.ndarray,
    robot: dict[str, np.ndarray],
    meta: H2HMeta,
) -> tuple[np.ndarray, str]:
    robot_body_pos = robot["body_pos"]
    human_root, robot_root = _root_positions(human_body_pos, robot, meta)
    human_yaw, robot_yaw, heading_source = _heading_yaws(human_body_pos, robot_body_pos, robot, meta)
    rot = _rotation_z(human_yaw - robot_yaw)
    robot_local = robot_body_pos - robot_root[:, None, :]
    aligned = np.einsum("tij,tbj->tbi", rot, robot_local) + human_root[:, None, :]
    return aligned.astype(np.float32, copy=False), heading_source


def _mean_l2(delta: np.ndarray) -> float:
    return float(np.linalg.norm(delta, axis=-1).mean())


def compute_global_mpjpe(human_body_pos: np.ndarray, robot_body_pos: np.ndarray, body_map: Sequence[tuple[int, int]]) -> float:
    human_matched, robot_matched = _matched_positions(human_body_pos, robot_body_pos, body_map)
    return _mean_l2(robot_matched - human_matched)


def compute_root_aligned_mpjpe(human_body_pos: np.ndarray, robot: dict[str, np.ndarray], meta: H2HMeta) -> tuple[float, str]:
    aligned_robot, heading_source = _aligned_robot_positions(human_body_pos, robot, meta)
    human_matched, robot_matched = _matched_positions(human_body_pos, aligned_robot, meta.body_map)
    return _mean_l2(robot_matched - human_matched), heading_source


def compute_end_effector_error(human_body_pos: np.ndarray, robot: dict[str, np.ndarray], meta: H2HMeta) -> dict[str, float]:
    aligned_robot, _ = _aligned_robot_positions(human_body_pos, robot, meta)
    values: dict[str, float] = {}
    for name, pair in meta.ee_map.items():
        human_idx, robot_idx = pair
        _validate_body_indices(human_body_pos, aligned_robot, [pair])
        values[f"ee_{name}_error"] = _mean_l2(aligned_robot[:, robot_idx] - human_body_pos[:, human_idx])
    values["end_effector_error"] = float(np.mean([value for value in values.values()]))
    hand_values = [value for key, value in values.items() if "hand" in key]
    foot_values = [value for key, value in values.items() if "foot" in key]
    if hand_values:
        values["hand_error"] = float(np.mean(hand_values))
    if foot_values:
        values["foot_error"] = float(np.mean(foot_values))
    return values


def _finite_difference_positions(positions: np.ndarray, dt: float) -> np.ndarray:
    if positions.shape[0] < 2:
        raise ValueError("velocity metrics require at least 2 frames")
    return np.diff(positions, axis=0) / float(dt)


def compute_velocity_error(human_body_pos: np.ndarray, robot: dict[str, np.ndarray], meta: H2HMeta, dt: float) -> float:
    if float(dt) <= 0.0 or not math.isfinite(float(dt)):
        raise ValueError(f"dt must be finite and positive, got {dt}")
    human_vel = _as_float_array(robot["human_body_vel"], name="human_body_vel", ndim=3, shape_tail=(3,)) if "human_body_vel" in robot else _finite_difference_positions(human_body_pos, dt)
    if "body_vel" in robot:
        robot_vel_source = _as_float_array(robot["body_vel"], name="robot_body_vel", ndim=3, shape_tail=(3,))
    else:
        robot_vel_source = _finite_difference_positions(robot["body_pos"], dt)
    frames = min(human_vel.shape[0], robot_vel_source.shape[0], human_body_pos.shape[0] - 1)
    human_yaw, robot_yaw, _ = _heading_yaws(human_body_pos, robot["body_pos"], robot, meta)
    rot = _rotation_z(human_yaw[:frames] - robot_yaw[:frames])
    robot_vel = np.einsum("tij,tbj->tbi", rot, robot_vel_source[:frames])
    human_matched, robot_matched = _matched_positions(human_vel[:frames], robot_vel, meta.body_map)
    return _mean_l2(robot_matched - human_matched)


def compute_joint_jump_rate(joint_qpos: np.ndarray, threshold: float = 0.5) -> float:
    joint_qpos = _as_float_array(joint_qpos, name="robot_joint_qpos", ndim=2)
    if joint_qpos.shape[0] < 2:
        raise ValueError("joint jump rate requires at least 2 frames")
    jumps = np.max(np.abs(np.diff(joint_qpos, axis=0)), axis=1) > float(threshold)
    return float(jumps.mean())


def compute_joint_limit_rates(joint_qpos: np.ndarray, joint_lower: np.ndarray, joint_upper: np.ndarray, margin: float = 0.05) -> dict[str, float]:
    joint_qpos = _as_float_array(joint_qpos, name="robot_joint_qpos", ndim=2)
    lower = _as_float_array(joint_lower, name="joint_lower", ndim=1)
    upper = _as_float_array(joint_upper, name="joint_upper", ndim=1)
    if lower.shape != upper.shape:
        raise ValueError(f"joint_lower shape {lower.shape} != joint_upper shape {upper.shape}")
    if joint_qpos.shape[1] != lower.shape[0]:
        raise ValueError(f"robot_joint_qpos dim {joint_qpos.shape[1]} != joint limit dim {lower.shape[0]}")
    if np.any(lower >= upper):
        raise ValueError("joint_lower must be strictly smaller than joint_upper")
    soft = np.any((joint_qpos < lower[None] + float(margin)) | (joint_qpos > upper[None] - float(margin)), axis=1)
    hard = np.any((joint_qpos < lower[None]) | (joint_qpos > upper[None]), axis=1)
    return {
        "joint_limit_rate": float(soft.mean()),
        "joint_hard_limit_rate": float(hard.mean()),
    }


def compute_ground_penetration(
    robot_body_pos: np.ndarray,
    body_ids: Sequence[int] | None = None,
    *,
    ground_height: float = 0.0,
    threshold: float = 0.01,
) -> dict[str, float]:
    robot_body_pos = _as_float_array(robot_body_pos, name="robot_body_pos", ndim=3, shape_tail=(3,))
    ids = list(range(robot_body_pos.shape[1])) if body_ids is None else [int(idx) for idx in body_ids]
    if not ids:
        raise ValueError("ground penetration body ids must not be empty")
    for idx in ids:
        if not 0 <= idx < robot_body_pos.shape[1]:
            raise IndexError(f"ground penetration body index {idx} is outside [0, {robot_body_pos.shape[1]})")
    z = robot_body_pos[:, ids, 2]
    depth = np.maximum(0.0, float(ground_height) - z)
    return {
        "ground_penetration_depth": float(depth.mean()),
        "ground_penetration_rate": float((depth.max(axis=1) > float(threshold)).mean()),
    }


def compute_foot_sliding(
    robot_body_pos: np.ndarray,
    foot_ids: Sequence[int],
    dt: float,
    *,
    height_threshold: float = 0.05,
    vertical_velocity_threshold: float = 0.1,
    slide_threshold: float = 0.2,
    ground_height: float = 0.0,
) -> dict[str, float]:
    robot_body_pos = _as_float_array(robot_body_pos, name="robot_body_pos", ndim=3, shape_tail=(3,))
    feet = [int(idx) for idx in foot_ids]
    if not feet:
        raise ValueError("foot_ids must not be empty")
    for idx in feet:
        if not 0 <= idx < robot_body_pos.shape[1]:
            raise IndexError(f"foot id {idx} is outside [0, {robot_body_pos.shape[1]})")
    if float(dt) <= 0.0 or not math.isfinite(float(dt)):
        raise ValueError(f"dt must be finite and positive, got {dt}")
    foot_pos = robot_body_pos[:, feet]
    foot_vel = _finite_difference_positions(foot_pos, dt)
    contact_z = foot_pos[:-1, :, 2] < float(ground_height) + float(height_threshold)
    slow_vertical = np.abs(foot_vel[:, :, 2]) < float(vertical_velocity_threshold)
    contact = contact_z & slow_vertical
    horizontal_speed = np.linalg.norm(foot_vel[:, :, :2], axis=-1)
    contact_count = int(contact.sum())
    if contact_count == 0:
        return {
            "foot_sliding": 0.0,
            "foot_slide_rate": 0.0,
            "foot_contact_count": 0.0,
        }
    return {
        "foot_sliding": float(horizontal_speed[contact].mean()),
        "foot_slide_rate": float((horizontal_speed[contact] > float(slide_threshold)).mean()),
        "foot_contact_count": float(contact_count),
    }


def load_sequence(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    seq_path = Path(path).expanduser().resolve()
    with np.load(seq_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    required = ("human_body_pos", "robot_root_pos", "robot_root_rot", "robot_qpos", "robot_body_pos", "dt")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"{seq_path} missing required arrays: {missing}")
    human = {"body_pos": _as_float_array(arrays["human_body_pos"], name="human_body_pos", ndim=3, shape_tail=(3,))}
    robot = {
        "root_pos": _as_float_array(arrays["robot_root_pos"], name="robot_root_pos", ndim=2, shape_tail=(3,)),
        "root_rot": _as_float_array(arrays["robot_root_rot"], name="robot_root_rot", ndim=2),
        "qpos": _as_float_array(arrays["robot_qpos"], name="robot_qpos", ndim=2),
        "body_pos": _as_float_array(arrays["robot_body_pos"], name="robot_body_pos", ndim=3, shape_tail=(3,)),
    }
    optional_keys = {
        "human_body_vel": "human_body_vel",
        "human_root_rot": "human_root_rot",
        "robot_body_vel": "body_vel",
        "robot_body_ang_vel": "body_ang_vel",
    }
    for npz_key, robot_key in optional_keys.items():
        if npz_key in arrays:
            robot[robot_key] = _as_float_array(arrays[npz_key], name=npz_key)
    dt = float(np.asarray(arrays["dt"]).reshape(-1)[0])
    if human["body_pos"].shape[0] != robot["body_pos"].shape[0]:
        raise ValueError(f"{seq_path}: human/robot frame count mismatch {human['body_pos'].shape[0]} != {robot['body_pos'].shape[0]}")
    if robot["root_pos"].shape[0] != robot["body_pos"].shape[0] or robot["root_rot"].shape[0] != robot["body_pos"].shape[0]:
        raise ValueError(f"{seq_path}: robot root arrays must match robot_body_pos frame count")
    return human, robot, dt


def compute_sequence_metrics(
    human: dict[str, np.ndarray],
    robot: dict[str, np.ndarray],
    meta: H2HMeta,
    *,
    dt: float,
    joint_jump_threshold: float = 0.5,
    joint_limit_margin: float = 0.05,
    penetration_threshold: float = 0.01,
    foot_height_threshold: float = 0.05,
    foot_vertical_velocity_threshold: float = 0.1,
    foot_slide_threshold: float = 0.2,
    ground_height: float = 0.0,
) -> dict[str, Any]:
    human_body_pos = _as_float_array(human["body_pos"], name="human_body_pos", ndim=3, shape_tail=(3,))
    robot_body_pos = _as_float_array(robot["body_pos"], name="robot_body_pos", ndim=3, shape_tail=(3,))
    full_qpos = _as_float_array(robot["qpos"], name="robot_qpos", ndim=2)
    if full_qpos.shape[1] != 36:
        raise ValueError(f"robot_qpos must be MuJoCo qpos_36 with shape (T, 36), got {full_qpos.shape}")
    joint_qpos = full_qpos[:, 7:]
    if joint_qpos.shape[1] != meta.joint_lower.shape[0]:
        raise ValueError(
            f"robot_qpos[7:] joint dim {joint_qpos.shape[1]} != meta joint limit dim {meta.joint_lower.shape[0]}; "
            "meta joint_lower/joint_upper must match qpos[7:] MuJoCo joint order"
        )
    if human_body_pos.shape[0] != robot_body_pos.shape[0] or full_qpos.shape[0] != robot_body_pos.shape[0]:
        raise ValueError("human_body_pos, robot_body_pos, and robot_qpos must have the same frame count")

    ra_mpjpe, heading_source = compute_root_aligned_mpjpe(human_body_pos, robot, meta)
    metrics: dict[str, Any] = {
        "root_aligned_mpjpe": ra_mpjpe,
        "global_mpjpe": compute_global_mpjpe(human_body_pos, robot_body_pos, meta.body_map),
        "velocity_error": compute_velocity_error(human_body_pos, robot, meta, dt),
        "joint_jump_rate": compute_joint_jump_rate(joint_qpos, threshold=joint_jump_threshold),
        "self_collision_rate": float("nan"),
        "tracking_success_rate": float("nan"),
    }
    metrics.update(compute_end_effector_error(human_body_pos, robot, meta))
    metrics.update(compute_joint_limit_rates(joint_qpos, meta.joint_lower, meta.joint_upper, margin=joint_limit_margin))
    metrics.update(
        compute_ground_penetration(
            robot_body_pos,
            body_ids=meta.foot_ids,
            ground_height=ground_height,
            threshold=penetration_threshold,
        )
    )
    metrics.update(
        compute_foot_sliding(
            robot_body_pos,
            meta.foot_ids,
            dt,
            height_threshold=foot_height_threshold,
            vertical_velocity_threshold=foot_vertical_velocity_threshold,
            slide_threshold=foot_slide_threshold,
            ground_height=ground_height,
        )
    )
    metrics["metric_reasons"] = {
        "root_aligned_mpjpe_heading_source": heading_source,
        "self_collision_rate": "not_computed_collision_engine_not_connected",
        "tracking_success_rate": "not_computed_tracking_eval_logs_not_provided",
    }
    metrics["frames"] = int(robot_body_pos.shape[0])
    metrics["dt"] = float(dt)
    return metrics


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metric_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.floating)) and key not in {"frames", "dt"}:
                names.add(str(key))
    return sorted(names)


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty metrics")
    mean: dict[str, float] = {}
    median: dict[str, float] = {}
    p95: dict[str, float] = {}
    nan_reasons: dict[str, str] = {}
    for key in _metric_names(rows):
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            mean[key] = float("nan")
            median[key] = float("nan")
            p95[key] = float("nan")
            for row in rows:
                reason = row.get("metric_reasons", {}).get(key)
                if reason:
                    nan_reasons[key] = str(reason)
                    break
            continue
        mean[key] = float(finite.mean())
        median[key] = float(np.median(finite))
        p95[key] = float(np.percentile(finite, 95))
    return {
        "num_sequences": len(rows),
        "metrics_mean": mean,
        "metrics_median": median,
        "metrics_p95": p95,
        "nan_reasons": nan_reasons,
        "per_sequence": rows,
    }


def _sequence_paths(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz sequences found under {input_dir}")
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    meta = load_meta(args.meta)
    input_dir = Path(args.input_dir).expanduser().resolve()
    per_sequence = []
    for path in tqdm(_sequence_paths(input_dir), desc="H2H retarget metrics", unit="seq"):
        human, robot, dt = load_sequence(path)
        row = compute_sequence_metrics(
            human,
            robot,
            meta,
            dt=dt,
            joint_jump_threshold=args.joint_jump_threshold,
            joint_limit_margin=args.joint_limit_margin,
            penetration_threshold=args.penetration_threshold,
            foot_height_threshold=args.foot_height_threshold,
            foot_vertical_velocity_threshold=args.foot_vertical_velocity_threshold,
            foot_slide_threshold=args.foot_slide_threshold,
            ground_height=args.ground_height,
        )
        row["sequence"] = path.stem
        row["path"] = str(path)
        per_sequence.append(row)
    result = aggregate_metrics(per_sequence)
    result["benchmark"] = "h2h_retarget"
    result["input_dir"] = str(input_dir)
    result["meta"] = str(Path(args.meta).expanduser().resolve())
    if args.output is not None:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute human-motion to humanoid-motion retargeting metrics.")
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir", required=True, help="Directory containing canonical per-sequence .npz files.")
    parser.add_argument("--meta", required=True, help="YAML file containing body_map, ee_map, foot_ids, and joint limits.")
    parser.add_argument("--output", default=None, help="Output metrics JSON path.")
    parser.add_argument("--joint-jump-threshold", "--joint_jump_threshold", dest="joint_jump_threshold", type=float, default=0.5)
    parser.add_argument("--joint-limit-margin", "--joint_limit_margin", dest="joint_limit_margin", type=float, default=0.05)
    parser.add_argument("--penetration-threshold", "--penetration_threshold", dest="penetration_threshold", type=float, default=0.01)
    parser.add_argument("--foot-height-threshold", "--foot_height_threshold", dest="foot_height_threshold", type=float, default=0.05)
    parser.add_argument("--foot-vertical-velocity-threshold", "--foot_vertical_velocity_threshold", dest="foot_vertical_velocity_threshold", type=float, default=0.1)
    parser.add_argument("--foot-slide-threshold", "--foot_slide_threshold", dest="foot_slide_threshold", type=float, default=0.2)
    parser.add_argument("--ground-height", "--ground_height", dest="ground_height", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run(_parse_args(argv))
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
