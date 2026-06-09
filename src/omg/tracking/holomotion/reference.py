from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("Quaternion norm is too small")
    q = q / norm
    if q[0] < 0.0:
        q = -q
    return q.astype(np.float32, copy=False)


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float32).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float32).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_rotate_inv_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = normalize_quat_wxyz(q)
    v = np.asarray(v, dtype=np.float32)
    vq = np.concatenate([np.zeros(v.shape[:-1] + (1,), dtype=np.float32), v], axis=-1)
    q_conj = quat_conj_wxyz(q)
    if v.ndim == 1:
        return quat_mul_wxyz(quat_mul_wxyz(q_conj, vq), q)[1:].astype(np.float32, copy=False)
    out = np.zeros_like(v, dtype=np.float32)
    for idx in range(v.shape[0]):
        out[idx] = quat_mul_wxyz(quat_mul_wxyz(q_conj, vq[idx]), q)[1:]
    return out


def gravity_orientation_wxyz(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = normalize_quat_wxyz(q)
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0] = 2.0 * (-qz * qx + qw * qy)
    gravity[1] = -2.0 * (qz * qy + qw * qx)
    gravity[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity


def finite_difference(x: np.ndarray, fps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    if x.shape[0] <= 1:
        return out
    dt = 1.0 / float(fps)
    out[0] = (x[1] - x[0]) / dt
    out[-1] = (x[-1] - x[-2]) / dt
    if x.shape[0] > 2:
        out[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    return out.astype(np.float32, copy=False)


def quat_to_rotvec_wxyz(q: np.ndarray) -> np.ndarray:
    q = normalize_quat_wxyz(q)
    xyz = q[1:]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = xyz / sin_half
    angle = 2.0 * math.atan2(sin_half, float(q[0]))
    return (axis * angle).astype(np.float32, copy=False)


def body_angvel_from_quats(quats_wxyz: np.ndarray, fps: float) -> np.ndarray:
    quats_wxyz = np.asarray(quats_wxyz, dtype=np.float32)
    out = np.zeros((quats_wxyz.shape[0], 3), dtype=np.float32)
    if quats_wxyz.shape[0] <= 1:
        return out
    dt = 1.0 / float(fps)
    for idx in range(quats_wxyz.shape[0] - 1):
        q0 = normalize_quat_wxyz(quats_wxyz[idx])
        q1 = normalize_quat_wxyz(quats_wxyz[idx + 1])
        delta_local = quat_mul_wxyz(quat_conj_wxyz(q0), q1)
        out[idx] = quat_to_rotvec_wxyz(delta_local) / dt
    out[-1] = out[-2]
    return out.astype(np.float32, copy=False)


def resample_qpos(qpos_36: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    qpos_36 = np.asarray(qpos_36, dtype=np.float32)
    if qpos_36.ndim != 2 or qpos_36.shape[1] != 36:
        raise ValueError(f"Expected qpos_36 shape (T,36), got {qpos_36.shape}")
    if np.isclose(float(source_fps), float(target_fps)):
        out = qpos_36.copy()
        out[:, 3:7] = np.stack([normalize_quat_wxyz(q) for q in out[:, 3:7]], axis=0)
        return out.astype(np.float32, copy=False)
    if qpos_36.shape[0] == 1:
        out = qpos_36.copy()
        out[:, 3:7] = normalize_quat_wxyz(out[0, 3:7])
        return out.astype(np.float32, copy=False)
    source_t = np.arange(qpos_36.shape[0], dtype=np.float64) / float(source_fps)
    duration = float(source_t[-1])
    target_frames = int(np.floor(duration * float(target_fps))) + 1
    target_t = np.arange(target_frames, dtype=np.float64) / float(target_fps)
    target_t = np.clip(target_t, source_t[0], source_t[-1])
    out = np.empty((target_frames, 36), dtype=np.float32)
    for dim in list(range(3)) + list(range(7, 36)):
        out[:, dim] = np.interp(target_t, source_t, qpos_36[:, dim]).astype(np.float32)
    quat_wxyz = np.stack([normalize_quat_wxyz(q) for q in qpos_36[:, 3:7]], axis=0)
    slerp = Slerp(source_t, Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]]))
    target_xyzw = slerp(target_t).as_quat().astype(np.float32)
    out[:, 3:7] = target_xyzw[:, [3, 0, 1, 2]]
    out[:, 3:7] = np.stack([normalize_quat_wxyz(q) for q in out[:, 3:7]], axis=0)
    return out.astype(np.float32, copy=False)


def g1_qpos_joint_slice_to_onnx(qpos_g1: np.ndarray, onnx_to_g1: np.ndarray) -> np.ndarray:
    qpos_g1 = np.asarray(qpos_g1, dtype=np.float32)
    return qpos_g1[..., 7:36][..., onnx_to_g1].astype(np.float32, copy=False)


def precompute_reference_features(qpos_g1: np.ndarray, fps: float, onnx_to_g1: np.ndarray) -> dict[str, np.ndarray]:
    qpos_g1 = np.asarray(qpos_g1, dtype=np.float32)
    root_pos = qpos_g1[:, :3].astype(np.float32, copy=False)
    root_quat = np.stack([normalize_quat_wxyz(q) for q in qpos_g1[:, 3:7]], axis=0)
    root_linvel_world = finite_difference(root_pos, fps)
    base_linvel = np.stack([quat_rotate_inv_wxyz(root_quat[idx], root_linvel_world[idx]) for idx in range(root_quat.shape[0])])
    return {
        "qpos_g1": qpos_g1.astype(np.float32, copy=False),
        "ref_dof_pos_onnx": g1_qpos_joint_slice_to_onnx(qpos_g1, onnx_to_g1),
        "ref_root_height": root_pos[:, 2].astype(np.float32, copy=False),
        "ref_gravity_projection": np.stack([gravity_orientation_wxyz(q) for q in root_quat], axis=0),
        "ref_base_linvel": base_linvel.astype(np.float32, copy=False),
        "ref_base_angvel": body_angvel_from_quats(root_quat, fps),
    }


def future_indices(frame_idx: int, total_frames: int, n_fut_frames: int) -> np.ndarray:
    idx = np.arange(frame_idx + 1, frame_idx + 1 + int(n_fut_frames), dtype=np.int64)
    return np.clip(idx, 0, max(total_frames - 1, 0))


def history_indices(frame_idx: int, total_frames: int, context_length: int) -> np.ndarray:
    start = int(frame_idx) - int(context_length) + 1
    idx = np.arange(start, int(frame_idx) + 1, dtype=np.int64)
    return np.clip(idx, 0, max(total_frames - 1, 0))


def _append_history(
    history: dict[str, list[np.ndarray]],
    name: str,
    value: np.ndarray,
    context_length: int,
) -> np.ndarray:
    values = history.setdefault(name, [])
    values.append(np.asarray(value, dtype=np.float32).copy())
    if len(values) > int(context_length):
        del values[: len(values) - int(context_length)]
    padded = [values[0]] * (int(context_length) - len(values)) + values
    return np.stack(padded, axis=0).astype(np.float32, copy=False)


def current_robot_obs_terms(data: Any, g1_handles: dict[str, Any], holomotion_handles: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qpos_g1 = np.asarray(data.qpos[g1_handles["joint_qpos_adr"]], dtype=np.float32)
    qvel_g1 = np.asarray(data.qvel[g1_handles["joint_dof_adr"]], dtype=np.float32)
    qpos_onnx = qpos_g1[holomotion_handles.onnx_to_g1]
    qvel_onnx = qvel_g1[holomotion_handles.onnx_to_g1]
    dof_pos = (qpos_onnx - holomotion_handles.default_joint_pos).astype(np.float32, copy=False)
    dof_vel = qvel_onnx.astype(np.float32, copy=False)
    root_quat = np.asarray(data.xquat[g1_handles["pelvis_body_id"]], dtype=np.float32)
    projected_gravity = gravity_orientation_wxyz(root_quat)
    gyro_adr = g1_handles["pelvis_gyro_adr"]
    gyro_dim = g1_handles["pelvis_gyro_dim"]
    root_ang_vel = np.asarray(data.sensordata[gyro_adr : gyro_adr + gyro_dim], dtype=np.float32)
    return projected_gravity, root_ang_vel, dof_pos, dof_vel


def build_holomotion_obs(
    ref_features: dict[str, np.ndarray],
    frame_idx: int,
    n_fut_frames: int,
    data: Any,
    g1_handles: dict[str, Any],
    holomotion_handles: Any,
    last_action_onnx: np.ndarray,
    *,
    context_length: int = 1,
    robot_history: dict[str, list[np.ndarray]] | None = None,
) -> np.ndarray:
    total_frames = int(ref_features["qpos_g1"].shape[0])
    if total_frames <= 0:
        raise ValueError("Reference features are empty")
    frame_idx = int(np.clip(frame_idx, 0, total_frames - 1))
    context_length = int(context_length)
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    hist_idx = history_indices(frame_idx, total_frames, context_length)
    fut_idx = future_indices(frame_idx, total_frames, n_fut_frames)
    projected_gravity, root_ang_vel, dof_pos, dof_vel = current_robot_obs_terms(data, g1_handles, holomotion_handles)

    if context_length == 1:
        current_terms = [
            ref_features["ref_gravity_projection"][frame_idx],
            ref_features["ref_base_linvel"][frame_idx],
            ref_features["ref_base_angvel"][frame_idx],
            ref_features["ref_dof_pos_onnx"][frame_idx],
            np.array([ref_features["ref_root_height"][frame_idx]], dtype=np.float32),
            projected_gravity,
            root_ang_vel,
            dof_pos,
            dof_vel,
            np.asarray(last_action_onnx, dtype=np.float32),
        ]
    else:
        if robot_history is None:
            raise ValueError("robot_history is required when context_length > 1")
        current_terms = [
            ref_features["ref_gravity_projection"][hist_idx].reshape(-1),
            ref_features["ref_base_linvel"][hist_idx].reshape(-1),
            ref_features["ref_base_angvel"][hist_idx].reshape(-1),
            ref_features["ref_dof_pos_onnx"][hist_idx].reshape(-1),
            ref_features["ref_root_height"][hist_idx].reshape(-1),
            _append_history(robot_history, "projected_gravity", projected_gravity, context_length).reshape(-1),
            _append_history(robot_history, "root_ang_vel", root_ang_vel, context_length).reshape(-1),
            _append_history(robot_history, "dof_pos", dof_pos, context_length).reshape(-1),
            _append_history(robot_history, "dof_vel", dof_vel, context_length).reshape(-1),
            _append_history(
                robot_history,
                "last_action",
                np.asarray(last_action_onnx, dtype=np.float32),
                context_length,
            ).reshape(-1),
        ]
    current = np.concatenate(current_terms, axis=0)
    future = np.concatenate([
        ref_features["ref_dof_pos_onnx"][fut_idx].reshape(-1),
        ref_features["ref_root_height"][fut_idx].reshape(-1),
        ref_features["ref_gravity_projection"][fut_idx].reshape(-1),
        ref_features["ref_base_linvel"][fut_idx].reshape(-1),
        ref_features["ref_base_angvel"][fut_idx].reshape(-1),
    ], axis=0)
    return np.concatenate([current, future], axis=0).astype(np.float32, copy=False)[None, :]
