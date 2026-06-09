from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from omg.robots.g1.constants import G1_JOINT_NAMES
from omg.tracking.holomotion.io import load_reference_motion
from omg.tracking.holomotion.reference import finite_difference, resample_qpos
from omg.tracking.holomotion.runtime import build_g1_state_handles, resolve_robot_xml, set_g1_qpos

try:  # Optional outside tracking/deployment environments.
    import mujoco  # type: ignore
except ImportError:  # pragma: no cover
    mujoco = None


@dataclass(frozen=True)
class HoloMotionClipExportConfig:
    reference: str | Path
    output: str | Path
    reference_fps: float | None = None
    target_fps: float = 50.0
    robot_xml: str | Path | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class HoloMotionClipExportResult:
    output_path: Path
    frames: int
    fps: float


def _require_mujoco() -> Any:
    if mujoco is None:
        raise RuntimeError("Exporting HoloMotion deployment clips requires the mujoco package")
    return mujoco


def _body_names(model: Any) -> list[str]:
    mj = _require_mujoco()
    names: list[str] = []
    for body_id in range(1, int(model.nbody)):  # skip MuJoCo world body
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        if name:
            names.append(str(name))
    return names


def _xyzw_from_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    return quat_wxyz[..., [1, 2, 3, 0]].astype(np.float32, copy=False)


def _angular_velocity_from_xyzw(quat_xyzw: np.ndarray, fps: float) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    out = np.zeros(quat_xyzw.shape[:-1] + (3,), dtype=np.float32)
    if quat_xyzw.shape[0] <= 1:
        return out
    dt = 1.0 / float(fps)
    flat = quat_xyzw.reshape(quat_xyzw.shape[0], -1, 4)
    flat_out = out.reshape(out.shape[0], -1, 3)
    for frame_idx in range(flat.shape[0] - 1):
        r0 = Rotation.from_quat(flat[frame_idx])
        r1 = Rotation.from_quat(flat[frame_idx + 1])
        flat_out[frame_idx] = (r1 * r0.inv()).as_rotvec().astype(np.float32) / dt
    flat_out[-1] = flat_out[-2]
    return out.astype(np.float32, copy=False)


def build_holomotion_deployment_clip(
    qpos_36: np.ndarray,
    *,
    fps: float,
    robot_xml: str | Path | None = None,
) -> dict[str, np.ndarray]:
    mj = _require_mujoco()
    qpos = np.asarray(qpos_36, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos_36 shape (T,36), got {qpos.shape}")
    if qpos.shape[0] <= 0:
        raise ValueError("Cannot export an empty motion clip")

    model = mj.MjModel.from_xml_path(str(resolve_robot_xml(robot_xml)))
    data = mj.MjData(model)
    g1_handles = build_g1_state_handles(model)
    body_names = _body_names(model)
    body_ids = np.asarray(
        [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name) for body_name in body_names],
        dtype=np.int32,
    )
    if np.any(body_ids < 0):
        missing = [body_names[idx] for idx, body_id in enumerate(body_ids) if body_id < 0]
        raise KeyError(f"MuJoCo body names not found while exporting HoloMotion clip: {missing}")

    global_translation = np.zeros((qpos.shape[0], len(body_ids), 3), dtype=np.float32)
    global_rotation_wxyz = np.zeros((qpos.shape[0], len(body_ids), 4), dtype=np.float32)
    for frame_idx, qpos_frame in enumerate(qpos):
        set_g1_qpos(model, data, g1_handles, qpos_frame)
        global_translation[frame_idx] = np.asarray(data.xpos[body_ids], dtype=np.float32)
        global_rotation_wxyz[frame_idx] = np.asarray(data.xquat[body_ids], dtype=np.float32)

    ref_dof_pos = qpos[:, 7:36].astype(np.float32, copy=False)
    ref_dof_vel = finite_difference(ref_dof_pos, fps)
    global_rotation_quat = _xyzw_from_wxyz(global_rotation_wxyz)
    return {
        "ref_dof_pos": ref_dof_pos,
        "ref_dof_vel": ref_dof_vel,
        "ref_global_translation": global_translation,
        "ref_global_rotation_quat": global_rotation_quat,
        "ref_global_velocity": finite_difference(global_translation, fps),
        "ref_global_angular_velocity": _angular_velocity_from_xyzw(global_rotation_quat, fps),
        "fps": np.asarray([float(fps)], dtype=np.float32),
        "joint_names": np.asarray(G1_JOINT_NAMES, dtype=np.str_),
        "body_names": np.asarray(body_names, dtype=np.str_),
    }


def export_holomotion_deployment_clip(config: HoloMotionClipExportConfig) -> HoloMotionClipExportResult:
    reference = load_reference_motion(config.reference, fps=config.reference_fps)
    qpos = resample_qpos(reference.qpos_36, source_fps=reference.fps, target_fps=float(config.target_fps))
    payload = build_holomotion_deployment_clip(qpos, fps=float(config.target_fps), robot_xml=config.robot_xml)
    metadata = {
        "source_reference": str(reference.path),
        "source_fps": float(reference.fps),
        "target_fps": float(config.target_fps),
        "frames": int(qpos.shape[0]),
        "format": "holomotion_deployment_npz",
        "quat_order": "xyzw",
        "joint_order": "g1_urdf",
        "body_order": "mujoco_body_order_without_world",
    }
    if reference.metadata:
        metadata["source_metadata"] = reference.metadata
    if config.metadata:
        metadata.update(config.metadata)
    payload["metadata"] = np.asarray([json.dumps(metadata, sort_keys=True)], dtype=np.str_)

    output_path = Path(config.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return HoloMotionClipExportResult(output_path=output_path, frames=int(qpos.shape[0]), fps=float(config.target_fps))
