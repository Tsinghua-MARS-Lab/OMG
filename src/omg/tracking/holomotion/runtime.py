from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from omg.runtime.onnx_providers import (
    parse_onnx_providers,
    prepare_onnx_provider_runtime,
    validate_active_onnx_providers,
    validate_onnx_providers,
)

from omg.core.paths import resolve_repo_path
from omg.robots.g1.constants import G1_JOINT_NAMES

try:  # Optional outside tracking runtime.
    import mujoco  # type: ignore
except ImportError:  # pragma: no cover
    mujoco = None


HOLOMOTION_SCENE_XML = resolve_repo_path("assets/holomotion/g1_29dof/scene_29dof.xml")
_SESSION_MODEL_PATHS: dict[int, Path] = {}



@dataclass
class HoloMotionMetadata:
    joint_names: list[str]
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    joint_stiffness: np.ndarray
    joint_damping: np.ndarray
    n_fut_frames: int
    context_length: int = 1


@dataclass
class HoloMotionHandles:
    joint_names: list[str]
    joint_qpos_adr: np.ndarray
    joint_dof_adr: np.ndarray
    actuator_ids: np.ndarray
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    onnx_to_g1: np.ndarray
    g1_to_onnx: np.ndarray


def _require_mujoco() -> Any:
    if mujoco is None:
        raise RuntimeError("HoloMotion tracking requires the mujoco package")
    return mujoco


def build_onnx_session(model_path: str | Path, providers: str | list[str]):
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("HoloMotion ONNX tracking requires onnxruntime") from exc
    provider_list = parse_onnx_providers(providers)
    validate_onnx_providers(provider_list, ort.get_available_providers())
    prepare_onnx_provider_runtime(provider_list)
    resolved_model_path = Path(model_path).expanduser().resolve()
    session = ort.InferenceSession(str(resolved_model_path), providers=provider_list)
    _SESSION_MODEL_PATHS[id(session)] = resolved_model_path
    validate_active_onnx_providers(provider_list, session.get_providers())
    return session


def name_id(model: Any, obj_type: int, name: str) -> int:
    mj = _require_mujoco()
    idx = mj.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise KeyError(f"MuJoCo object not found: {name}")
    return int(idx)


def build_g1_state_handles(model: Any) -> dict[str, Any]:
    mj = _require_mujoco()
    joint_qpos_adr = []
    joint_dof_adr = []
    for name in G1_JOINT_NAMES:
        jid = name_id(model, mj.mjtObj.mjOBJ_JOINT, name)
        joint_qpos_adr.append(int(model.jnt_qposadr[jid]))
        joint_dof_adr.append(int(model.jnt_dofadr[jid]))
    pelvis_body_id = name_id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")
    gyro_sensor_name = None
    for candidate in ("imu_gyro", "imu-pelvis-angular-velocity"):
        idx = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, candidate)
        if idx >= 0:
            gyro_sensor_name = candidate
            break
    if gyro_sensor_name is None:
        raise KeyError("MuJoCo gyro sensor not found for HoloMotion runtime")
    pelvis_gyro_id = name_id(model, mj.mjtObj.mjOBJ_SENSOR, gyro_sensor_name)
    return {
        "joint_qpos_adr": np.asarray(joint_qpos_adr, dtype=np.int32),
        "joint_dof_adr": np.asarray(joint_dof_adr, dtype=np.int32),
        "pelvis_body_id": pelvis_body_id,
        "pelvis_gyro_adr": int(model.sensor_adr[pelvis_gyro_id]),
        "pelvis_gyro_dim": int(model.sensor_dim[pelvis_gyro_id]),
    }


def parse_float_array(payload: str, key: str) -> np.ndarray:
    cleaned = payload.replace(",", " ").replace("[", " ").replace("]", " ").replace("\n", " ")
    arr = np.fromstring(cleaned, sep=" ", dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"Failed to parse float metadata '{key}'")
    return arr.astype(np.float32, copy=False)


def parse_joint_names(payload: str) -> list[str]:
    names = [item.strip() for item in payload.split(",") if item.strip()]
    if not names:
        raise ValueError("Failed to parse joint_names metadata")
    return names


def infer_holomotion_obs_schema(
    obs_dim: int,
    *,
    context_length: int | None = None,
    n_fut_frames: int | None = None,
) -> tuple[int, int]:
    current_dim = 132
    future_dim = 39
    if context_length is not None or n_fut_frames is not None:
        if context_length is None or n_fut_frames is None:
            raise ValueError("context_length and n_fut_frames must be provided together")
        expected = int(context_length) * current_dim + int(n_fut_frames) * future_dim
        if expected != int(obs_dim):
            raise ValueError(
                f"HoloMotion obs dim {obs_dim} does not match context_length={context_length} "
                f"and n_fut_frames={n_fut_frames} (expected {expected})"
            )
        if int(context_length) <= 0 or int(n_fut_frames) <= 0:
            raise ValueError("context_length and n_fut_frames must be positive")
        return int(context_length), int(n_fut_frames)

    candidates: list[tuple[int, int]] = []
    for fut in range(1, max(2, int(obs_dim) // future_dim + 1)):
        rem = int(obs_dim) - future_dim * fut
        if rem <= 0:
            break
        if rem % current_dim == 0:
            ctx = rem // current_dim
            if ctx > 0:
                candidates.append((int(ctx), int(fut)))
    if len(candidates) != 1:
        raise ValueError(
            f"Unsupported or ambiguous HoloMotion obs dim {obs_dim}; "
            "export metadata or neighboring config.yaml must provide context_length and n_fut_frames"
        )
    return candidates[0]


def infer_n_fut_frames(obs_dim: int) -> int:
    context_length, n_fut_frames = infer_holomotion_obs_schema(obs_dim)
    if context_length != 1:
        raise ValueError(
            f"HoloMotion obs dim {obs_dim} uses context_length={context_length}; "
            "call infer_holomotion_obs_schema instead"
        )
    return int(n_fut_frames)


def _parse_int_config_value(text: str, key: str) -> int | None:
    import re

    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([0-9]+)\s*$", text)
    return int(match.group(1)) if match else None


def _load_obs_schema_from_neighbor_config(session: Any) -> tuple[int | None, int | None]:
    model_path = _SESSION_MODEL_PATHS.get(id(session))
    if model_path is None:
        return None, None
    for directory in (model_path.parent, model_path.parent.parent):
        config_path = directory / "config.yaml"
        if not config_path.is_file():
            continue
        text = config_path.read_text(errors="ignore")
        context_length = _parse_int_config_value(text, "context_length")
        n_fut_frames = _parse_int_config_value(text, "n_fut_frames")
        if context_length is not None and n_fut_frames is not None:
            return int(context_length), int(n_fut_frames)
    return None, None


def load_holomotion_metadata(session: Any) -> HoloMotionMetadata:
    model_meta = session.get_modelmeta()
    meta = getattr(model_meta, "custom_metadata_map", None) or {}
    required = {"joint_names", "default_joint_pos", "action_scale"}
    missing = sorted(required.difference(meta))
    if missing:
        raise KeyError(f"HoloMotion ONNX metadata missing keys: {missing}")
    stiffness_key = "joint_stiffness" if "joint_stiffness" in meta else "kps"
    damping_key = "joint_damping" if "joint_damping" in meta else "kds"
    if stiffness_key not in meta or damping_key not in meta:
        raise KeyError("HoloMotion ONNX metadata is missing joint stiffness/damping")
    obs_input = next((inp for inp in session.get_inputs() if "obs" in inp.name), None)
    if obs_input is None:
        raise KeyError("HoloMotion ONNX does not expose an obs input")
    if len(obs_input.shape) != 2 or not isinstance(obs_input.shape[-1], int):
        raise ValueError(f"Unsupported HoloMotion obs input shape {obs_input.shape}")
    meta_context = int(meta["context_length"]) if "context_length" in meta else None
    meta_future = int(meta["n_fut_frames"]) if "n_fut_frames" in meta else None
    config_context, config_future = (None, None)
    if meta_context is None or meta_future is None:
        config_context, config_future = _load_obs_schema_from_neighbor_config(session)
    context_length, n_fut_frames = infer_holomotion_obs_schema(
        int(obs_input.shape[-1]),
        context_length=meta_context if meta_context is not None else config_context,
        n_fut_frames=meta_future if meta_future is not None else config_future,
    )
    return HoloMotionMetadata(
        joint_names=parse_joint_names(meta["joint_names"]),
        default_joint_pos=parse_float_array(meta["default_joint_pos"], "default_joint_pos"),
        action_scale=parse_float_array(meta["action_scale"], "action_scale"),
        joint_stiffness=parse_float_array(meta[stiffness_key], stiffness_key),
        joint_damping=parse_float_array(meta[damping_key], damping_key),
        n_fut_frames=n_fut_frames,
        context_length=context_length,
    )


def build_holomotion_handles(model: Any, metadata: HoloMotionMetadata) -> HoloMotionHandles:
    mj = _require_mujoco()
    g1_index = {name: idx for idx, name in enumerate(G1_JOINT_NAMES)}
    unknown = [name for name in metadata.joint_names if name not in g1_index]
    if unknown:
        raise KeyError(f"HoloMotion metadata contains non-G1 joints: {unknown}")
    onnx_to_g1 = np.asarray([g1_index[name] for name in metadata.joint_names], dtype=np.int32)
    g1_to_onnx = np.zeros(len(G1_JOINT_NAMES), dtype=np.int32)
    for onnx_idx, g1_idx in enumerate(onnx_to_g1):
        g1_to_onnx[g1_idx] = int(onnx_idx)
    joint_qpos_adr = []
    joint_dof_adr = []
    actuator_ids = []
    for name in metadata.joint_names:
        jid = name_id(model, mj.mjtObj.mjOBJ_JOINT, name)
        actuator_name = name
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if aid < 0 and actuator_name.endswith("_joint"):
            aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name[: -len("_joint")])
        if aid < 0:
            raise KeyError(f"MuJoCo actuator not found for HoloMotion joint '{name}'")
        joint_qpos_adr.append(int(model.jnt_qposadr[jid]))
        joint_dof_adr.append(int(model.jnt_dofadr[jid]))
        actuator_ids.append(int(aid))
    expected = len(metadata.joint_names)
    for key, arr in (
        ("default_joint_pos", metadata.default_joint_pos),
        ("action_scale", metadata.action_scale),
        ("joint_stiffness", metadata.joint_stiffness),
        ("joint_damping", metadata.joint_damping),
    ):
        if arr.shape != (expected,):
            raise ValueError(f"HoloMotion metadata '{key}' expected shape {(expected,)}, got {arr.shape}")
    return HoloMotionHandles(
        joint_names=list(metadata.joint_names),
        joint_qpos_adr=np.asarray(joint_qpos_adr, dtype=np.int32),
        joint_dof_adr=np.asarray(joint_dof_adr, dtype=np.int32),
        actuator_ids=np.asarray(actuator_ids, dtype=np.int32),
        default_joint_pos=metadata.default_joint_pos.astype(np.float32, copy=False),
        action_scale=metadata.action_scale.astype(np.float32, copy=False),
        kp=metadata.joint_stiffness.astype(np.float32, copy=False),
        kd=metadata.joint_damping.astype(np.float32, copy=False),
        onnx_to_g1=onnx_to_g1,
        g1_to_onnx=g1_to_onnx,
    )


class HoloMotionTrackerSession:
    def __init__(self, session: Any, metadata: HoloMotionMetadata):
        self.session = session
        self.metadata = metadata
        self.obs_input_name = "obs"
        self.kv_input_name = None
        self.step_input_name = None
        self.kv_output_name = None
        self.action_output_name = None
        self.kv_dtype = np.float32
        self.kv_shape = None
        for inp in session.get_inputs():
            if "obs" in inp.name:
                self.obs_input_name = inp.name
            elif "past_key_values" in inp.name:
                self.kv_input_name = inp.name
                self.kv_shape = inp.shape
                if isinstance(inp.type, str) and "float16" in inp.type:
                    self.kv_dtype = np.float16
            elif "step_idx" in inp.name:
                self.step_input_name = inp.name
        for out in session.get_outputs():
            if "present_key_values" in out.name:
                self.kv_output_name = out.name
            elif "actions" in out.name:
                self.action_output_name = out.name
        if self.action_output_name is None:
            self.action_output_name = session.get_outputs()[0].name
        self.kv_cache = None
        self.step_idx = 0
        self.reset()

    def reset(self) -> None:
        self.step_idx = 0
        if self.kv_input_name is not None:
            shape = [int(v) if isinstance(v, int) and v > 0 else 1 for v in self.kv_shape]
            self.kv_cache = np.zeros(shape, dtype=self.kv_dtype)
        else:
            self.kv_cache = None

    def run(self, obs: np.ndarray) -> np.ndarray:
        feed = {self.obs_input_name: np.asarray(obs, dtype=np.float32)}
        output_names = [self.action_output_name]
        if self.kv_input_name is not None:
            if self.kv_cache is None:
                raise RuntimeError("HoloMotion KV-cache was not initialized")
            feed[self.kv_input_name] = self.kv_cache
            if self.step_input_name is not None:
                feed[self.step_input_name] = np.asarray([self.step_idx], dtype=np.int64)
            if self.kv_output_name is not None:
                output_names.append(self.kv_output_name)
        result = self.session.run(output_names, feed)
        action = np.asarray(result[0], dtype=np.float32).reshape(-1)
        if action.shape != (len(self.metadata.joint_names),):
            raise ValueError(f"Expected HoloMotion action shape {(len(self.metadata.joint_names),)}, got {action.shape}")
        if self.kv_output_name is not None and len(result) > 1:
            self.kv_cache = np.asarray(result[1], dtype=self.kv_dtype)
        self.step_idx += 1
        return action


def set_g1_qpos(model: Any, data: Any, g1_handles: dict[str, Any], qpos_36: np.ndarray) -> None:
    mj = _require_mujoco()
    qpos_36 = np.asarray(qpos_36, dtype=np.float32)
    if qpos_36.shape != (36,):
        raise ValueError(f"Expected one qpos_36 frame, got {qpos_36.shape}")
    data.qpos[:7] = qpos_36[:7]
    data.qpos[g1_handles["joint_qpos_adr"]] = qpos_36[7:]
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)


def extract_g1_qpos(data: Any, g1_handles: dict[str, Any]) -> np.ndarray:
    out = np.zeros(36, dtype=np.float32)
    out[:7] = np.asarray(data.qpos[:7], dtype=np.float32)
    out[7:] = np.asarray(data.qpos[g1_handles["joint_qpos_adr"]], dtype=np.float32)
    return out


def apply_holomotion_action_pd(model: Any, data: Any, holomotion_handles: HoloMotionHandles, action: np.ndarray, control_substeps: int, action_clip: float) -> np.ndarray:
    mj = _require_mujoco()
    action = np.asarray(action, dtype=np.float32).reshape(len(holomotion_handles.joint_names))
    if action_clip > 0:
        action = np.clip(action, -float(action_clip), float(action_clip))
    desired_pos = holomotion_handles.default_joint_pos + holomotion_handles.action_scale * action
    for _ in range(int(control_substeps)):
        q = np.asarray(data.qpos[holomotion_handles.joint_qpos_adr], dtype=np.float32)
        qd = np.asarray(data.qvel[holomotion_handles.joint_dof_adr], dtype=np.float32)
        torque = holomotion_handles.kp * (desired_pos - q) - holomotion_handles.kd * qd
        data.ctrl[holomotion_handles.actuator_ids] = torque
        mj.mj_step(model, data)
    return action.astype(np.float32, copy=False)


def resolve_robot_xml(path: str | Path | None) -> Path:
    if path is not None:
        robot_xml = Path(path).expanduser()
        if not robot_xml.is_absolute():
            robot_xml = resolve_repo_path(robot_xml)
        robot_xml = robot_xml.resolve()
        if not robot_xml.exists():
            raise FileNotFoundError(f"Robot XML not found: {robot_xml}")
        return robot_xml
    if not HOLOMOTION_SCENE_XML.exists():
        raise FileNotFoundError(f"Default HoloMotion scene XML not found: {HOLOMOTION_SCENE_XML}")
    return HOLOMOTION_SCENE_XML.resolve()
