from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import torch

from omg.robots.g1.kinematics import G1Kinematics
from omg.utils.rotation_conversions import euler_angles_to_matrix, matrix_to_quaternion


@dataclass(frozen=True)
class UrdfVisual:
    mesh_file: Path
    quat_wxyz: np.ndarray
    xyz: np.ndarray
    material_name: str | None


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    quat_wxyz: np.ndarray
    xyz: np.ndarray
    axis_xyz: np.ndarray | None


def save_video(frames_rgb: np.ndarray, output_path: str | Path, fps: int = 30) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_rgb = np.asarray(frames_rgb, dtype=np.uint8)
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape (T,H,W,3), got {frames_rgb.shape}")
    height, width = frames_rgb.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    try:
        for frame in frames_rgb:
            writer.write(frame[..., ::-1])
    finally:
        writer.release()
    _transcode_mp4_h264_in_place(path)
    return path


def _transcode_mp4_h264_in_place(path: Path) -> None:
    if path.suffix.lower() != ".mp4":
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    temp_path = path.with_name(f"{path.stem}.h264.tmp{path.suffix}")
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        temp_path.replace(path)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        print(f"[WARN] Failed to transcode {path} to H.264; keeping original mp4v video: {exc}")


def _load_qpos(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".npy":
        obj = torch.from_numpy(np.load(path))
    elif path.suffix == ".npz":
        payload = np.load(path, allow_pickle=True)
        for key in ("executed_qpos_36", "tracker_qpos_36", "qpos_36", "pred_qpos_36", "reference_qpos_36", "qpos"):
            if key in payload:
                obj = torch.from_numpy(np.asarray(payload[key], dtype=np.float32))
                break
        else:
            raise ValueError(f"No qpos array found in {path}; keys={payload.files}")
    else:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("qpos_36", "pred_qpos_36", "qpos"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise ValueError(f"No qpos tensor found in {path}")
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
    if not isinstance(obj, torch.Tensor):
        raise ValueError(f"Unsupported qpos payload in {path}")
    return torch.from_numpy(_as_qpos_36_np(obj, name=str(path))).float().cpu()


def _vec(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def _yaw_degrees_from_wxyz(quat_wxyz: np.ndarray) -> float:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Invalid root quaternion for camera heading: {quat_wxyz}")
    w, x, y, z = quat / norm
    return float(np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))))


def _overlay_lines(lines: Sequence[str] | None) -> list[str]:
    if lines is None:
        return []
    return [str(line) for line in lines if str(line) != ""]


def _frame_overlay_lines(lines: Sequence[Sequence[str] | str] | None, frame_count: int) -> list[list[str]]:
    if lines is None:
        return [[] for _ in range(frame_count)]
    if len(lines) != frame_count:
        raise ValueError(f"Expected {frame_count} frame overlay entries, got {len(lines)}")
    formatted = []
    for item in lines:
        if isinstance(item, str):
            formatted.append(_overlay_lines([item]))
        else:
            formatted.append(_overlay_lines(item))
    return formatted


def _parse_xyz(attr: str | None) -> np.ndarray:
    if attr is None:
        return np.zeros(3, dtype=np.float32)
    return np.array([float(value) for value in attr.split()], dtype=np.float32)


def _rpy_to_quat_wxyz(rpy_xyz: np.ndarray) -> np.ndarray:
    rpy = torch.tensor(rpy_xyz, dtype=torch.float32)
    matrix = euler_angles_to_matrix(rpy[[2, 1, 0]], convention="ZYX")
    return matrix_to_quaternion(matrix).cpu().numpy().astype(np.float32)


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def parse_urdf(urdf_path: str | Path) -> dict:
    urdf_path = Path(urdf_path).resolve()
    root = ET.parse(urdf_path).getroot()
    mesh_root = urdf_path.parent / "meshes"

    materials: dict[str, str] = {}
    for material in root.findall("material"):
        color = material.find("color")
        if color is not None and color.get("rgba") is not None:
            materials[material.get("name")] = color.get("rgba")

    link_visuals: dict[str, list[UrdfVisual]] = {}
    for link in root.findall("link"):
        visuals = []
        for visual in link.findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None or mesh.get("filename") is None:
                continue
            mesh_rel = mesh.get("filename")
            if not mesh_rel.startswith("meshes/"):
                raise ValueError(f"Unexpected mesh path in URDF: {mesh_rel}")
            mesh_file = (mesh_root / Path(mesh_rel).name).resolve()
            if not mesh_file.exists():
                raise FileNotFoundError(f"Missing mesh asset: {mesh_file}")
            origin = visual.find("origin")
            material = visual.find("material")
            visuals.append(
                UrdfVisual(
                    mesh_file=mesh_file,
                    quat_wxyz=_rpy_to_quat_wxyz(_parse_xyz(None if origin is None else origin.get("rpy"))),
                    xyz=_parse_xyz(None if origin is None else origin.get("xyz")),
                    material_name=None if material is None else material.get("name"),
                )
            )
        link_visuals[link.get("name")] = visuals

    joints_by_name: dict[str, UrdfJoint] = {}
    child_links = set()
    children_by_parent: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        joint_type = joint.get("type")
        if joint_type not in {"fixed", "revolute"}:
            raise ValueError(f"Unsupported URDF joint type: {joint_type}")
        origin = joint.find("origin")
        axis = joint.find("axis")
        parent_link = joint.find("parent").get("link")
        child_link = joint.find("child").get("link")
        item = UrdfJoint(
            name=joint.get("name"),
            joint_type=joint_type,
            parent_link=parent_link,
            child_link=child_link,
            quat_wxyz=_rpy_to_quat_wxyz(_parse_xyz(None if origin is None else origin.get("rpy"))),
            xyz=_parse_xyz(None if origin is None else origin.get("xyz")),
            axis_xyz=None if axis is None else _parse_xyz(axis.get("xyz")),
        )
        joints_by_name[item.name] = item
        child_links.add(child_link)
        children_by_parent.setdefault(parent_link, []).append(item.name)

    root_links = [name for name in link_visuals if name not in child_links]
    if len(root_links) != 1:
        raise ValueError(f"Expected one root link, got {root_links}")
    return {
        "root_link": root_links[0],
        "materials": materials,
        "link_visuals": link_visuals,
        "joints_by_name": joints_by_name,
        "children_by_parent": children_by_parent,
    }


def build_mjcf(
    urdf_data: dict,
    offscreen_width: int,
    offscreen_height: int,
    scene_preset: str = "studio",
    *,
    body_prefix: str = "",
    body_prefixes: Sequence[str] | None = None,
    ghost_body_prefix: str | None = None,
    ghost_rgba: str = "0.35 0.72 1.0 0.28",
) -> str:
    materials = urdf_data["materials"]
    link_visuals = urdf_data["link_visuals"]
    joints_by_name = urdf_data["joints_by_name"]
    children_by_parent = urdf_data["children_by_parent"]
    root_link = urdf_data["root_link"]
    mesh_name_map: dict[Path, str] = {}
    asset_lines = ["  <asset>"]
    if scene_preset in {"studio", "holomotion"}:
        asset_lines.extend(
            [
                '    <texture name="skybox_tex" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>'
                if scene_preset == "holomotion"
                else '    <texture name="skybox_tex" type="skybox" builtin="gradient" rgb1="0.90 0.95 1.00" rgb2="0.54 0.68 0.84" width="512" height="3072"/>',
                '    <material name="ground_mat" rgba="0.92 0.96 1.00 1" reflectance="0"/>'
                if scene_preset == "holomotion"
                else '    <material name="ground_mat" rgba="0.07 0.14 0.32 1" reflectance="0.06"/>',
                '    <material name="grid_minor_mat" rgba="0.58 0.74 0.95 1" reflectance="0"/>'
                if scene_preset == "holomotion"
                else '    <material name="grid_minor_mat" rgba="0.34 0.50 0.74 1" reflectance="0.02"/>',
                '    <material name="grid_major_mat" rgba="0.30 0.48 0.72 1" reflectance="0"/>'
                if scene_preset == "holomotion"
                else '    <material name="grid_major_mat" rgba="0.56 0.76 0.96 1" reflectance="0.03"/>',
            ]
        )
    for material_name, rgba in sorted(materials.items()):
        asset_lines.append(f'    <material name="{_sanitize_name(material_name)}" rgba="{rgba}" specular="0.15" shininess="0.2"/>')
    for mesh_file in sorted({visual.mesh_file for visuals in link_visuals.values() for visual in visuals}):
        mesh_name = f"mesh_{_sanitize_name(mesh_file.stem)}"
        mesh_name_map[mesh_file] = mesh_name
        asset_lines.append(f'    <mesh name="{mesh_name}" file="{mesh_file}" />')
    asset_lines.append("  </asset>")

    def build_body_xml(
        link_name: str,
        indent: int = 4,
        incoming_joint: UrdfJoint | None = None,
        *,
        prefix_name: str = "",
        override_rgba: str | None = None,
    ) -> list[str]:
        prefix = " " * indent
        body_name = f"{prefix_name}{link_name}"
        lines = []
        if incoming_joint is None:
            lines.append(f'{prefix}<body name="{body_name}">')
            lines.append(f'{prefix}  <freejoint name="{prefix_name}root"/>')
        else:
            lines.append(f'{prefix}<body name="{body_name}" pos="{_vec(incoming_joint.xyz)}" quat="{_vec(incoming_joint.quat_wxyz)}">')
            if incoming_joint.joint_type == "revolute":
                if incoming_joint.axis_xyz is None:
                    raise ValueError(f"Revolute joint missing axis: {incoming_joint.name}")
                lines.append(f'{prefix}  <joint name="{prefix_name}{incoming_joint.name}" type="hinge" axis="{_vec(incoming_joint.axis_xyz)}" damping="0.2"/>')
        for visual_idx, visual in enumerate(link_visuals.get(link_name, [])):
            material_attr = "" if visual.material_name is None else f' material="{_sanitize_name(visual.material_name)}"'
            rgba_attr = ""
            if override_rgba is not None:
                material_attr = ""
                rgba_attr = f' rgba="{override_rgba}"'
            lines.append(
                f'{prefix}  <geom name="{_sanitize_name(prefix_name + link_name)}_visual_{visual_idx}" type="mesh" mesh="{mesh_name_map[visual.mesh_file]}" '
                f'pos="{_vec(visual.xyz)}" quat="{_vec(visual.quat_wxyz)}"{material_attr}{rgba_attr} contype="0" conaffinity="0"/>'
            )
        for child_joint_name in children_by_parent.get(link_name, []):
            child_joint = joints_by_name[child_joint_name]
            lines.extend(
                build_body_xml(
                    child_joint.child_link,
                    indent + 2,
                    incoming_joint=child_joint,
                    prefix_name=prefix_name,
                    override_rgba=override_rgba,
                )
            )
        lines.append(f"{prefix}</body>")
        return lines

    if scene_preset in {"studio", "holomotion"}:
        world_lines = [
            '    <light name="key" pos="-3 -3 5" dir="3 3 -5" diffuse="0.65 0.65 0.65" castshadow="true"/>'
            if scene_preset == "holomotion"
            else '    <light name="key" pos="2.5 -3.0 4.8" dir="-0.35 0.35 -1" directional="true" diffuse="1.0 1.0 1.0" specular="0.15 0.15 0.15" castshadow="true"/>',
            '    <light name="fill" pos="3 4 4.5" dir="-3 -4 -4.5" diffuse="0.35 0.35 0.35" castshadow="false"/>'
            if scene_preset == "holomotion"
            else '    <light name="fill" pos="-3.0 2.5 3.8" dir="0.35 -0.25 -1" directional="true" diffuse="0.55 0.58 0.62" specular="0.05 0.05 0.05" castshadow="false"/>',
            '    <light name="rim" pos="0.0 4.0 3.2" dir="0 -0.6 -1" directional="true" diffuse="0.35 0.38 0.42" specular="0.03 0.03 0.03" castshadow="false"/>',
            '    <geom name="floor" type="plane" size="14 14 0.1" material="ground_mat"/>',
        ]
        for idx in range(-12, 13):
            coord = float(idx)
            major = abs(idx) % 5 == 0
            half = 0.018 if major else 0.01
            mat = "grid_major_mat" if major else "grid_minor_mat"
            world_lines.append(f'    <geom name="grid_x_{idx + 12}" type="box" pos="{coord:.6g} 0 0.001" size="{half:.6g} 12 0.001" material="{mat}" contype="0" conaffinity="0"/>')
            world_lines.append(f'    <geom name="grid_y_{idx + 12}" type="box" pos="0 {coord:.6g} 0.001" size="12 {half:.6g} 0.001" material="{mat}" contype="0" conaffinity="0"/>')
        haze = "0.15 0.25 0.35 1" if scene_preset == "holomotion" else "0.98 0.99 1 1"
        headlight = (
            '    <headlight diffuse="0.75 0.75 0.75" ambient="0.18 0.18 0.18" specular="0.95 0.95 0.95"/>'
            if scene_preset == "holomotion"
            else '    <headlight ambient="0.35 0.35 0.35" diffuse="0.75 0.75 0.75" specular="0.08 0.08 0.08"/>'
        )
    else:
        world_lines = [
            '    <light name="sun" pos="0 0 4.5" dir="0 0 -1"/>',
            '    <geom name="floor" type="plane" size="8 8 0.1" rgba="0.95 0.95 0.95 1"/>',
        ]
        haze = "1 1 1 1"
        headlight = '    <headlight ambient="0.6 0.6 0.6" diffuse="0.8 0.8 0.8" specular="0.15 0.15 0.15"/>'

    if body_prefixes is None:
        robot_prefixes = [body_prefix]
        if ghost_body_prefix is not None:
            robot_prefixes.append(ghost_body_prefix)
    else:
        robot_prefixes = list(body_prefixes)
        if ghost_body_prefix is not None:
            raise ValueError("ghost_body_prefix cannot be combined with body_prefixes")
    if not robot_prefixes:
        raise ValueError("At least one robot body prefix is required")
    if len(set(robot_prefixes)) != len(robot_prefixes):
        raise ValueError(f"Robot body prefixes must be unique, got {robot_prefixes}")

    body_lines: list[str] = []
    for prefix in robot_prefixes:
        override_rgba = ghost_rgba if ghost_body_prefix is not None and prefix == ghost_body_prefix else None
        body_lines.extend(build_body_xml(root_link, prefix_name=prefix, override_rgba=override_rgba))

    return "\n".join(
        [
            '<mujoco model="g1_qpos_viewer_mesh">',
            '  <compiler angle="radian" autolimits="true"/>',
            '  <option timestep="0.0333333" gravity="0 0 -9.81"/>',
            "  <visual>",
            f'    <global offwidth="{offscreen_width}" offheight="{offscreen_height}"/>',
            headlight,
            f'    <rgba haze="{haze}"/>',
            "  </visual>",
            *asset_lines,
            "  <worldbody>",
            *world_lines,
            *body_lines,
            "  </worldbody>",
            "</mujoco>",
        ]
    ) + "\n"


def _make_camera(model: mujoco.MjModel, lookat: np.ndarray, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def _camera_eye(lookat: np.ndarray, distance: float, azimuth: float, elevation: float) -> np.ndarray:
    azimuth_rad = np.deg2rad(float(azimuth))
    elevation_rad = np.deg2rad(float(elevation))
    return np.asarray(
        [
            lookat[0] + distance * np.cos(elevation_rad) * np.sin(azimuth_rad),
            lookat[1] - distance * np.cos(elevation_rad) * np.cos(azimuth_rad),
            lookat[2] + distance * np.sin(elevation_rad),
        ],
        dtype=np.float32,
    )


def _project_world_to_pixels(
    point: np.ndarray,
    *,
    lookat: np.ndarray,
    distance: float,
    azimuth: float,
    elevation: float,
    width: int,
    height: int,
    fovy_degrees: float = 45.0,
) -> tuple[int, int] | None:
    eye = _camera_eye(lookat, distance, azimuth, elevation)
    forward = np.asarray(lookat, dtype=np.float32) - eye
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-8:
        return None
    forward = forward / forward_norm
    up_hint = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-8:
        return None
    right = right / right_norm
    up = np.cross(right, forward)
    rel = np.asarray(point, dtype=np.float32) - eye
    z = float(np.dot(rel, forward))
    if z <= 1e-4:
        return None
    x = float(np.dot(rel, right))
    y = float(np.dot(rel, up))
    fovy = np.deg2rad(float(fovy_degrees))
    fy = 0.5 * float(height) / np.tan(0.5 * fovy)
    fx = fy
    px = int(round(0.5 * float(width) + fx * x / z))
    py = int(round(0.5 * float(height) - fy * y / z))
    if px < -width or px > 2 * width or py < -height or py > 2 * height:
        return None
    return px, py


def _draw_dimensional_label(frame: np.ndarray, text: str, x: int, y: int) -> None:
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x0 = int(x - text_w // 2)
    y0 = int(y)
    pad_x = 8
    pad_y = 5
    cv2.rectangle(
        frame,
        (x0 - pad_x + 5, y0 - text_h - pad_y + 5),
        (x0 + text_w + pad_x + 5, y0 + baseline + pad_y + 5),
        (35, 68, 98),
        -1,
    )
    cv2.rectangle(
        frame,
        (x0 - pad_x, y0 - text_h - pad_y),
        (x0 + text_w + pad_x, y0 + baseline + pad_y),
        (235, 248, 255),
        -1,
    )
    cv2.rectangle(
        frame,
        (x0 - pad_x, y0 - text_h - pad_y),
        (x0 + text_w + pad_x, y0 + baseline + pad_y),
        (20, 60, 90),
        2,
    )
    for offset in range(3, 0, -1):
        cv2.putText(frame, text, (x0 + offset, y0 + offset), font, scale, (25, 75, 115), thickness, cv2.LINE_AA)
    cv2.putText(frame, text, (x0, y0), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x0, y0), font, scale, (90, 200, 255), thickness, cv2.LINE_AA)


def _set_data_qpos(data: mujoco.MjData, qpos_frame: np.ndarray, joint_qposadr: dict[str, int], joint_name_to_qpos_index: dict[str, int]) -> None:
    data.qpos[:7] = qpos_frame[:7]
    for joint_name, qpos_index in joint_name_to_qpos_index.items():
        data.qpos[joint_qposadr[joint_name]] = qpos_frame[7 + qpos_index]


def _set_data_qpos_with_prefix(
    data: mujoco.MjData,
    qpos_frame: np.ndarray,
    joint_qposadr: dict[str, int],
    joint_name_to_qpos_index: dict[str, int],
    *,
    prefix: str = "",
) -> None:
    data.qpos[joint_qposadr[f"{prefix}root"] : joint_qposadr[f"{prefix}root"] + 7] = qpos_frame[:7]
    for joint_name, qpos_index in joint_name_to_qpos_index.items():
        prefixed_joint = f"{prefix}{joint_name}"
        data.qpos[joint_qposadr[prefixed_joint]] = qpos_frame[7 + qpos_index]


def _as_qpos_36_np(qpos_36: torch.Tensor | np.ndarray, *, name: str = "qpos_36") -> np.ndarray:
    if isinstance(qpos_36, torch.Tensor):
        qpos_np = qpos_36.detach().float().cpu().numpy()
    else:
        qpos_np = np.asarray(qpos_36, dtype=np.float32)
    if qpos_np.ndim == 3:
        if qpos_np.shape[0] != 1:
            raise ValueError(f"Expected batch dimension 1 for {name}, got {qpos_np.shape}")
        qpos_np = qpos_np[0]
    if qpos_np.ndim != 2 or qpos_np.shape[-1] != 36:
        raise ValueError(f"Expected {name} shape (T,36), got {qpos_np.shape}")
    return np.asarray(qpos_np, dtype=np.float32)


def _joint_qpos_addresses(model: mujoco.MjModel, joint_names: Iterable[str]) -> dict[str, int]:
    joint_qposadr = {}
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name} missing in MuJoCo model")
        joint_qposadr[joint_name] = int(model.jnt_qposadr[joint_id])
    return joint_qposadr


def _camera_azimuths(camera_view: str, iso_azimuth: float, side_azimuth: float) -> list[float]:
    if camera_view == "iso":
        return [iso_azimuth]
    if camera_view == "side":
        return [side_azimuth]
    if camera_view == "front":
        return [180.0]
    if camera_view == "back":
        return [0.0]
    if camera_view == "both":
        return [iso_azimuth, side_azimuth]
    raise ValueError(f"Unsupported camera_view: {camera_view}")


def render_qpos_frames(
    qpos_36: np.ndarray,
    body_pos_w: np.ndarray,
    kinematics: G1Kinematics,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    width: int = 1280,
    height: int = 720,
    elevation: float = -18.0,
    camera_azimuths: list[float] | None = None,
    follow_mode: str = "xy",
    title: str = "G1 Motion",
    overlay_lines: Sequence[str] | None = None,
    music_end_frame: int | None = None,
    music_ended_message: str = "Music ended; using null audio",
    per_frame_info_lines: list[list[str]] | None = None,
    camera_distance_scale: float = 1.0,
) -> np.ndarray:
    if camera_azimuths is None:
        camera_azimuths = [135.0]
    view_width = width if len(camera_azimuths) == 1 else width // len(camera_azimuths)
    renderer = mujoco.Renderer(model, height=height, width=view_width)
    joint_qposadr = {}
    for joint_name in kinematics.joint_order:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name} missing in MuJoCo model")
        joint_qposadr[joint_name] = int(model.jnt_qposadr[joint_id])

    all_body_pos = body_pos_w.reshape(-1, 3)
    center = 0.5 * (all_body_pos.min(axis=0) + all_body_pos.max(axis=0))
    radius = max(float(np.linalg.norm(all_body_pos.max(axis=0) - all_body_pos.min(axis=0))), 1.5)
    distance = radius * 1.35 * float(camera_distance_scale)
    frames = []
    info_lines = _overlay_lines(overlay_lines)
    if per_frame_info_lines is None:
        per_frame_info_lines = [[] for _ in range(len(qpos_36))]
    if len(per_frame_info_lines) != len(qpos_36):
        raise ValueError(f"Expected {len(qpos_36)} per-frame info entries, got {len(per_frame_info_lines)}")
    extra_status_lines = 1 if music_end_frame is not None else 0
    max_frame_info_lines = max((len(lines) for lines in per_frame_info_lines), default=0)
    title_pad = 24 + 20 * (len(info_lines) + max_frame_info_lines + extra_status_lines)
    for frame_idx in range(len(qpos_36)):
        _set_data_qpos(data, qpos_36[frame_idx], joint_qposadr, kinematics.joint_name_to_qpos_index)
        mujoco.mj_forward(model, data)
        root_pos = body_pos_w[frame_idx, 0]
        lookat = center.copy()
        if follow_mode == "xyz":
            lookat = root_pos.copy()
            lookat[2] += 0.35
        elif follow_mode == "xy":
            lookat[:2] = root_pos[:2]
            lookat[2] = center[2] + 0.35
        else:
            lookat[2] += 0.35
        rendered = []
        frame_azimuths = camera_azimuths
        if follow_mode == "heading":
            yaw = _yaw_degrees_from_wxyz(qpos_36[frame_idx, 3:7])
            frame_azimuths = [yaw + azimuth for azimuth in camera_azimuths]
            lookat[:2] = root_pos[:2]
            lookat[2] = root_pos[2] + 0.35
        for azimuth in frame_azimuths:
            camera = _make_camera(model, lookat, distance, azimuth, elevation)
            renderer.update_scene(data, camera=camera)
            rendered.append(renderer.render())
        frame = np.concatenate(rendered, axis=1)
        canvas = np.full((frame.shape[0] + title_pad, frame.shape[1], 3), 255, dtype=np.uint8)
        canvas[title_pad:] = frame
        cv2.putText(
            canvas,
            f"{title} | frame {frame_idx + 1}/{len(qpos_36)}",
            (16, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        frame_info_lines = info_lines + per_frame_info_lines[frame_idx]
        for line_idx, line in enumerate(frame_info_lines):
            cv2.putText(
                canvas,
                line,
                (16, 18 + 20 * (line_idx + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        if music_end_frame is not None and frame_idx >= int(music_end_frame):
            cv2.putText(
                canvas,
                music_ended_message,
                (16, 18 + 20 * (len(info_lines) + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        frames.append(canvas)
    renderer.close()
    return np.stack(frames, axis=0)


def render_qpos_overlay_frames(
    qpos_36: np.ndarray,
    ghost_qpos_36: np.ndarray,
    body_pos_w: np.ndarray,
    ghost_body_pos_w: np.ndarray,
    kinematics: G1Kinematics,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    width: int = 1280,
    height: int = 720,
    elevation: float = -18.0,
    camera_azimuths: list[float] | None = None,
    follow_mode: str = "xy",
    title: str = "Generated + Ground Truth",
    overlay_lines: Sequence[str] | None = None,
    music_end_frame: int | None = None,
    music_ended_message: str = "Music ended; using null audio",
    per_frame_info_lines: list[list[str]] | None = None,
    ghost_prefix: str = "gt_",
    camera_distance_scale: float = 1.0,
) -> np.ndarray:
    if camera_azimuths is None:
        camera_azimuths = [135.0]
    view_width = width if len(camera_azimuths) == 1 else width // len(camera_azimuths)
    renderer = mujoco.Renderer(model, height=height, width=view_width)
    joint_names = ["root", *kinematics.joint_order, f"{ghost_prefix}root", *(f"{ghost_prefix}{name}" for name in kinematics.joint_order)]
    joint_qposadr = _joint_qpos_addresses(model, joint_names)

    all_body_pos = np.concatenate([body_pos_w.reshape(-1, 3), ghost_body_pos_w.reshape(-1, 3)], axis=0)
    center = 0.5 * (all_body_pos.min(axis=0) + all_body_pos.max(axis=0))
    radius = max(float(np.linalg.norm(all_body_pos.max(axis=0) - all_body_pos.min(axis=0))), 1.5)
    distance = radius * 1.45 * float(camera_distance_scale)
    frames = []
    info_lines = _overlay_lines(overlay_lines)
    per_frame_info_lines = [[] for _ in range(len(qpos_36))] if per_frame_info_lines is None else per_frame_info_lines
    extra_status_lines = 1 if music_end_frame is not None else 0
    title_pad = 24 + 20 * (len(info_lines) + extra_status_lines)
    for frame_idx in range(len(qpos_36)):
        _set_data_qpos_with_prefix(data, qpos_36[frame_idx], joint_qposadr, kinematics.joint_name_to_qpos_index)
        _set_data_qpos_with_prefix(data, ghost_qpos_36[frame_idx], joint_qposadr, kinematics.joint_name_to_qpos_index, prefix=ghost_prefix)
        mujoco.mj_forward(model, data)
        root_pos = body_pos_w[frame_idx, 0]
        ghost_root_pos = ghost_body_pos_w[frame_idx, 0]
        root_center = 0.5 * (root_pos + ghost_root_pos)
        lookat = center.copy()
        if follow_mode == "xyz":
            lookat = root_center.copy()
            lookat[2] += 0.35
        elif follow_mode == "xy":
            lookat[:2] = root_center[:2]
            lookat[2] = center[2] + 0.35
        else:
            lookat[2] += 0.35
        rendered = []
        frame_azimuths = camera_azimuths
        if follow_mode == "heading":
            yaw = _yaw_degrees_from_wxyz(qpos_36[frame_idx, 3:7])
            frame_azimuths = [yaw + azimuth for azimuth in camera_azimuths]
            lookat[:2] = root_center[:2]
            lookat[2] = root_center[2] + 0.35
        for azimuth in frame_azimuths:
            camera = _make_camera(model, lookat, distance, azimuth, elevation)
            renderer.update_scene(data, camera=camera)
            rendered.append(renderer.render())
        frame = np.concatenate(rendered, axis=1)
        canvas = np.full((frame.shape[0] + title_pad, frame.shape[1], 3), 255, dtype=np.uint8)
        canvas[title_pad:] = frame
        cv2.putText(
            canvas,
            f"{title} | frame {frame_idx + 1}/{len(qpos_36)}",
            (16, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        frame_info_lines = info_lines + per_frame_info_lines[frame_idx]
        for line_idx, line in enumerate(frame_info_lines):
            cv2.putText(
                canvas,
                line,
                (16, 18 + 20 * (line_idx + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        if music_end_frame is not None and frame_idx >= int(music_end_frame):
            cv2.putText(
                canvas,
                music_ended_message,
                (16, 18 + 20 * (len(info_lines) + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        frames.append(canvas)
    renderer.close()
    return np.stack(frames, axis=0)


def _grid_offsets(num_items: int, spacing: float) -> np.ndarray:
    if num_items <= 0:
        raise ValueError(f"num_items must be positive, got {num_items}")
    cols = int(np.ceil(np.sqrt(float(num_items))))
    rows = int(np.ceil(float(num_items) / float(cols)))
    offsets = []
    for index in range(num_items):
        row = index // cols
        col = index % cols
        x = (float(col) - 0.5 * float(cols - 1)) * float(spacing)
        y = (0.5 * float(rows - 1) - float(row)) * float(spacing)
        offsets.append([x, y, 0.0])
    return np.asarray(offsets, dtype=np.float32)


def _circle_offsets(num_items: int, spacing: float) -> np.ndarray:
    if num_items <= 0:
        raise ValueError(f"num_items must be positive, got {num_items}")
    if num_items == 1:
        return np.zeros((1, 3), dtype=np.float32)
    radius = float(spacing) * float(num_items) / (2.0 * np.pi)
    angles = np.linspace(0.0, 2.0 * np.pi, num_items, endpoint=False)
    return np.stack(
        [
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros_like(angles),
        ],
        axis=1,
    ).astype(np.float32)


def render_qpos_multi_frames(
    qpos_36_list: Sequence[torch.Tensor | np.ndarray],
    kinematics: G1Kinematics,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    offsets: np.ndarray,
    labels: Sequence[str] | None = None,
    width: int = 1280,
    height: int = 720,
    elevation: float = -18.0,
    camera_azimuths: list[float] | None = None,
    follow_mode: str = "none",
    title: str = "OMG Multi-Robot",
    overlay_lines: Sequence[str] | None = None,
    camera_distance_scale: float = 1.0,
    draw_labels: bool = True,
    label_height: float = 1.25,
) -> np.ndarray:
    if camera_azimuths is None:
        camera_azimuths = [135.0]
    qpos_items = [_as_qpos_36_np(qpos, name=f"qpos_36[{idx}]") for idx, qpos in enumerate(qpos_36_list)]
    if not qpos_items:
        raise ValueError("qpos_36_list must contain at least one motion")
    frame_count = min(item.shape[0] for item in qpos_items)
    if frame_count <= 0:
        raise ValueError("All qpos motions must contain at least one frame")
    offsets_np = np.asarray(offsets, dtype=np.float32)
    if offsets_np.shape != (len(qpos_items), 3):
        raise ValueError(f"Expected offsets shape ({len(qpos_items)},3), got {offsets_np.shape}")
    labels = [f"robot {idx + 1}" for idx in range(len(qpos_items))] if labels is None else list(labels)
    if len(labels) != len(qpos_items):
        raise ValueError(f"Expected {len(qpos_items)} labels, got {len(labels)}")

    qpos_offset_items = []
    body_pos_items = []
    for qpos, offset in zip(qpos_items, offsets_np, strict=True):
        shifted = qpos[:frame_count].copy()
        shifted[:, :3] += offset[None, :]
        qpos_offset_items.append(shifted)
        body_state = kinematics.forward_kinematics(torch.from_numpy(shifted))
        body_pos_items.append(body_state["body_pos_w"].cpu().numpy())

    prefixes = [f"r{idx}_" for idx in range(len(qpos_items))]
    joint_names = []
    for prefix in prefixes:
        joint_names.extend([f"{prefix}root", *(f"{prefix}{name}" for name in kinematics.joint_order)])
    joint_qposadr = _joint_qpos_addresses(model, joint_names)

    all_body_pos = np.concatenate([body_pos.reshape(-1, 3) for body_pos in body_pos_items], axis=0)
    center = 0.5 * (all_body_pos.min(axis=0) + all_body_pos.max(axis=0))
    radius = max(float(np.linalg.norm(all_body_pos.max(axis=0) - all_body_pos.min(axis=0))), 2.5)
    distance = radius * 1.35 * float(camera_distance_scale)
    view_width = width if len(camera_azimuths) == 1 else width // len(camera_azimuths)
    renderer = mujoco.Renderer(model, height=height, width=view_width)
    info_lines = _overlay_lines(overlay_lines)
    title_pad = 24 + 20 * len(info_lines)
    frames = []
    for frame_idx in range(frame_count):
        root_positions = []
        for prefix, qpos in zip(prefixes, qpos_offset_items, strict=True):
            _set_data_qpos_with_prefix(
                data,
                qpos[frame_idx],
                joint_qposadr,
                kinematics.joint_name_to_qpos_index,
                prefix=prefix,
            )
            root_positions.append(qpos[frame_idx, :3])
        mujoco.mj_forward(model, data)
        lookat = center.copy()
        if follow_mode in {"xy", "xyz"}:
            root_center = np.mean(np.asarray(root_positions, dtype=np.float32), axis=0)
            if follow_mode == "xyz":
                lookat = root_center.copy()
                lookat[2] += 0.35
            else:
                lookat[:2] = root_center[:2]
                lookat[2] = center[2] + 0.35
        else:
            lookat[2] += 0.35
        rendered = []
        for azimuth in camera_azimuths:
            camera = _make_camera(model, lookat, distance, azimuth, elevation)
            renderer.update_scene(data, camera=camera)
            rendered_frame = renderer.render()
            if draw_labels:
                for label, root_pos in zip(labels, root_positions, strict=True):
                    label_point = np.asarray(root_pos, dtype=np.float32).copy()
                    label_point[2] += float(label_height)
                    pixels = _project_world_to_pixels(
                        label_point,
                        lookat=lookat,
                        distance=distance,
                        azimuth=azimuth,
                        elevation=elevation,
                        width=view_width,
                        height=height,
                    )
                    if pixels is not None:
                        _draw_dimensional_label(rendered_frame, str(label), pixels[0], pixels[1])
            rendered.append(rendered_frame)
        frame = np.concatenate(rendered, axis=1)
        canvas = np.full((frame.shape[0] + title_pad, frame.shape[1], 3), 255, dtype=np.uint8)
        canvas[title_pad:] = frame
        cv2.putText(
            canvas,
            f"{title} | frame {frame_idx + 1}/{frame_count}",
            (16, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        for line_idx, line in enumerate(info_lines):
            cv2.putText(
                canvas,
                line,
                (16, 18 + 20 * (line_idx + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        frames.append(canvas)
    renderer.close()
    return np.stack(frames, axis=0)


def render_qpos_video(
    qpos_36: torch.Tensor | np.ndarray,
    output_path: str | Path,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    elevation: float = -18.0,
    camera_view: str = "iso",
    iso_azimuth: float = 135.0,
    side_azimuth: float = 90.0,
    follow_mode: str = "xy",
    kinematics_path: str | Path = "assets/robots/g1/g1_kinematics.json",
    urdf_path: str | Path = "assets/robots/g1/g1_29dof.urdf",
    scene_preset: str = "studio",
    title: str = "G1 Motion",
    overlay_lines: Sequence[str] | None = None,
    frame_overlay_lines: Sequence[Sequence[str] | str] | None = None,
    music_end_frame: int | None = None,
    ended_message: str = "Music ended; using null audio",
    per_frame_info_lines: list[list[str]] | None = None,
    camera_distance_scale: float = 1.0,
) -> Path:
    if isinstance(qpos_36, torch.Tensor):
        qpos_np = qpos_36.detach().float().cpu().numpy()
    else:
        qpos_np = np.asarray(qpos_36, dtype=np.float32)
    if qpos_np.ndim == 3:
        if qpos_np.shape[0] != 1:
            raise ValueError(f"Expected batch dimension 1, got {qpos_np.shape}")
        qpos_np = qpos_np[0]
    if qpos_np.ndim != 2 or qpos_np.shape[-1] != 36:
        raise ValueError(f"Expected qpos_36 shape (T,36), got {qpos_np.shape}")

    if camera_view == "iso":
        camera_azimuths = [iso_azimuth]
    elif camera_view == "side":
        camera_azimuths = [side_azimuth]
    elif camera_view == "front":
        camera_azimuths = [180.0]
    elif camera_view == "back":
        camera_azimuths = [0.0]
    elif camera_view == "both":
        camera_azimuths = [iso_azimuth, side_azimuth]
    else:
        raise ValueError(f"Unsupported camera_view: {camera_view}")

    kinematics = G1Kinematics(kinematics_path)
    body_state = kinematics.forward_kinematics(torch.from_numpy(qpos_np))
    body_pos_w = body_state["body_pos_w"].cpu().numpy()
    urdf_data = parse_urdf(urdf_path)
    xml_string = build_mjcf(
        urdf_data,
        offscreen_width=max(1, width if len(camera_azimuths) == 1 else width // len(camera_azimuths)),
        offscreen_height=height,
        scene_preset=scene_preset,
    )
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    if per_frame_info_lines is None:
        per_frame_info_lines = _frame_overlay_lines(frame_overlay_lines, len(qpos_np))
    frames = render_qpos_frames(
        qpos_np,
        body_pos_w,
        kinematics,
        model,
        data,
        width=width,
        height=height,
        elevation=elevation,
        camera_azimuths=camera_azimuths,
        follow_mode=follow_mode,
        title=title,
        overlay_lines=overlay_lines,
        music_end_frame=music_end_frame,
        music_ended_message=ended_message,
        per_frame_info_lines=per_frame_info_lines,
        camera_distance_scale=camera_distance_scale,
    )
    path = save_video(frames, output_path, fps=fps)
    metadata = {
        "num_frames": int(qpos_np.shape[0]),
        "fps": int(fps),
        "urdf_path": str(Path(urdf_path).resolve()),
        "camera_view": camera_view,
        "follow_mode": follow_mode,
        "scene_preset": scene_preset,
        "music_end_frame": None if music_end_frame is None else int(music_end_frame),
        "camera_distance_scale": float(camera_distance_scale),
    }
    Path(path).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def render_qpos_comparison_video(
    left_qpos_36: torch.Tensor | np.ndarray,
    right_qpos_36: torch.Tensor | np.ndarray,
    output_path: str | Path,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    elevation: float = -18.0,
    camera_view: str = "iso",
    iso_azimuth: float = 135.0,
    side_azimuth: float = 90.0,
    follow_mode: str = "xy",
    kinematics_path: str | Path = "assets/robots/g1/g1_kinematics.json",
    urdf_path: str | Path = "assets/robots/g1/g1_29dof.urdf",
    scene_preset: str = "studio",
    left_title: str = "Generated",
    right_title: str = "Ground Truth",
    overlay_lines: Sequence[str] | None = None,
    music_end_frame: int | None = None,
    ended_message: str = "Music ended; using null audio",
    per_frame_info_lines: list[list[str]] | None = None,
    left_ghost_qpos_36: torch.Tensor | np.ndarray | None = None,
    left_ghost_alpha: float = 0.28,
    camera_distance_scale: float = 1.0,
) -> Path:
    def as_qpos(value: torch.Tensor | np.ndarray, name: str) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            qpos = value.detach().float().cpu().numpy()
        else:
            qpos = np.asarray(value, dtype=np.float32)
        if qpos.ndim == 3:
            if qpos.shape[0] != 1:
                raise ValueError(f"Expected {name} batch dimension 1, got {qpos.shape}")
            qpos = qpos[0]
        if qpos.ndim != 2 or qpos.shape[-1] != 36:
            raise ValueError(f"Expected {name} shape (T,36), got {qpos.shape}")
        return qpos

    left_np = as_qpos(left_qpos_36, "left_qpos_36")
    right_np = as_qpos(right_qpos_36, "right_qpos_36")
    num_frames = min(int(left_np.shape[0]), int(right_np.shape[0]))
    if num_frames <= 0:
        raise ValueError("Comparison motions must contain at least one frame")
    left_np = left_np[:num_frames]
    right_np = right_np[:num_frames]

    ghost_np = None if left_ghost_qpos_36 is None else as_qpos(left_ghost_qpos_36, "left_ghost_qpos_36")
    if ghost_np is not None:
        num_frames = min(num_frames, int(ghost_np.shape[0]))
        if num_frames <= 0:
            raise ValueError("Comparison motions must contain at least one frame")
        left_np = left_np[:num_frames]
        right_np = right_np[:num_frames]
        ghost_np = ghost_np[:num_frames]

    camera_azimuths = _camera_azimuths(camera_view, iso_azimuth, side_azimuth)

    panel_width = max(1, width // 2)
    kinematics = G1Kinematics(kinematics_path)
    urdf_data = parse_urdf(urdf_path)
    xml_string = build_mjcf(
        urdf_data,
        offscreen_width=max(1, panel_width if len(camera_azimuths) == 1 else panel_width // len(camera_azimuths)),
        offscreen_height=height,
        scene_preset=scene_preset,
    )

    def render_one(qpos_np: np.ndarray, title: str) -> np.ndarray:
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        body_state = kinematics.forward_kinematics(torch.from_numpy(qpos_np))
        body_pos_w = body_state["body_pos_w"].cpu().numpy()
        return render_qpos_frames(
            qpos_np,
            body_pos_w,
            kinematics,
            model,
            data,
            width=panel_width,
            height=height,
            elevation=elevation,
            camera_azimuths=camera_azimuths,
            follow_mode=follow_mode,
            title=title,
            overlay_lines=overlay_lines,
            music_end_frame=music_end_frame,
            music_ended_message=ended_message,
            per_frame_info_lines=per_frame_info_lines,
            camera_distance_scale=camera_distance_scale,
        )

    if ghost_np is None:
        left_frames = render_one(left_np, left_title)
    else:
        ghost_alpha = min(max(float(left_ghost_alpha), 0.0), 1.0)
        overlay_xml_string = build_mjcf(
            urdf_data,
            offscreen_width=max(1, panel_width if len(camera_azimuths) == 1 else panel_width // len(camera_azimuths)),
            offscreen_height=height,
            scene_preset=scene_preset,
            ghost_body_prefix="gt_",
            ghost_rgba=f"0.24 0.68 1.0 {ghost_alpha:.6g}",
        )
        overlay_model = mujoco.MjModel.from_xml_string(overlay_xml_string)
        overlay_data = mujoco.MjData(overlay_model)
        left_body_state = kinematics.forward_kinematics(torch.from_numpy(left_np))
        ghost_body_state = kinematics.forward_kinematics(torch.from_numpy(ghost_np))
        left_frames = render_qpos_overlay_frames(
            left_np,
            ghost_np,
            left_body_state["body_pos_w"].cpu().numpy(),
            ghost_body_state["body_pos_w"].cpu().numpy(),
            kinematics,
            overlay_model,
            overlay_data,
            width=panel_width,
            height=height,
            elevation=elevation,
            camera_azimuths=camera_azimuths,
            follow_mode=follow_mode,
            title=left_title,
            overlay_lines=overlay_lines,
            music_end_frame=music_end_frame,
            music_ended_message=ended_message,
            per_frame_info_lines=per_frame_info_lines,
            camera_distance_scale=camera_distance_scale,
        )
    right_frames = render_one(right_np, right_title)
    frames = np.concatenate([left_frames, right_frames], axis=2)
    path = save_video(frames, output_path, fps=fps)
    metadata = {
        "num_frames": int(num_frames),
        "fps": int(fps),
        "urdf_path": str(Path(urdf_path).resolve()),
        "camera_view": camera_view,
        "follow_mode": follow_mode,
        "scene_preset": scene_preset,
        "left_title": left_title,
        "right_title": right_title,
        "music_end_frame": None if music_end_frame is None else int(music_end_frame),
        "left_ghost_qpos": bool(ghost_np is not None),
        "left_ghost_alpha": None if ghost_np is None else float(left_ghost_alpha),
        "camera_distance_scale": float(camera_distance_scale),
    }
    Path(path).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


HUMAN_22_EDGES = (
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),
)


def _human_joints_np(human_motion: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(human_motion, torch.Tensor):
        arr = human_motion.detach().float().cpu().numpy()
    else:
        arr = np.asarray(human_motion, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[-1] % 3 != 0:
            raise ValueError(f"Expected flattened xyz human motion, got {arr.shape}")
        arr = arr.reshape(arr.shape[0], arr.shape[-1] // 3, 3)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected human motion shape (T,J,3) or (T,J*3), got {arr.shape}")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"Human motion is empty: {arr.shape}")
    return arr.astype(np.float32, copy=False)


def render_human_motion_frames(
    human_motion: torch.Tensor | np.ndarray,
    width: int = 1280,
    height: int = 720,
    title: str = "Human Reference",
    overlay_lines: Sequence[str] | None = None,
    ended_frame: int | None = None,
    ended_message: str = "Human reference ended; using null reference",
) -> np.ndarray:
    joints = _human_joints_np(human_motion)
    info_lines = _overlay_lines(overlay_lines)
    extra_status_lines = 1 if ended_frame is not None else 0
    title_pad = 24 + 20 * (len(info_lines) + extra_status_lines)
    body_h = int(height)
    canvas_h = body_h + title_pad
    valid_points = joints.reshape(-1, 3)
    center = 0.5 * (valid_points.min(axis=0) + valid_points.max(axis=0))
    span = np.maximum(valid_points.max(axis=0) - valid_points.min(axis=0), 1e-6)
    scale = 0.74 * min(float(width) / max(float(span[0]), 1e-6), float(body_h) / max(float(span[2]), 1e-6))
    frames = []
    edges = HUMAN_22_EDGES if joints.shape[1] >= 22 else tuple((idx, idx + 1) for idx in range(joints.shape[1] - 1))
    for frame_idx, pose in enumerate(joints):
        canvas = np.full((canvas_h, int(width), 3), 255, dtype=np.uint8)
        cv2.putText(canvas, f"{title} | frame {frame_idx + 1}/{len(joints)}", (16, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        for line_idx, line in enumerate(info_lines):
            cv2.putText(canvas, line, (16, 18 + 20 * (line_idx + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        if ended_frame is not None and frame_idx >= int(ended_frame):
            cv2.putText(canvas, ended_message, (16, 18 + 20 * (len(info_lines) + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        pts = []
        for joint in pose:
            x = int(round(width * 0.5 + (float(joint[0] - center[0]) * scale)))
            y = int(round(title_pad + body_h * 0.55 - (float(joint[2] - center[2]) * scale)))
            pts.append((x, y))
        for a, b in edges:
            if a < len(pts) and b < len(pts):
                cv2.line(canvas, pts[a], pts[b], (40, 95, 170), 3, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(canvas, pt, 4, (18, 38, 70), -1, cv2.LINE_AA)
        frames.append(canvas)
    return np.stack(frames, axis=0)


def render_human_motion_video(
    human_motion: torch.Tensor | np.ndarray,
    output_path: str | Path,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    title: str = "Human Reference",
    overlay_lines: Sequence[str] | None = None,
    ended_frame: int | None = None,
) -> Path:
    frames = render_human_motion_frames(
        human_motion,
        width=width,
        height=height,
        title=title,
        overlay_lines=overlay_lines,
        ended_frame=ended_frame,
    )
    path = save_video(frames, output_path, fps=fps)
    metadata = {
        "num_frames": int(frames.shape[0]),
        "fps": int(fps),
        "ended_frame": None if ended_frame is None else int(ended_frame),
    }
    Path(path).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def render_robot_human_comparison_video(
    qpos_36: torch.Tensor | np.ndarray,
    human_motion: torch.Tensor | np.ndarray,
    output_path: str | Path,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    camera_view: str = "iso",
    follow_mode: str = "xy",
    scene_preset: str = "studio",
    left_title: str = "Generated Robot",
    right_title: str = "Human Reference",
    overlay_lines: Sequence[str] | None = None,
    ended_frame: int | None = None,
    left_ghost_qpos_36: torch.Tensor | np.ndarray | None = None,
    left_ghost_alpha: float = 0.28,
    camera_distance_scale: float = 1.0,
) -> Path:
    def as_qpos(value: torch.Tensor | np.ndarray, name: str) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            qpos = value.detach().float().cpu().numpy()
        else:
            qpos = np.asarray(value, dtype=np.float32)
        if qpos.ndim == 3:
            if qpos.shape[0] != 1:
                raise ValueError(f"Expected {name} batch dimension 1, got {qpos.shape}")
            qpos = qpos[0]
        if qpos.ndim != 2 or qpos.shape[-1] != 36:
            raise ValueError(f"Expected {name} shape (T,36), got {qpos.shape}")
        return qpos

    panel_width = max(1, width // 2)
    qpos_np = as_qpos(qpos_36, "qpos_36")
    human_np = _human_joints_np(human_motion)
    num_frames = min(int(qpos_np.shape[0]), int(human_np.shape[0]))
    ghost_np = None if left_ghost_qpos_36 is None else as_qpos(left_ghost_qpos_36, "left_ghost_qpos_36")
    if ghost_np is not None:
        num_frames = min(num_frames, int(ghost_np.shape[0]))
    if num_frames <= 0:
        raise ValueError("Robot/human comparison inputs must contain at least one frame")
    qpos_np = qpos_np[:num_frames]
    human_np = human_np[:num_frames]
    if ghost_np is not None:
        ghost_np = ghost_np[:num_frames]

    if ghost_np is None:
        robot_path = Path(output_path).with_name(f"{Path(output_path).stem}_robot_panel.mp4")
        robot_video = render_qpos_video(
            qpos_np,
            robot_path,
            fps=fps,
            width=panel_width,
            height=height,
            camera_view=camera_view,
            follow_mode=follow_mode,
            scene_preset=scene_preset,
            title=left_title,
            overlay_lines=overlay_lines,
            music_end_frame=ended_frame,
            ended_message="Human reference ended; using null reference",
            camera_distance_scale=camera_distance_scale,
        )
        robot_frames = _read_video_frames(robot_video)
        try:
            Path(robot_video).unlink(missing_ok=True)
            Path(robot_video).with_suffix(".json").unlink(missing_ok=True)
        except OSError:
            pass
    else:
        kinematics = G1Kinematics("assets/robots/g1/g1_kinematics.json")
        urdf_data = parse_urdf("assets/robots/g1/g1_29dof.urdf")
        camera_azimuths = _camera_azimuths(camera_view, 135.0, 90.0)
        ghost_alpha = min(max(float(left_ghost_alpha), 0.0), 1.0)
        xml_string = build_mjcf(
            urdf_data,
            offscreen_width=max(1, panel_width if len(camera_azimuths) == 1 else panel_width // len(camera_azimuths)),
            offscreen_height=height,
            scene_preset=scene_preset,
            ghost_body_prefix="gt_",
            ghost_rgba=f"0.24 0.68 1.0 {ghost_alpha:.6g}",
        )
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        body_state = kinematics.forward_kinematics(torch.from_numpy(qpos_np))
        ghost_body_state = kinematics.forward_kinematics(torch.from_numpy(ghost_np))
        robot_frames = render_qpos_overlay_frames(
            qpos_np,
            ghost_np,
            body_state["body_pos_w"].cpu().numpy(),
            ghost_body_state["body_pos_w"].cpu().numpy(),
            kinematics,
            model,
            data,
            width=panel_width,
            height=height,
            camera_azimuths=camera_azimuths,
            follow_mode=follow_mode,
            title=left_title,
            overlay_lines=overlay_lines,
            music_end_frame=ended_frame,
            music_ended_message="Human reference ended; using null reference",
            camera_distance_scale=camera_distance_scale,
        )
    human_frames = render_human_motion_frames(
        human_np,
        width=panel_width,
        height=height,
        title=right_title,
        overlay_lines=overlay_lines,
        ended_frame=ended_frame,
    )
    frames = np.concatenate([robot_frames[:num_frames], human_frames[:num_frames]], axis=2)
    path = save_video(frames, output_path, fps=fps)
    metadata = {
        "num_frames": int(num_frames),
        "fps": int(fps),
        "camera_view": camera_view,
        "follow_mode": follow_mode,
        "scene_preset": scene_preset,
        "ended_frame": None if ended_frame is None else int(ended_frame),
        "left_ghost_qpos": bool(ghost_np is not None),
        "left_ghost_alpha": None if ghost_np is None else float(left_ghost_alpha),
        "camera_distance_scale": float(camera_distance_scale),
    }
    Path(path).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def _read_video_frames(video_path: str | Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(frame_bgr[..., ::-1])
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Failed to read rendered video frames from {video_path}")
    return np.stack(frames, axis=0)
