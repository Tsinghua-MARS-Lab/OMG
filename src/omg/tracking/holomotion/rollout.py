from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def default_rollout_path(reference_path: str | Path, output_root: str | Path = "outputs/holomotion_rollouts") -> Path:
    reference = Path(reference_path)
    return (Path(output_root) / f"{reference.stem}_holomotion_tracker.npz").resolve()


def save_tracker_rollout(
    *,
    output_path: str | Path,
    executed_qpos_36: list[np.ndarray],
    reference_qpos_36: list[np.ndarray],
    actions: list[np.ndarray],
    fps: float,
    reference_path: str | Path,
    holomotion_onnx: str | Path,
    robot_xml: str | Path,
    mode: str,
    planner_mode: str | None = None,
    plan_cursor: list[int] | None = None,
    plan_id: list[int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = len(executed_qpos_36)
    if plan_cursor is None:
        plan_cursor = list(range(frames))
    if plan_id is None:
        plan_id = [0 for _ in range(frames)]
    payload = {
        "executed_qpos_36": np.asarray(executed_qpos_36, dtype=np.float32),
        "reference_qpos_36": np.asarray(reference_qpos_36, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "fps": np.asarray([float(fps)], dtype=np.float32),
        "mode": np.asarray([str(mode)], dtype=np.str_),
        "planner_mode": np.asarray(["" if planner_mode is None else str(planner_mode)], dtype=np.str_),
        "plan_cursor": np.asarray(plan_cursor, dtype=np.int32),
        "plan_id": np.asarray(plan_id, dtype=np.int32),
        "reference_path": np.asarray([str(Path(reference_path).expanduser().resolve())], dtype=np.str_),
        "holomotion_onnx": np.asarray([str(Path(holomotion_onnx).expanduser().resolve())], dtype=np.str_),
        "robot_xml": np.asarray([str(Path(robot_xml).expanduser().resolve())], dtype=np.str_),
    }
    if metadata is not None:
        payload["metadata_json"] = np.asarray([json.dumps(metadata, sort_keys=True)], dtype=np.str_)
    np.savez_compressed(path, **payload)
    return path
