from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from omg.generation.metrics import physical_qpos_metrics
from omg.motion.representation import G1MotionRepresentation

DEFAULT_QPOS_KEYS = ("executed_qpos_36", "qpos_36", "qpos", "reference_qpos_36")
DEFAULT_STATS_PATH = "assets/stats/g1_125d_stats.json"
DEFAULT_KINEMATICS_PATH = "assets/robots/g1/g1_kinematics.json"


def _select_qpos_key(keys: set[str], requested: str | None) -> str:
    if requested is not None:
        if requested not in keys:
            raise KeyError(f"qpos key '{requested}' not found in motion file; available keys: {sorted(keys)}")
        return requested
    for key in DEFAULT_QPOS_KEYS:
        if key in keys:
            return key
    raise KeyError(f"No qpos key found. Expected one of {DEFAULT_QPOS_KEYS}; available keys: {sorted(keys)}")


def _single_float(value: Any, *, name: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one scalar value, got shape {np.asarray(value).shape}")
    return float(array[0])


def load_motion_qpos(path: Path, *, qpos_key: str | None) -> tuple[np.ndarray, str | None, float | None]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        if qpos_key is not None:
            raise ValueError("--qpos-key only applies to .npz motion files")
        return np.asarray(np.load(path), dtype=np.float32), None, None
    if suffix == ".npz":
        with np.load(path) as data:
            key = _select_qpos_key(set(data.files), qpos_key)
            qpos = np.asarray(data[key], dtype=np.float32)
            fps = _single_float(data["fps"], name="fps") if "fps" in data.files else None
        return qpos, key, fps
    raise ValueError(f"Unsupported motion file suffix '{path.suffix}'. Expected .npz or .npy")


def _qpos_shape(qpos: np.ndarray) -> tuple[int, int]:
    if qpos.ndim == 2 and qpos.shape[-1] == 36:
        return 1, int(qpos.shape[0])
    if qpos.ndim == 3 and qpos.shape[-1] == 36:
        return int(qpos.shape[0]), int(qpos.shape[1])
    raise ValueError(f"qpos must have shape (T, 36) or (B, T, 36), got {qpos.shape}")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(name)


def _default_output_path(motion_path: Path) -> Path:
    return motion_path.with_name("physical_metrics.json")


def run(args: argparse.Namespace) -> dict[str, Any]:
    motion_path = Path(args.motion)
    qpos_np, qpos_key, file_fps = load_motion_qpos(motion_path, qpos_key=args.qpos_key)
    fps = float(args.fps) if args.fps is not None else file_fps
    if fps is None:
        raise ValueError("fps is required for physical metrics; pass --fps or provide an 'fps' key in the .npz file")

    batch_size, num_frames = _qpos_shape(qpos_np)
    device = _device(args.device)
    qpos = torch.from_numpy(qpos_np).to(device=device, dtype=torch.float32)
    representation = G1MotionRepresentation(
        stats_path=args.stats_path,
        kinematics_path=args.kinematics_path,
        num_prev_states=10,
        canonical_frame_idx=9,
        feat_dim=125,
        sequence_length=60,
    ).to(device).eval()

    with torch.inference_mode():
        metrics = physical_qpos_metrics(
            qpos_36=qpos,
            representation=representation,
            fps=fps,
            contact_height_threshold=float(args.contact_height_threshold),
            contact_penetration_tolerance=float(args.contact_penetration_tolerance),
        )

    result: dict[str, Any] = {
        "motion": str(motion_path),
        "qpos_key": qpos_key,
        "fps": fps,
        "frames": num_frames,
        "batch_size": batch_size,
        "contact_height_threshold": float(args.contact_height_threshold),
        "contact_penetration_tolerance": float(args.contact_penetration_tolerance),
        "contact_sliding_speed": float(metrics["contact_sliding_speed"].detach().cpu()),
        "foot_ground_error": float(metrics["foot_ground_error"].detach().cpu()),
        "body_jerk_mean": float(metrics["body_jerk_mean"].detach().cpu()),
        "units": {
            "contact_sliding_speed": "m/s",
            "foot_ground_error": "m",
            "body_jerk_mean": "m/s^3",
        },
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute physical benchmark metrics for G1 qpos motion files.")
    parser.add_argument("--motion", required=True, help="Input .npz or .npy motion file containing qpos_36 data.")
    parser.add_argument("--qpos-key", default=None, help="qpos key to read from .npz. Defaults to executed_qpos_36, qpos_36, then reference_qpos_36.")
    parser.add_argument("--fps", type=float, default=None, help="Motion frame rate. Defaults to the .npz fps key when present.")
    parser.add_argument("--output", default=None, help="Output JSON path. Defaults to physical_metrics.json next to --motion.")
    parser.add_argument("--stats-path", default=DEFAULT_STATS_PATH)
    parser.add_argument("--kinematics-path", default=DEFAULT_KINEMATICS_PATH)
    parser.add_argument("--contact-height-threshold", type=float, default=0.12)
    parser.add_argument("--contact-penetration-tolerance", type=float, default=0.02)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run(args)
    output_path = Path(args.output) if args.output is not None else _default_output_path(Path(args.motion))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
