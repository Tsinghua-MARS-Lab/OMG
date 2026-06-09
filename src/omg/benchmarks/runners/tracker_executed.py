"""Tracker-executed benchmark stage for generated G1 motions.

This runner executes generated qpos trajectories with the HoloMotion tracker and
reports tracking errors between the executed robot motion and the generated
reference trajectory. It is an optional stage used by text, audio, and
human-reference generation benchmarks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from omg.benchmarks.metrics import e_acc, e_vel, g_mpjpe, mpjpe
from omg.benchmarks.runners.common import (
    _finite_metrics,
    _save_json,
    _summary_stats,
    _write_jsonl,
    qpos_to_body_positions,
)
from omg.runtime.onnx_providers import DEFAULT_TRACKER_ONNX_PROVIDERS_CSV


TRACKER_EXECUTED_METRIC_KEYS = ("g_mpjpe", "mpjpe", "e_vel", "e_acc")


def add_tracker_executed_args(parser: argparse.ArgumentParser) -> None:
    """Add optional HoloMotion tracker-executed benchmark arguments."""
    parser.add_argument(
        "--tracker-executed",
        "--tracker_executed",
        dest="tracker_executed",
        action="store_true",
        help="Run generated qpos through HoloMotion and report executed-vs-reference tracking metrics.",
    )
    parser.add_argument("--holomotion-onnx", "--holomotion_onnx", dest="holomotion_onnx", default=None)
    parser.add_argument(
        "--tracker-output-name",
        "--tracker_output_name",
        dest="tracker_output_name",
        default="tracker_executed",
        help="Subdirectory name for tracker-executed artifacts inside each benchmark run directory.",
    )
    parser.add_argument(
        "--tracker-num-samples",
        "--tracker_num_samples",
        dest="tracker_num_samples",
        type=int,
        default=None,
        help="Optional subset size for tracker execution. Defaults to all generated benchmark samples.",
    )
    parser.add_argument("--tracker-sample-seed", "--tracker_sample_seed", dest="tracker_sample_seed", type=int, default=0)
    parser.add_argument("--tracker-target-fps", "--tracker_target_fps", dest="tracker_target_fps", type=float, default=50.0)
    parser.add_argument("--tracker-providers", "--tracker_providers", dest="tracker_providers", default=DEFAULT_TRACKER_ONNX_PROVIDERS_CSV)
    parser.add_argument("--tracker-robot-xml", "--tracker_robot_xml", dest="tracker_robot_xml", default=None)
    parser.add_argument("--tracker-steps", "--tracker_steps", dest="tracker_steps", type=int, default=None)
    parser.add_argument("--tracker-control-substeps", "--tracker_control_substeps", dest="tracker_control_substeps", type=int, default=10)
    parser.add_argument("--tracker-action-clip", "--tracker_action_clip", dest="tracker_action_clip", type=float, default=10.0)
    parser.add_argument("--tracker-root-index", "--tracker_root_index", dest="tracker_root_index", type=int, default=0)


def validate_tracker_executed_args(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "tracker_executed", False)):
        return
    if getattr(args, "holomotion_onnx", None) is None:
        raise ValueError("--tracker-executed requires --holomotion_onnx")
    onnx_path = Path(str(args.holomotion_onnx)).expanduser()
    if not onnx_path.exists():
        raise FileNotFoundError(f"--holomotion_onnx does not exist: {onnx_path}")
    if getattr(args, "tracker_num_samples", None) is not None and int(args.tracker_num_samples) <= 0:
        raise ValueError("--tracker_num_samples must be positive")
    if float(args.tracker_target_fps) <= 0.0:
        raise ValueError("--tracker_target_fps must be positive")
    if getattr(args, "tracker_steps", None) is not None and int(args.tracker_steps) <= 2:
        raise ValueError("--tracker_steps must be greater than 2 for E_acc")
    if int(args.tracker_control_substeps) <= 0:
        raise ValueError("--tracker_control_substeps must be positive")
    if float(args.tracker_action_clip) <= 0.0:
        raise ValueError("--tracker_action_clip must be positive")
    if int(args.tracker_root_index) < 0:
        raise ValueError("--tracker_root_index must be non-negative")


def tracker_disabled_result() -> dict[str, Any]:
    return {"enabled": False}


def _as_numpy(value: Any, *, dtype: np.dtype | type = np.float32) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _fps_values(fps: Any, num_samples: int) -> np.ndarray:
    values = _as_numpy(fps, dtype=np.float32)
    if values.ndim == 0:
        values = np.full((num_samples,), float(values), dtype=np.float32)
    values = values.reshape(-1)
    if values.shape[0] != int(num_samples):
        raise ValueError(f"fps must have {num_samples} values, got {values.shape[0]}")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("fps values must be finite and positive")
    return values.astype(np.float32, copy=False)


def _select_tracker_indices(num_samples: int, requested: int | None, seed: int) -> np.ndarray:
    if int(num_samples) <= 0:
        raise ValueError("tracker benchmark requires at least one generated sample")
    if requested is None or int(requested) == int(num_samples):
        return np.arange(int(num_samples), dtype=np.int64)
    if int(requested) <= 0:
        raise ValueError("requested tracker sample count must be positive")
    if int(requested) > int(num_samples):
        raise ValueError(f"requested {requested} tracker samples but only {num_samples} generated samples exist")
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(num_samples), size=int(requested), replace=False)).astype(np.int64, copy=False)


def _pad_sequence_list(sequences: list[np.ndarray], *, dtype: np.dtype | type = np.float32) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("cannot pad an empty sequence list")
    arrays = [np.asarray(sequence, dtype=dtype) for sequence in sequences]
    if any(array.ndim < 1 for array in arrays):
        raise ValueError("sequences must have at least one dimension")
    tail_shape = arrays[0].shape[1:]
    for array in arrays:
        if array.shape[1:] != tail_shape:
            raise ValueError(f"all sequences must share trailing shape {tail_shape}, got {array.shape[1:]}")
    max_len = max(int(array.shape[0]) for array in arrays)
    if max_len <= 0:
        raise ValueError("sequences must contain at least one frame")
    padded = np.zeros((len(arrays), max_len, *tail_shape), dtype=dtype)
    valid = np.zeros((len(arrays), max_len), dtype=np.bool_)
    for idx, array in enumerate(arrays):
        length = int(array.shape[0])
        padded[idx, :length] = array
        valid[idx, :length] = True
    return padded, valid


def _body_positions(model: Any, qpos_36: np.ndarray, *, device: torch.device) -> np.ndarray:
    qpos_tensor = torch.from_numpy(np.asarray(qpos_36, dtype=np.float32))[None]
    positions = qpos_to_body_positions(model, qpos_tensor, batch_size=1, device=device)
    return positions[0].numpy().astype(np.float32, copy=False)


def _tracker_metric_values(
    executed_body_pos: np.ndarray,
    tracker_reference_body_pos: np.ndarray,
    *,
    root_index: int = 0,
) -> dict[str, float]:
    return {
        "g_mpjpe": g_mpjpe(executed_body_pos, tracker_reference_body_pos),
        "mpjpe": mpjpe(executed_body_pos, tracker_reference_body_pos, root_index=int(root_index)),
        "e_vel": e_vel(executed_body_pos, tracker_reference_body_pos),
        "e_acc": e_acc(executed_body_pos, tracker_reference_body_pos),
    }


def _metric_summary(values: dict[str, np.ndarray]) -> dict[str, Any]:
    first = next(iter(values.values()))
    summary = {key: _summary_stats(np.asarray(value, dtype=np.float64)) for key, value in values.items()}
    summary["num_samples"] = int(first.shape[0])
    return summary


def _write_tracker_artifacts(
    *,
    stage_dir: Path,
    selected_indices: np.ndarray,
    executed_qpos: list[np.ndarray],
    tracker_reference_qpos: list[np.ndarray],
    actions: list[np.ndarray],
    generated_qpos: np.ndarray,
    reference_qpos: np.ndarray,
    fps: np.ndarray,
    captions: list[str],
    dataset_names: list[str],
    dataset_indices: list[int],
    holomotion_onnx: str,
    tracker_fps: float,
) -> tuple[Path, np.ndarray]:
    executed_padded, tracker_valid = _pad_sequence_list(executed_qpos, dtype=np.float32)
    tracker_reference_padded, tracker_reference_valid = _pad_sequence_list(tracker_reference_qpos, dtype=np.float32)
    if not np.array_equal(tracker_valid, tracker_reference_valid):
        raise ValueError("executed and tracker reference valid masks differ")
    actions_padded, actions_valid = _pad_sequence_list(actions, dtype=np.float32)
    if not np.array_equal(tracker_valid, actions_valid):
        raise ValueError("executed and action valid masks differ")

    selected = np.asarray(selected_indices, dtype=np.int64)
    artifact_path = stage_dir / "tracker_executed_qpos.npz"
    np.savez_compressed(
        artifact_path,
        executed_qpos_36=executed_padded.astype(np.float32, copy=False),
        tracker_reference_qpos_36=tracker_reference_padded.astype(np.float32, copy=False),
        generated_qpos_36=generated_qpos[selected].astype(np.float32, copy=False),
        reference_qpos_36=reference_qpos[selected].astype(np.float32, copy=False),
        actions=actions_padded.astype(np.float32, copy=False),
        tracker_valid=tracker_valid.astype(np.bool_, copy=False),
        tracker_fps=np.asarray([float(tracker_fps)], dtype=np.float32),
        generation_fps=fps[selected].astype(np.float32, copy=False),
        sample_index=selected.astype(np.int32, copy=False),
        dataset=np.asarray([dataset_names[int(idx)] for idx in selected], dtype=np.str_),
        dataset_index=np.asarray([dataset_indices[int(idx)] for idx in selected], dtype=np.int32),
        caption=np.asarray([captions[int(idx)] for idx in selected], dtype=np.str_),
        holomotion_onnx=np.asarray([str(Path(holomotion_onnx).expanduser().resolve())], dtype=np.str_),
    )
    return artifact_path, tracker_valid


def run_tracker_executed_benchmark(
    *,
    output_dir: Path,
    generated_qpos: Any,
    reference_qpos: Any,
    fps: Any,
    captions: list[str],
    dataset_names: list[str],
    dataset_indices: list[int],
    model: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Execute generated qpos with HoloMotion and write tracker artifacts."""
    if not bool(getattr(args, "tracker_executed", False)):
        return tracker_disabled_result()

    validate_tracker_executed_args(args)
    generated_np = _as_numpy(generated_qpos, dtype=np.float32)
    reference_np = _as_numpy(reference_qpos, dtype=np.float32)
    if generated_np.ndim != 3 or generated_np.shape[-1] != 36:
        raise ValueError(f"generated_qpos must have shape (N, T, 36), got {generated_np.shape}")
    if reference_np.ndim != 3 or reference_np.shape[-1] != 36:
        raise ValueError(f"reference_qpos must have shape (N, T, 36), got {reference_np.shape}")
    if reference_np.shape[0] != generated_np.shape[0]:
        raise ValueError(f"generated/reference sample count mismatch: {generated_np.shape[0]} != {reference_np.shape[0]}")
    num_samples = int(generated_np.shape[0])
    if not (len(captions) == len(dataset_names) == len(dataset_indices) == num_samples):
        raise ValueError("captions, dataset_names, and dataset_indices must match generated sample count")
    fps_np = _fps_values(fps, num_samples)
    selected_indices = _select_tracker_indices(num_samples, getattr(args, "tracker_num_samples", None), int(args.tracker_sample_seed))

    stage_dir = output_dir / str(args.tracker_output_name)
    stage_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] Tracker-executed benchmark: samples={len(selected_indices)}/{num_samples} "
        f"target_fps={float(args.tracker_target_fps)} output_dir={stage_dir.resolve()}"
    )

    from omg.tracking.holomotion.runner import HoloMotionRolloutRunner

    runner = HoloMotionRolloutRunner(
        holomotion_onnx=args.holomotion_onnx,
        target_fps=float(args.tracker_target_fps),
        robot_xml=args.tracker_robot_xml,
        providers=args.tracker_providers,
        control_substeps=int(args.tracker_control_substeps),
        action_clip=float(args.tracker_action_clip),
        video=False,
    )
    executed_qpos: list[np.ndarray] = []
    tracker_reference_qpos: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    metric_values: dict[str, list[float]] = {key: [] for key in TRACKER_EXECUTED_METRIC_KEYS}
    sample_rows: list[dict[str, Any]] = []
    try:
        for output_index, sample_index in enumerate(tqdm(selected_indices, desc="Tracker execute", unit="sample")):
            idx = int(sample_index)
            runner.clear_rollout_buffers(reset_initialized=True)
            chunk = runner.run_reference_chunk(
                generated_np[idx],
                reference_fps=float(fps_np[idx]),
                steps=args.tracker_steps,
                plan_id=0,
                plan_start=0,
            )
            executed = np.asarray(chunk.qpos_36, dtype=np.float32)
            tracker_reference = np.asarray(runner.reference_qpos_36, dtype=np.float32)
            action = np.asarray(runner.actions, dtype=np.float32)
            if executed.shape != tracker_reference.shape:
                raise ValueError(f"tracker output/reference shape mismatch: {executed.shape} != {tracker_reference.shape}")
            if action.shape[0] != executed.shape[0]:
                raise ValueError(f"tracker action frame count mismatch: {action.shape[0]} != {executed.shape[0]}")
            executed_body = _body_positions(model, executed, device=device)
            reference_body = _body_positions(model, tracker_reference, device=device)
            metrics = _tracker_metric_values(
                executed_body,
                reference_body,
                root_index=int(args.tracker_root_index),
            )
            executed_qpos.append(executed)
            tracker_reference_qpos.append(tracker_reference)
            actions.append(action)
            for key, value in metrics.items():
                metric_values[key].append(float(value))
            sample_rows.append(
                {
                    "sample_index": idx,
                    "tracker_output_index": int(output_index),
                    "dataset": dataset_names[idx],
                    "dataset_index": int(dataset_indices[idx]),
                    "caption": captions[idx],
                    "frames": int(executed.shape[0]),
                    "generation_fps": float(fps_np[idx]),
                    "tracker_fps": float(args.tracker_target_fps),
                    **{key: float(metrics[key]) for key in TRACKER_EXECUTED_METRIC_KEYS},
                }
            )
    finally:
        runner.close()

    metric_arrays = {key: np.asarray(values, dtype=np.float64) for key, values in metric_values.items()}
    metrics_summary = _metric_summary(metric_arrays)
    metrics_summary.update(
        {
            "enabled": True,
            "root_index": int(args.tracker_root_index),
            "target_fps": float(args.tracker_target_fps),
        }
    )
    artifact_path, tracker_valid = _write_tracker_artifacts(
        stage_dir=stage_dir,
        selected_indices=selected_indices,
        executed_qpos=executed_qpos,
        tracker_reference_qpos=tracker_reference_qpos,
        actions=actions,
        generated_qpos=generated_np,
        reference_qpos=reference_np,
        fps=fps_np,
        captions=captions,
        dataset_names=dataset_names,
        dataset_indices=dataset_indices,
        holomotion_onnx=str(args.holomotion_onnx),
        tracker_fps=float(args.tracker_target_fps),
    )
    sample_metrics_path = stage_dir / "tracker_sample_metrics.jsonl"
    metrics_path = stage_dir / "tracker_metrics.json"
    benchmark_path = stage_dir / "benchmark_tracker.json"
    _write_jsonl(sample_metrics_path, sample_rows)
    _save_json(metrics_path, metrics_summary)

    benchmark = {
        "enabled": True,
        "num_samples": int(len(selected_indices)),
        "num_generated_samples": num_samples,
        "selected_indices": selected_indices.astype(int).tolist(),
        "holomotion_onnx": str(Path(str(args.holomotion_onnx)).expanduser().resolve()),
        "robot_xml": None if args.tracker_robot_xml is None else str(Path(str(args.tracker_robot_xml)).expanduser().resolve()),
        "providers": args.tracker_providers,
        "target_fps": float(args.tracker_target_fps),
        "steps": None if args.tracker_steps is None else int(args.tracker_steps),
        "control_substeps": int(args.tracker_control_substeps),
        "action_clip": float(args.tracker_action_clip),
        "root_index": int(args.tracker_root_index),
        "artifact_path": str(artifact_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "sample_metrics_path": str(sample_metrics_path.resolve()),
        "valid_shape": list(tracker_valid.shape),
        "metrics": metrics_summary,
    }
    _finite_metrics(benchmark)
    _save_json(benchmark_path, benchmark)
    benchmark["benchmark_path"] = str(benchmark_path.resolve())
    return benchmark
