from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from tqdm import tqdm

from omg.benchmarks.runners.common import (
    finite_valid,
    load_condition_model,
    output_dir as conditioned_output_dir,
    qpos_to_body_positions,
    select_condition_records,
    summarize_values,
)
from omg.benchmarks.runners.common import (
    BenchmarkResult,
    SampleRecord,
    _build_datasets,
    _cfg_output_name,
    _cfg_scale_json,
    _config_dir,
    _dataset_indices,
    _device,
    _finite_metrics,
    _load_sample_records,
    _parse_cfg_scale_value,
    _physical_summary_from_values,
    _physical_values,
    _sample_file_path,
    _save_json,
    _validate_sample_records,
    _write_jsonl,
    _write_sample_records,
)
from omg.benchmarks.metrics import (
    aistpp_edge_features,
    aistpp_edge_metric_summary,
    beat_align,
    transition_metric_summary,
    transition_metric_values,
)
from omg.benchmarks.runners.tracker_executed import (
    add_tracker_executed_args,
    run_tracker_executed_benchmark,
    validate_tracker_executed_args,
)
from omg.benchmarks.report import generation_benchmark_markdown
from omg.data.datamodule import motion_collate_fn


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an audio-conditioned generation checkpoint.")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--ckpts", nargs="+", default=None)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--data", default="omg_data")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_scale", "--cfg_audio_scale", dest="cfg_audio_scale", type=float, default=None)
    parser.add_argument("--cfg_scales", "--cfg_audio_scales", dest="cfg_audio_scales", nargs="+", default=None)
    parser.add_argument("--samples_path", default=None)
    parser.add_argument("--contact_height_threshold", type=float, default=0.12)
    parser.add_argument("--contact_penetration_tolerance", type=float, default=0.02)
    parser.add_argument("--transition-metrics", "--transition_metrics", dest="transition_metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transition-chunk-length", "--transition_chunk_length", dest="transition_chunk_length", type=int, default=None)
    parser.add_argument("--beat_threshold", type=float, default=0.5)
    parser.add_argument("--sigma_frames", type=float, default=3.0)
    parser.add_argument("--min_motion_beat_distance_seconds", type=float, default=0.25)
    parser.add_argument("--beat_direction", choices=["music_to_motion", "motion_to_music"], default="music_to_motion")
    parser.add_argument("--aistpp_up_axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--pfc_root_index", type=int, default=0)
    parser.add_argument("--pfc_left_foot_indices", type=int, nargs="+", default=[5, 6])
    parser.add_argument("--pfc_right_foot_indices", type=int, nargs="+", default=[11, 12])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    add_tracker_executed_args(parser)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides, e.g. model.use_audio=true model.audio_dim=35",
    )
    args = parser.parse_args(argv)
    if args.ckpts is None:
        if args.ckpt_path is None:
            raise ValueError("Specify --ckpts or --ckpt_path")
        args.ckpts = [args.ckpt_path]
    if args.ckpt_path is None:
        args.ckpt_path = args.ckpts[0]
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.num_frames <= 1:
        raise ValueError("--num_frames must be greater than 1 for BeatAlign")
    if args.cfg_audio_scale is not None and not math.isfinite(float(args.cfg_audio_scale)):
        raise ValueError("--cfg_audio_scale must be finite")
    if args.transition_chunk_length is not None and args.transition_chunk_length <= 1:
        raise ValueError("--transition_chunk_length must be greater than 1")
    if args.cfg_audio_scales is None:
        args.resolved_cfg_scales = [None if args.cfg_audio_scale is None else float(args.cfg_audio_scale)]
    else:
        args.resolved_cfg_scales = [_parse_cfg_scale_value(value) for value in args.cfg_audio_scales]
    if len(args.pfc_left_foot_indices) == 0 or len(args.pfc_right_foot_indices) == 0:
        raise ValueError("PFC foot index lists must be non-empty")
    validate_tracker_executed_args(args)
    return args


def _generate_audio_qpos(
    *,
    model: Any,
    datasets: dict[str, Any],
    records: list[SampleRecord],
    args: argparse.Namespace,
    cfg_audio_scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str], list[int]]:
    generated_chunks = []
    reference_chunks = []
    reference_body_chunks = []
    valid_chunks = []
    fps_chunks = []
    audio_chunks = []
    captions: list[str] = []
    dataset_names: list[str] = []
    dataset_indices: list[int] = []

    for start in tqdm(
        range(0, len(records), args.batch_size),
        desc="Benchmark generate",
        unit="batch",
    ):
        batch_records = records[start : start + args.batch_size]
        items = [datasets[record.dataset][record.index] for record in batch_records]
        batch = motion_collate_fn(items)
        batch_size = len(items)
        batch["has_text"] = torch.zeros(batch_size, dtype=torch.bool)
        with torch.inference_mode():
            sample = model.generate(
                batch,
                num_frames=int(args.num_frames),
                cfg_text_scale=0.0,
                cfg_audio_scale=cfg_audio_scale,
                cfg_human_scale=0.0,
            )
        generated_chunks.append(sample["qpos_36"].detach().cpu())
        reference_chunks.append(batch["qpos_36"][:, : args.num_frames].detach().cpu())
        reference_body_chunks.append(batch["body_pos_w"][:, : args.num_frames].detach().cpu())
        valid_chunks.append(finite_valid(batch["mask"]["valid"], args.num_frames))
        fps_chunks.append(batch["fps"].detach().cpu().float())
        audio_chunks.append(batch["audio_features"][:, : args.num_frames].detach().cpu())
        captions.extend(str(item.get("caption", "")) for item in items)
        dataset_names.extend(record.dataset for record in batch_records)
        dataset_indices.extend(int(record.index) for record in batch_records)

    return (
        torch.cat(generated_chunks, dim=0),
        torch.cat(reference_chunks, dim=0),
        torch.cat(reference_body_chunks, dim=0),
        torch.cat(valid_chunks, dim=0),
        torch.cat(fps_chunks, dim=0),
        torch.cat(audio_chunks, dim=0),
        captions,
        dataset_names,
        dataset_indices,
    )


def _beat_align_values(
    *,
    audio_features: torch.Tensor,
    generated_body_pos: torch.Tensor,
    reference_body_pos: torch.Tensor,
    fps: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    generated_scores = []
    reference_scores = []
    audio_np = audio_features.numpy()
    generated_np = generated_body_pos.numpy()
    reference_np = reference_body_pos.numpy()
    fps_np = fps.numpy()
    for idx in tqdm(range(audio_np.shape[0]), desc="BeatAlign", unit="sample", leave=False):
        kwargs = {
            "audio_features": audio_np[idx],
            "fps": float(fps_np[idx]),
            "audio_fps": float(fps_np[idx]),
            "sigma_frames": float(args.sigma_frames),
            "direction": args.beat_direction,
            "beat_threshold": float(args.beat_threshold),
            "min_motion_beat_distance_seconds": float(args.min_motion_beat_distance_seconds),
        }
        generated_scores.append(float(beat_align(motion_positions=generated_np[idx], **kwargs)))
        reference_scores.append(float(beat_align(motion_positions=reference_np[idx], **kwargs)))
    generated = np.asarray(generated_scores, dtype=np.float64)
    reference = np.asarray(reference_scores, dtype=np.float64)
    return {
        "beat_align_generated": generated,
        "beat_align_reference": reference,
        "beat_align_gap": generated - reference,
    }


def _kinematic_parent_edges(model: Any) -> list[tuple[int, int]]:
    kinematics = model.representation.kinematics
    parent = kinematics.parent_body_indices.detach().cpu().numpy()
    child = kinematics.child_body_indices.detach().cpu().numpy()
    return [(int(parent_idx), int(child_idx)) for parent_idx, child_idx in zip(parent, child)]


def _dance_sample_values(dance_features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    generated = np.asarray(dance_features["generated_pfc"], dtype=np.float64)
    reference = np.asarray(dance_features["reference_pfc"], dtype=np.float64)
    return {
        "pfc_generated": generated,
        "pfc_reference": reference,
        "pfc_gap": generated - reference,
    }


def _transition_chunk_length(model: Any, args: argparse.Namespace) -> int:
    return int(args.transition_chunk_length or getattr(model.representation, "sequence_length"))


def _compute_transition_metrics(
    *,
    model: Any,
    generated_qpos: torch.Tensor,
    reference_qpos: torch.Tensor,
    generated_body_pos: torch.Tensor,
    reference_body_pos: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    chunk_length = _transition_chunk_length(model, args)
    if not args.transition_metrics:
        return {}, {"enabled": False, "reason": "disabled", "chunk_length": chunk_length}
    if generated_qpos.shape[1] <= chunk_length:
        return (
            {},
            {
                "enabled": False,
                "reason": "num_frames must be greater than chunk_length",
                "num_frames": int(generated_qpos.shape[1]),
                "chunk_length": chunk_length,
            },
        )
    values = transition_metric_values(
        qpos=generated_qpos.numpy(),
        reference_qpos=reference_qpos.numpy(),
        body_pos=generated_body_pos.numpy(),
        reference_body_pos=reference_body_pos.numpy(),
        chunk_length=chunk_length,
    )
    summary = transition_metric_summary(values, chunk_length=chunk_length, num_frames=int(generated_qpos.shape[1]))
    summary["enabled"] = True
    return values, summary


def _audio_dataset_metrics(
    *,
    dataset_names: list[str],
    physical_values: dict[str, np.ndarray],
    audio_values: dict[str, np.ndarray],
    dance_features: dict[str, np.ndarray],
    transition_values: dict[str, np.ndarray] | None = None,
    transition_chunk_length: int | None = None,
    transition_num_frames: int | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, indices in _dataset_indices(dataset_names).items():
        item = {
            "num_samples": int(len(indices)),
            "audio": summarize_values({key: value[indices] for key, value in audio_values.items()}),
            "aistpp_edge": aistpp_edge_metric_summary(dance_features, indices=indices),
            "physical": _physical_summary_from_values(physical_values, indices),
        }
        if transition_values:
            if transition_chunk_length is None or transition_num_frames is None:
                raise ValueError("transition dataset metrics require chunk length and num frames")
            item["transition"] = transition_metric_summary(
                transition_values,
                chunk_length=int(transition_chunk_length),
                num_frames=int(transition_num_frames),
                indices=indices,
            )
        metrics[name] = item
    return metrics


def _sample_rows(
    *,
    captions: list[str],
    dataset_names: list[str],
    dataset_indices: list[int],
    audio_values: dict[str, np.ndarray],
    physical_values: dict[str, np.ndarray],
    dance_values: dict[str, np.ndarray],
    transition_values: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for idx, caption in enumerate(captions):
        row: dict[str, Any] = {
            "sample_index": idx,
            "dataset": dataset_names[idx],
            "dataset_index": int(dataset_indices[idx]),
            "caption": caption,
        }
        for key, values in audio_values.items():
            row[key] = float(values[idx])
        for key, values in physical_values.items():
            row[key] = float(values[idx])
        for key, values in dance_values.items():
            row[key] = float(values[idx])
        if transition_values:
            for key, values in transition_values.items():
                array = np.asarray(values)
                row[key] = float(array if array.ndim == 0 else array[idx])
        rows.append(row)
    return rows


def _summary_row(benchmark: dict[str, Any], label: str | None = None) -> dict[str, Any]:
    audio = benchmark["audio_metrics"]
    physical = benchmark["physical_metrics"]
    aistpp_edge = benchmark["aistpp_edge_metrics"]
    transition = benchmark.get("transition_metrics", {})
    tracker = benchmark.get("tracker_executed", {})
    tracker_metrics = tracker.get("metrics", {}) if tracker.get("enabled") else {}
    return {
        "ckpt": label or Path(benchmark["ckpt_path"]).name,
        "BeatAlign": audio["beat_align_generated"],
        "FIDk": aistpp_edge.get("fid_k"),
        "FIDg": aistpp_edge.get("fid_g"),
        "Divk": aistpp_edge.get("div_k_generated"),
        "Divg": aistpp_edge.get("div_g_generated"),
        "PFC": aistpp_edge.get("pfc_generated"),
        "PFC Ref": aistpp_edge.get("pfc_reference"),
        "contact_sliding": physical.get("contact_sliding_speed"),
        "foot_ground_error": physical.get("foot_ground_error"),
        "body_jerk": physical.get("body_jerk_mean"),
        "PJ": transition.get("pj") if transition.get("enabled") else None,
        "AUJ": transition.get("auj") if transition.get("enabled") else None,
        "Tracker g-MPJPE": tracker_metrics.get("g_mpjpe"),
        "Tracker MPJPE": tracker_metrics.get("mpjpe"),
        "Tracker E_vel": tracker_metrics.get("e_vel"),
        "Tracker E_acc": tracker_metrics.get("e_acc"),
    }


def _write_audio_summary(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace, datasets: dict[str, Any]) -> None:
    markdown = generation_benchmark_markdown(
        rows=rows,
        metric_directions={
            "BeatAlign": "max",
            "FIDk": "min",
            "FIDg": "min",
            "Divk": "max",
            "Divg": "max",
            "PFC": "min",
            "PFC Ref": "min",
            "contact_sliding": "min",
            "foot_ground_error": "min",
            "body_jerk": "min",
            "PJ": "min",
            "AUJ": "min",
            "Tracker g-MPJPE": "min",
            "Tracker MPJPE": "min",
            "Tracker E_vel": "min",
            "Tracker E_acc": "min",
        },
        metadata={
            "benchmark": "audio",
            "ckpts": [str(Path(path).resolve()) for path in args.ckpts],
            "datasets": list(datasets.keys()),
            "num_samples": args.num_samples,
            "num_frames": args.num_frames,
            "batch_size": args.batch_size,
            "split": args.split,
            "data": args.data,
            "beat_direction": args.beat_direction,
            "sigma_frames": args.sigma_frames,
            "aistpp_up_axis": args.aistpp_up_axis,
            "pfc_root_index": args.pfc_root_index,
            "pfc_left_foot_indices": list(args.pfc_left_foot_indices),
            "pfc_right_foot_indices": list(args.pfc_right_foot_indices),
            "transition_metrics": "TextOp-style PJ/AUJ on body/link jerk over transition windows centered at autoregressive chunk seams.",
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")


def _run_single_benchmark(
    *,
    output_dir: Path,
    ckpt_path: str,
    cfg_audio_scale: float | None,
    datasets: dict[str, Any],
    records: list[SampleRecord],
    model: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> BenchmarkResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_sample_records(output_dir / "samples.jsonl", records, datasets)
    print(
        f"[INFO] Audio benchmark run: num_samples={len(records)} device={device} "
        f"output_dir={output_dir.resolve()}"
    )

    (
        generated_qpos,
        reference_qpos,
        reference_body_pos,
        reference_valid,
        fps,
        audio_features,
        captions,
        dataset_names,
        dataset_indices,
    ) = _generate_audio_qpos(model=model, datasets=datasets, records=records, args=args, cfg_audio_scale=cfg_audio_scale)

    print("[INFO] Forward kinematics on generated motion, physical metrics, BeatAlign…")
    generated_body_pos = qpos_to_body_positions(model, generated_qpos, batch_size=args.batch_size, device=device)
    transition_values, transition_metrics = _compute_transition_metrics(
        model=model,
        generated_qpos=generated_qpos,
        reference_qpos=reference_qpos,
        generated_body_pos=generated_body_pos,
        reference_body_pos=reference_body_pos,
        args=args,
    )
    physical_values = _physical_values(
        generated_qpos,
        fps,
        representation=model.representation,
        device=device,
        contact_height_threshold=args.contact_height_threshold,
        contact_penetration_tolerance=args.contact_penetration_tolerance,
    )
    audio_values = _beat_align_values(
        audio_features=audio_features,
        generated_body_pos=generated_body_pos,
        reference_body_pos=reference_body_pos,
        fps=fps,
        args=args,
    )
    print("[INFO] AIST++ edge / dance features…")
    dance_features = aistpp_edge_features(
        reference_body_pos.numpy(),
        generated_body_pos.numpy(),
        fps=fps.numpy(),
        root_index=int(args.pfc_root_index),
        up_axis=args.aistpp_up_axis,
        parent_edges=_kinematic_parent_edges(model),
        left_foot_indices=tuple(args.pfc_left_foot_indices),
        right_foot_indices=tuple(args.pfc_right_foot_indices),
    )
    dance_values = _dance_sample_values(dance_features)

    audio_metrics = summarize_values(audio_values)
    audio_metrics.update(
        {
            "frames": int(generated_qpos.shape[1]),
            "beat_direction": args.beat_direction,
            "beat_threshold": float(args.beat_threshold),
            "sigma_frames": float(args.sigma_frames),
            "min_motion_beat_distance_seconds": float(args.min_motion_beat_distance_seconds),
        }
    )
    physical_metrics = _physical_summary_from_values(physical_values)
    physical_metrics["frames"] = int(generated_qpos.shape[1])
    physical_metrics["contact_height_threshold"] = float(args.contact_height_threshold)
    physical_metrics["contact_penetration_tolerance"] = float(args.contact_penetration_tolerance)
    dance_metrics = aistpp_edge_metric_summary(dance_features)
    dance_metrics.update(
        {
            "frames": int(generated_qpos.shape[1]),
            "up_axis": args.aistpp_up_axis,
            "root_index": int(args.pfc_root_index),
            "left_foot_indices": list(args.pfc_left_foot_indices),
            "right_foot_indices": list(args.pfc_right_foot_indices),
            "geometric_feature": "g1_root_relative_body_geometry",
            "kinetic_feature": "fairmotion_root_relative_joint_kinetic_energy",
        }
    )
    dataset_metrics = _audio_dataset_metrics(
        dataset_names=dataset_names,
        physical_values=physical_values,
        audio_values=audio_values,
        dance_features=dance_features,
        transition_values=transition_values,
        transition_chunk_length=transition_metrics.get("chunk_length"),
        transition_num_frames=int(generated_qpos.shape[1]),
    )
    sample_rows = _sample_rows(
        captions=captions,
        dataset_names=dataset_names,
        dataset_indices=dataset_indices,
        audio_values=audio_values,
        physical_values=physical_values,
        dance_values=dance_values,
        transition_values=transition_values,
    )
    tracker_executed = run_tracker_executed_benchmark(
        output_dir=output_dir,
        generated_qpos=generated_qpos,
        reference_qpos=reference_qpos,
        fps=fps,
        captions=captions,
        dataset_names=dataset_names,
        dataset_indices=dataset_indices,
        model=model,
        args=args,
        device=device,
    )

    print("[INFO] Saving metrics and motion artifacts…")
    _finite_metrics(audio_metrics)
    _finite_metrics(physical_metrics)
    _finite_metrics(dance_metrics)
    _finite_metrics(transition_metrics)
    _finite_metrics(dataset_metrics)
    _finite_metrics(tracker_executed)
    _save_json(output_dir / "audio_metrics.json", audio_metrics)
    _save_json(output_dir / "aistpp_edge_metrics.json", dance_metrics)
    _save_json(output_dir / "physical_metrics.json", physical_metrics)
    _save_json(output_dir / "transition_metrics.json", transition_metrics)
    _save_json(output_dir / "dataset_metrics.json", dataset_metrics)
    _write_jsonl(output_dir / "sample_metrics.jsonl", sample_rows)
    np.savez_compressed(
        output_dir / "generated_qpos.npz",
        qpos_36=generated_qpos.numpy().astype(np.float32, copy=False),
        body_pos_w=generated_body_pos.numpy().astype(np.float32, copy=False),
        fps=fps.numpy().astype(np.float32, copy=False),
        captions=np.asarray(captions, dtype=np.str_),
        dataset=np.asarray(dataset_names, dtype=np.str_),
        dataset_index=np.asarray(dataset_indices, dtype=np.int32),
        ckpt_path=np.asarray([str(Path(ckpt_path).resolve())], dtype=np.str_),
    )
    np.savez_compressed(
        output_dir / "reference_qpos.npz",
        qpos_36=reference_qpos.numpy().astype(np.float32, copy=False),
        body_pos_w=reference_body_pos.numpy().astype(np.float32, copy=False),
        valid=reference_valid.numpy().astype(np.bool_, copy=False),
        fps=fps.numpy().astype(np.float32, copy=False),
        captions=np.asarray(captions, dtype=np.str_),
        dataset=np.asarray(dataset_names, dtype=np.str_),
        dataset_index=np.asarray(dataset_indices, dtype=np.int32),
    )
    np.savez_compressed(
        output_dir / "audio_features.npz",
        audio_features=audio_features.numpy().astype(np.float32, copy=False),
        fps=fps.numpy().astype(np.float32, copy=False),
        dataset=np.asarray(dataset_names, dtype=np.str_),
        dataset_index=np.asarray(dataset_indices, dtype=np.int32),
    )
    np.savez_compressed(
        output_dir / "aistpp_edge_features.npz",
        reference_kinetic=dance_features["reference_kinetic"].astype(np.float32, copy=False),
        generated_kinetic=dance_features["generated_kinetic"].astype(np.float32, copy=False),
        reference_geometric=dance_features["reference_geometric"].astype(np.float32, copy=False),
        generated_geometric=dance_features["generated_geometric"].astype(np.float32, copy=False),
        reference_pfc=dance_features["reference_pfc"].astype(np.float32, copy=False),
        generated_pfc=dance_features["generated_pfc"].astype(np.float32, copy=False),
    )

    benchmark = {
        "benchmark": "audio",
        "ckpt_path": str(Path(ckpt_path).resolve()),
        "exp": args.exp,
        "split": args.split,
        "num_samples": len(records),
        "num_frames": int(args.num_frames),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "cfg_audio_scale": _cfg_scale_json(cfg_audio_scale),
        "samples_path": str((output_dir / "samples.jsonl").resolve()),
        "generated_qpos": str((output_dir / "generated_qpos.npz").resolve()),
        "reference_qpos": str((output_dir / "reference_qpos.npz").resolve()),
        "audio_features": str((output_dir / "audio_features.npz").resolve()),
        "audio_metrics": audio_metrics,
        "aistpp_edge_metrics": dance_metrics,
        "physical_metrics": physical_metrics,
        "transition_metrics": transition_metrics,
        "tracker_executed": tracker_executed,
        "dataset_metrics": dataset_metrics,
        "sample_metrics_path": str((output_dir / "sample_metrics.jsonl").resolve()),
    }
    _save_json(output_dir / "benchmark.json", benchmark)
    return BenchmarkResult(benchmark=benchmark)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = _device(args.device)
    root_output_dir = conditioned_output_dir(args, "audio")
    root_output_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=str(_config_dir()), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"exp={args.exp}",
                f"data={args.data}",
                "logger=none",
                "trainer=1gpu",
                "model.text_encoder=null",
                *args.overrides,
            ],
        )
    datasets = _build_datasets(cfg, args.split, include=args.datasets)
    sample_path = _sample_file_path(root_output_dir, args.samples_path)
    if args.samples_path is not None and not sample_path.exists():
        raise FileNotFoundError(f"--samples_path does not exist: {sample_path}")
    if sample_path.exists():
        records = _load_sample_records(sample_path)
        print(f"[INFO] Loaded {len(records)} sample records from {sample_path.resolve()}")
    else:
        records = select_condition_records(
            datasets,
            num_samples=args.num_samples,
            seed=args.seed,
            num_frames=args.num_frames,
            tensor_key="audio_features",
            mask_key="has_audio",
            label="audio",
        )
    _validate_sample_records(records, datasets)
    _write_sample_records(root_output_dir / "samples.jsonl", records, datasets)
    print(
        f"[INFO] Benchmark setup: num_records={len(records)} device={device} "
        f"split={args.split} datasets={list(datasets.keys())} output_dir={root_output_dir.resolve()}"
    )

    rows = []
    runs = {}
    for ckpt_path in args.ckpts:
        model = load_condition_model(cfg, ckpt_path, device)
        if not bool(getattr(model, "use_audio", False)):
            raise ValueError("Audio benchmark requires a checkpoint/config with model.use_audio=true")
        ckpt_label = Path(ckpt_path).stem
        for cfg_audio_scale in args.resolved_cfg_scales:
            label = ckpt_label if len(args.resolved_cfg_scales) == 1 else f"{ckpt_label}_{_cfg_output_name(cfg_audio_scale)}"
            run_dir = root_output_dir if len(args.ckpts) == 1 and len(args.resolved_cfg_scales) == 1 else root_output_dir / label
            print(f"[INFO] Running audio benchmark ckpt={ckpt_path} cfg_audio_scale={cfg_audio_scale} output_dir={run_dir}")
            result = _run_single_benchmark(
                output_dir=run_dir,
                ckpt_path=ckpt_path,
                cfg_audio_scale=cfg_audio_scale,
                datasets=datasets,
                records=records,
                model=model,
                args=args,
                device=device,
            )
            rows.append(_summary_row(result.benchmark, label=label))
            runs[label] = {
                "ckpt_path": str(Path(ckpt_path).resolve()),
                "cfg_audio_scale": _cfg_scale_json(cfg_audio_scale),
                "output_dir": str(run_dir.resolve()),
                "benchmark_json": str((run_dir / "benchmark.json").resolve()),
                "audio_metrics": result.benchmark["audio_metrics"],
                "aistpp_edge_metrics": result.benchmark["aistpp_edge_metrics"],
                "physical_metrics": result.benchmark["physical_metrics"],
                "transition_metrics": result.benchmark["transition_metrics"],
                "tracker_executed": result.benchmark["tracker_executed"],
            }
    summary_payload = {
        "benchmark": "audio",
        "ckpts": [str(Path(path).resolve()) for path in args.ckpts],
        "exp": args.exp,
        "split": args.split,
        "num_samples": len(records),
        "num_frames": int(args.num_frames),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "datasets": list(datasets.keys()),
        "samples_path": str((root_output_dir / "samples.jsonl").resolve()),
        "runs": runs,
    }
    _finite_metrics(summary_payload)
    _save_json(root_output_dir / "metrics.json", summary_payload)
    _write_audio_summary(root_output_dir, rows, args, datasets)
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
