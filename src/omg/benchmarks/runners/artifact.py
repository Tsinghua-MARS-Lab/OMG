from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from omg.benchmarks.metrics import aistpp_edge_features, aistpp_edge_metric_summary
from omg.benchmarks.report import generation_benchmark_markdown
from omg.benchmarks.runners.audio import (
    _audio_dataset_metrics,
    _beat_align_values,
    _compute_transition_metrics as _compute_audio_transition_metrics,
    _dance_sample_values,
    _kinematic_parent_edges,
    _sample_rows as _audio_sample_rows,
    _summary_row as _audio_summary_row,
)
from omg.benchmarks.runners.common import (
    BenchmarkResult,
    SampleRecord,
    _build_datasets,
    _config_dir,
    _dataset_names_from_records,
    _device,
    _embedding_distribution_metrics,
    _encode_motion_embeddings,
    _finite_metrics,
    _load_evaluator_motion_encoder,
    _load_sample_records,
    _motion_for_evaluator,
    _physical_summary_from_values,
    _physical_values,
    _resolve_sample_records,
    _save_json,
    _validate_sample_records,
    _write_jsonl,
    _write_sample_records,
    finite_valid,
    qpos_to_body_positions,
    summarize_values,
)
from omg.benchmarks.runners.humanref import (
    _compute_transition_metrics as _compute_humanref_transition_metrics,
    _end_effector_indices as _humanref_end_effector_indices,
    _sample_rows as _humanref_sample_rows,
    _summary_row as _humanref_summary_row,
    _tracking_values as _humanref_tracking_values,
)
from omg.benchmarks.runners.text import (
    TEXT_RETRIEVAL_BATCH_SIZE,
    _compute_transition_metrics as _compute_text_transition_metrics,
    _dataset_metrics as _text_dataset_metrics,
    _encode_text_embeddings,
    _real_motion_metrics,
    _sample_metric_rows as _text_sample_metric_rows,
    _sample_rankings,
    _summary_row as _text_summary_row,
    _text_retrieval_summary,
)
from omg.benchmarks.runners.tracker_executed import (
    add_tracker_executed_args,
    run_tracker_executed_benchmark,
    validate_tracker_executed_args,
)
from omg.data.datamodule import motion_collate_fn


GENERATED_KEYS = ("qpos_36", "fps", "captions", "dataset", "dataset_index")


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"artifact does not exist: {resolved}")
    with np.load(resolved, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _require_keys(arrays: dict[str, np.ndarray], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in arrays]
    if missing:
        raise ValueError(f"{label} missing required keys: {missing}")


def _validate_generated(arrays: dict[str, np.ndarray]) -> None:
    _require_keys(arrays, GENERATED_KEYS, "generated_qpos")
    qpos = arrays["qpos_36"]
    if qpos.ndim != 3 or qpos.shape[-1] != 36:
        raise ValueError(f"generated qpos_36 must have shape (N,T,36), got {qpos.shape}")
    if qpos.dtype != np.float32:
        raise ValueError(f"generated qpos_36 must be float32, got {qpos.dtype}")
    num_samples = int(qpos.shape[0])
    fps = np.asarray(arrays["fps"])
    if fps.shape != (num_samples,):
        raise ValueError(f"generated fps must have shape ({num_samples},), got {fps.shape}")
    fps64 = fps.astype(np.float64)
    if not np.isfinite(fps64).all() or np.any(fps64 <= 0.0):
        raise ValueError("generated fps must be finite and positive")
    for key in ("captions", "dataset", "dataset_index"):
        if np.asarray(arrays[key]).shape != (num_samples,):
            raise ValueError(f"generated {key} must have shape ({num_samples},), got {arrays[key].shape}")


def _assert_records_match_artifact(records: list[SampleRecord], arrays: dict[str, np.ndarray]) -> None:
    if len(records) != int(arrays["qpos_36"].shape[0]):
        raise ValueError(f"sample count mismatch: {len(records)} records vs {arrays['qpos_36'].shape[0]} generated")
    artifact_datasets = [str(value) for value in arrays["dataset"].tolist()]
    artifact_indices = [int(value) for value in arrays["dataset_index"].tolist()]
    for idx, record in enumerate(records):
        if record.dataset != artifact_datasets[idx] or int(record.index) != artifact_indices[idx]:
            raise ValueError(
                "samples_path order does not match generated_qpos metadata at "
                f"sample {idx}: samples=({record.dataset},{record.index}) "
                f"artifact=({artifact_datasets[idx]},{artifact_indices[idx]})"
            )


def _compose_cfg(args: argparse.Namespace) -> Any:
    overrides = [f"exp={args.exp}", "logger=none", "trainer=1gpu", *args.overrides]
    overrides.insert(1, f"data={args.data}")
    if args.mode != "text":
        overrides.insert(4, "model.text_encoder=null")
    with initialize_config_dir(config_dir=str(_config_dir()), version_base="1.3"):
        return compose(config_name="train", overrides=overrides)


def _metric_model(cfg: Any, device: torch.device) -> Any:
    representation = instantiate(cfg.representation).to(device).eval()
    return SimpleNamespace(representation=representation)


def _frame_count(generated: np.ndarray, reference: np.ndarray | None, requested: int | None) -> int:
    candidates = [int(generated.shape[1])]
    if reference is not None:
        candidates.append(int(reference.shape[1]))
    if requested is not None:
        candidates.append(int(requested))
    frames = min(candidates)
    if frames <= 1:
        raise ValueError(f"benchmark frame count must be greater than 1, got {frames}")
    return frames


def _load_reference_artifact(path: str | Path, num_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = _load_npz(path)
    _require_keys(arrays, ("qpos_36", "fps"), "reference_qpos")
    qpos = np.asarray(arrays["qpos_36"], dtype=np.float32)
    if qpos.ndim != 3 or qpos.shape[-1] != 36:
        raise ValueError(f"reference qpos_36 must have shape (N,T,36), got {qpos.shape}")
    if qpos.shape[0] != int(num_samples):
        raise ValueError(f"reference sample count {qpos.shape[0]} != generated sample count {num_samples}")
    fps = np.asarray(arrays["fps"], dtype=np.float32).reshape(-1)
    if fps.shape != (num_samples,):
        raise ValueError(f"reference fps must have shape ({num_samples},), got {fps.shape}")
    if "valid" in arrays:
        valid = np.asarray(arrays["valid"], dtype=np.bool_)
        if valid.shape != qpos.shape[:2]:
            raise ValueError(f"reference valid must have shape {qpos.shape[:2]}, got {valid.shape}")
    else:
        valid = np.ones(qpos.shape[:2], dtype=np.bool_)
    return qpos, valid, fps



def _collect_reference_from_datasets(
    *,
    datasets: dict[str, Any],
    records: list[SampleRecord],
    frames: int,
    batch_size: int,
    mode: str,
) -> dict[str, Any]:
    reference_chunks = []
    valid_chunks = []
    fps_chunks = []
    reference_body_chunks = []
    audio_chunks = []
    captions = []
    dataset_names = []
    dataset_indices = []
    for start in range(0, len(records), int(batch_size)):
        batch_records = records[start : start + int(batch_size)]
        items = [datasets[record.dataset][record.index] for record in batch_records]
        batch = motion_collate_fn(items)
        reference_chunks.append(batch["qpos_36"][:, :frames].detach().cpu())
        valid = batch["mask"]["valid"][:, :frames].detach().cpu().bool()
        if mode in {"audio", "humanref"}:
            valid = finite_valid(batch["mask"]["valid"], frames)
            reference_body_chunks.append(batch["body_pos_w"][:, :frames].detach().cpu())
        if mode == "audio":
            audio_chunks.append(batch["audio_features"][:, :frames].detach().cpu())
        valid_chunks.append(valid)
        fps_chunks.append(batch["fps"].detach().cpu().float())
        captions.extend(str(item.get("caption", "")) for item in items)
        dataset_names.extend(record.dataset for record in batch_records)
        dataset_indices.extend(int(record.index) for record in batch_records)
    payload: dict[str, Any] = {
        "reference_qpos": torch.cat(reference_chunks, dim=0),
        "reference_valid": torch.cat(valid_chunks, dim=0),
        "fps": torch.cat(fps_chunks, dim=0),
        "captions": captions,
        "dataset_names": dataset_names,
        "dataset_indices": dataset_indices,
    }
    if mode in {"audio", "humanref"}:
        payload["reference_body_pos"] = torch.cat(reference_body_chunks, dim=0)
    if mode == "audio":
        payload["audio_features"] = torch.cat(audio_chunks, dim=0)
    return payload


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    metric_directions = {
        "FID(seq)": "min", "KID(seq)": "min", "R@1": "max", "R@2": "max", "R@3": "max",
        "Diversity": "max", "Matching": "min", "MultiModal Dist": "min",
        "BeatAlign": "max",
        "FIDk": "min", "FIDg": "min", "Divk": "max", "Divg": "max", "PFC": "min", "PFC Ref": "min",
        "contact_sliding": "min", "foot_ground_error": "min", "body_jerk": "min", "PJ": "min", "AUJ": "min",
        "Tracker g-MPJPE": "min", "Tracker MPJPE": "min", "Tracker E_vel": "min", "Tracker E_acc": "min",
    }
    markdown = generation_benchmark_markdown(rows=rows, metric_directions=metric_directions, metadata=metadata)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")


def _run_text_artifact(args: argparse.Namespace, output_dir: Path, model: Any, device: torch.device, payload: dict[str, Any]) -> BenchmarkResult:
    generated_qpos = payload["generated_qpos"]
    reference_qpos = payload["reference_qpos"]
    reference_valid = payload["reference_valid"]
    fps = payload["fps"]
    captions = payload["captions"]
    dataset_names = payload["dataset_names"]
    dataset_indices = payload["dataset_indices"]
    motion_encoder, evaluator_checkpoint = _load_evaluator_motion_encoder(args.evaluator_checkpoint, args.motion_key, device)
    embedding_frames = int(reference_qpos.shape[1])
    generated_for_embedding = generated_qpos[:, :embedding_frames]
    generated_valid = torch.ones(generated_for_embedding.shape[:2], dtype=torch.bool)
    transition_values, transition_metrics, _ = _compute_text_transition_metrics(
        model=model, generated_qpos=generated_qpos, reference_qpos=reference_qpos, args=args, device=device
    )
    reference_motion = _motion_for_evaluator(reference_qpos, args.motion_key, kinematics=model.representation.kinematics)
    generated_motion = _motion_for_evaluator(generated_for_embedding, args.motion_key, kinematics=model.representation.kinematics)
    reference_embeddings = _encode_motion_embeddings(motion_encoder, reference_motion, reference_valid, batch_size=args.batch_size, device=device)
    generated_embeddings = _encode_motion_embeddings(motion_encoder, generated_motion, generated_valid, batch_size=args.batch_size, device=device)
    physical_values = _physical_values(
        generated_qpos, fps, representation=model.representation, device=device, contact_height_threshold=args.contact_height_threshold, contact_penetration_tolerance=args.contact_penetration_tolerance,
    )
    text_embeddings = None
    if args.enable_text_metrics:
        text_embeddings = _encode_text_embeddings(
            evaluator_checkpoint, captions, model_name=args.t5_3b_model, cache_dir=args.text_cache_dir,
            batch_size=args.text_batch_size, device=device
        )
    real_motion_metrics = _real_motion_metrics(reference_embeddings, reference_embeddings, text_embeddings)
    embedding_metrics = _embedding_distribution_metrics(reference_embeddings, generated_embeddings)
    embedding_metrics.update({
        "embedding_frames": embedding_frames,
        "motion_key": args.motion_key,
        "evaluator_checkpoint": str(Path(args.evaluator_checkpoint).resolve()),
        "multimodality": None,
    })
    if text_embeddings is not None:
        generated_text = _text_retrieval_summary(
            generated_embeddings,
            text_embeddings,
            dataset_names=dataset_names,
            stratified=True,
            seed=int(args.seed),
        )
        reference_text = _text_retrieval_summary(
            reference_embeddings,
            text_embeddings,
            dataset_names=dataset_names,
            stratified=True,
            seed=int(args.seed),
        )
        embedding_metrics.update({
            "matching_score_generated": generated_text["matching_score"],
            "matching_score_reference": reference_text["matching_score"],
            "r_precision_generated": generated_text["r_precision"],
            "r_precision_reference": reference_text["r_precision"],
            "text_generated": generated_text,
            "text_reference": reference_text,
            "text_encoder_model": str(args.t5_3b_model),
            "text_cache_dir": str(args.text_cache_dir),
            "text_retrieval_batch_size": TEXT_RETRIEVAL_BATCH_SIZE,
        })
    physical_metrics = _physical_summary_from_values(physical_values)
    physical_metrics["frames"] = int(generated_qpos.shape[1])
    physical_metrics["contact_height_threshold"] = float(args.contact_height_threshold)
    physical_metrics["contact_penetration_tolerance"] = float(args.contact_penetration_tolerance)
    dataset_metrics = _text_dataset_metrics(
        dataset_names=dataset_names, reference_embeddings=reference_embeddings, generated_embeddings=generated_embeddings,
        physical_values=physical_values, text_embeddings=text_embeddings, transition_values=transition_values,
        transition_chunk_length=transition_metrics.get("chunk_length"), transition_num_frames=int(generated_qpos.shape[1]),
    )
    sample_rows = _text_sample_metric_rows(
        captions=captions, dataset_names=dataset_names, dataset_indices=dataset_indices, physical_values=physical_values,
        generated_embeddings=generated_embeddings, reference_embeddings=reference_embeddings, text_embeddings=text_embeddings,
        transition_values=transition_values,
    )
    sample_rankings = _sample_rankings(sample_rows)
    tracker_executed = run_tracker_executed_benchmark(
        output_dir=output_dir, generated_qpos=generated_qpos, reference_qpos=reference_qpos, fps=fps,
        captions=captions, dataset_names=dataset_names, dataset_indices=dataset_indices, model=model, args=args, device=device,
    )
    for name, data in (
        ("embedding_metrics", embedding_metrics), ("real_motion_metrics", real_motion_metrics),
        ("physical_metrics", physical_metrics), ("transition_metrics", transition_metrics),
        ("dataset_metrics", dataset_metrics), ("sample_rankings", sample_rankings), ("tracker_executed", tracker_executed),
    ):
        _finite_metrics(data)
        _save_json(output_dir / f"{name}.json", data)
    _write_jsonl(output_dir / "sample_metrics.jsonl", sample_rows)
    np.savez_compressed(output_dir / "embeddings.npz", reference_embeddings=reference_embeddings.astype(np.float32), generated_embeddings=generated_embeddings.astype(np.float32), **({"text_embeddings": text_embeddings.astype(np.float32)} if text_embeddings is not None else {}))
    benchmark = {
        "benchmark": "artifact_text", "label": args.label, "generated_qpos": str(Path(args.generated_qpos).resolve()),
        "reference_qpos": str((output_dir / "reference_qpos.npz").resolve()), "samples_path": None if args.samples_path is None else str(Path(args.samples_path).resolve()),
        "exp": args.exp, "split": args.split, "num_samples": int(generated_qpos.shape[0]), "num_frames": int(generated_qpos.shape[1]),
        "embedding_frames": embedding_frames, "embedding_metrics": embedding_metrics, "real_motion_metrics": real_motion_metrics,
        "physical_metrics": physical_metrics, "transition_metrics": transition_metrics, "tracker_executed": tracker_executed,
        "dataset_metrics": dataset_metrics, "sample_metrics_path": str((output_dir / "sample_metrics.jsonl").resolve()),
        "sample_rankings_path": str((output_dir / "sample_rankings.json").resolve()),
    }
    _save_json(output_dir / "benchmark.json", benchmark)
    _save_json(output_dir / "metrics.json", {"benchmark": benchmark, "label": args.label})
    _write_summary(output_dir, [_text_summary_row(benchmark, label=args.label)], {"benchmark": "artifact_text", "label": args.label, "exp": args.exp, "split": args.split})
    return BenchmarkResult(benchmark=benchmark, text_embeddings=text_embeddings)


def _run_audio_artifact(args: argparse.Namespace, output_dir: Path, model: Any, device: torch.device, payload: dict[str, Any]) -> BenchmarkResult:
    generated_qpos = payload["generated_qpos"]
    reference_qpos = payload["reference_qpos"]
    fps = payload["fps"]
    audio_features = payload["audio_features"]
    captions = payload["captions"]
    dataset_names = payload["dataset_names"]
    dataset_indices = payload["dataset_indices"]
    generated_body_pos = qpos_to_body_positions(model, generated_qpos, batch_size=args.batch_size, device=device)
    reference_body_pos = qpos_to_body_positions(model, reference_qpos, batch_size=args.batch_size, device=device)
    transition_values, transition_metrics = _compute_audio_transition_metrics(
        model=model, generated_qpos=generated_qpos, reference_qpos=reference_qpos,
        generated_body_pos=generated_body_pos, reference_body_pos=reference_body_pos, args=args,
    )
    physical_values = _physical_values(
        generated_qpos, fps, representation=model.representation, device=device, contact_height_threshold=args.contact_height_threshold, contact_penetration_tolerance=args.contact_penetration_tolerance,
    )
    audio_values = _beat_align_values(audio_features=audio_features, generated_body_pos=generated_body_pos, reference_body_pos=reference_body_pos, fps=fps, args=args)
    dance_features = aistpp_edge_features(
        reference_body_pos.numpy(), generated_body_pos.numpy(), fps=fps.numpy(), root_index=int(args.pfc_root_index),
        up_axis=args.aistpp_up_axis, parent_edges=_kinematic_parent_edges(model),
        left_foot_indices=tuple(args.pfc_left_foot_indices), right_foot_indices=tuple(args.pfc_right_foot_indices),
    )
    dance_values = _dance_sample_values(dance_features)
    audio_metrics = summarize_values(audio_values)
    audio_metrics.update({"frames": int(generated_qpos.shape[1]), "beat_direction": args.beat_direction, "beat_threshold": float(args.beat_threshold), "sigma_frames": float(args.sigma_frames), "min_motion_beat_distance_seconds": float(args.min_motion_beat_distance_seconds)})
    physical_metrics = _physical_summary_from_values(physical_values)
    physical_metrics["frames"] = int(generated_qpos.shape[1])
    physical_metrics["contact_height_threshold"] = float(args.contact_height_threshold)
    physical_metrics["contact_penetration_tolerance"] = float(args.contact_penetration_tolerance)
    dance_metrics = aistpp_edge_metric_summary(dance_features)
    dance_metrics.update({"frames": int(generated_qpos.shape[1]), "up_axis": args.aistpp_up_axis, "root_index": int(args.pfc_root_index), "left_foot_indices": list(args.pfc_left_foot_indices), "right_foot_indices": list(args.pfc_right_foot_indices), "geometric_feature": "g1_root_relative_body_geometry", "kinetic_feature": "fairmotion_root_relative_joint_kinetic_energy"})
    dataset_metrics = _audio_dataset_metrics(
        dataset_names=dataset_names, physical_values=physical_values, audio_values=audio_values, dance_features=dance_features,
        transition_values=transition_values, transition_chunk_length=transition_metrics.get("chunk_length"), transition_num_frames=int(generated_qpos.shape[1]),
    )
    sample_rows = _audio_sample_rows(captions=captions, dataset_names=dataset_names, dataset_indices=dataset_indices, audio_values=audio_values, physical_values=physical_values, dance_values=dance_values, transition_values=transition_values)
    tracker_executed = run_tracker_executed_benchmark(
        output_dir=output_dir, generated_qpos=generated_qpos, reference_qpos=reference_qpos, fps=fps,
        captions=captions, dataset_names=dataset_names, dataset_indices=dataset_indices, model=model, args=args, device=device,
    )
    for name, data in (("audio_metrics", audio_metrics), ("aistpp_edge_metrics", dance_metrics), ("physical_metrics", physical_metrics), ("transition_metrics", transition_metrics), ("dataset_metrics", dataset_metrics), ("tracker_executed", tracker_executed)):
        _finite_metrics(data)
        _save_json(output_dir / f"{name}.json", data)
    _write_jsonl(output_dir / "sample_metrics.jsonl", sample_rows)
    benchmark = {
        "benchmark": "artifact_audio", "label": args.label, "generated_qpos": str(Path(args.generated_qpos).resolve()),
        "reference_qpos": str((output_dir / "reference_qpos.npz").resolve()), "samples_path": None if args.samples_path is None else str(Path(args.samples_path).resolve()),
        "exp": args.exp, "split": args.split, "num_samples": int(generated_qpos.shape[0]), "num_frames": int(generated_qpos.shape[1]),
        "audio_metrics": audio_metrics, "aistpp_edge_metrics": dance_metrics, "physical_metrics": physical_metrics,
        "transition_metrics": transition_metrics, "tracker_executed": tracker_executed, "dataset_metrics": dataset_metrics,
        "sample_metrics_path": str((output_dir / "sample_metrics.jsonl").resolve()),
    }
    _save_json(output_dir / "benchmark.json", benchmark)
    _save_json(output_dir / "metrics.json", {"benchmark": benchmark, "label": args.label})
    _write_summary(output_dir, [_audio_summary_row(benchmark, label=args.label)], {"benchmark": "artifact_audio", "label": args.label, "exp": args.exp, "split": args.split})
    return BenchmarkResult(benchmark=benchmark)



def _dataset_metric_summaries(
    *,
    dataset_names: list[str],
    physical_values: dict[str, np.ndarray],
    tracking_values: dict[str, np.ndarray],
    transition_values: dict[str, np.ndarray] | None = None,
    transition_metrics: dict[str, Any] | None = None,
    num_frames: int | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    by_name: dict[str, list[int]] = {}
    for idx, name in enumerate(dataset_names):
        by_name.setdefault(str(name), []).append(idx)
    for name, indices in sorted(by_name.items()):
        idx = np.asarray(indices, dtype=np.int64)
        physical_subset = {key: np.asarray(values)[idx] for key, values in physical_values.items()}
        tracking_subset = {key: np.asarray(values)[idx] for key, values in tracking_values.items()}
        row: dict[str, Any] = {
            "num_samples": int(len(indices)),
            "physical": _physical_summary_from_values(physical_subset),
            "tracking": summarize_values(tracking_subset),
        }
        if transition_values and transition_metrics and transition_metrics.get("enabled"):
            transition_subset = {}
            for key, values in transition_values.items():
                array = np.asarray(values)
                if array.ndim == 0:
                    continue
                if array.shape[0] != len(dataset_names):
                    continue
                transition_subset[key] = array[idx]
            row["transition"] = summarize_values(transition_subset)
            row["transition"].update({"chunk_length": int(transition_metrics["chunk_length"]), "num_frames": int(num_frames or 0)})
        output[name] = row
    return output


def _run_humanref_artifact(args: argparse.Namespace, output_dir: Path, model: Any, device: torch.device, payload: dict[str, Any]) -> BenchmarkResult:
    generated_qpos = payload["generated_qpos"]
    reference_qpos = payload["reference_qpos"]
    fps = payload["fps"]
    captions = payload["captions"]
    dataset_names = payload["dataset_names"]
    dataset_indices = payload["dataset_indices"]

    generated_body_pos = qpos_to_body_positions(model, generated_qpos, batch_size=args.batch_size, device=device)
    reference_body_pos = qpos_to_body_positions(model, reference_qpos, batch_size=args.batch_size, device=device)
    physical_values = _physical_values(
        generated_qpos,
        fps,
        representation=model.representation,
        device=device,
        contact_height_threshold=args.contact_height_threshold,
        contact_penetration_tolerance=args.contact_penetration_tolerance,
    )
    transition_values, transition_metrics = _compute_humanref_transition_metrics(
        model=model,
        generated_qpos=generated_qpos,
        reference_qpos=reference_qpos,
        generated_body_pos=generated_body_pos,
        reference_body_pos=reference_body_pos,
        args=args,
    )
    tracking_values = _humanref_tracking_values(
        generated_body_pos=generated_body_pos,
        reference_body_pos=reference_body_pos,
        root_index=int(args.root_index),
        end_effector_indices=_humanref_end_effector_indices(model),
    )
    tracking_metrics = summarize_values(tracking_values)
    tracking_metrics.update({"frames": int(generated_qpos.shape[1]), "root_index": int(args.root_index)})
    physical_metrics = _physical_summary_from_values(physical_values)
    physical_metrics["frames"] = int(generated_qpos.shape[1])
    physical_metrics["contact_height_threshold"] = float(args.contact_height_threshold)
    physical_metrics["contact_penetration_tolerance"] = float(args.contact_penetration_tolerance)
    embedding_metrics = {
        "enabled": False,
        "reason": "humanref artifact benchmark does not use a text evaluator",
        "motion_fid": None,
        "motion_kid": None,
        "diversity_generated": None,
    }
    dataset_metrics = _dataset_metric_summaries(
        dataset_names=dataset_names,
        physical_values=physical_values,
        tracking_values=tracking_values,
        transition_values=transition_values,
        transition_metrics=transition_metrics,
        num_frames=int(generated_qpos.shape[1]),
    )
    sample_rows = _humanref_sample_rows(
        captions=captions,
        dataset_names=dataset_names,
        dataset_indices=dataset_indices,
        tracking_values=tracking_values,
        physical_values=physical_values,
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
    for name, data in (
        ("embedding_metrics", embedding_metrics),
        ("tracking_metrics", tracking_metrics),
        ("physical_metrics", physical_metrics),
        ("transition_metrics", transition_metrics),
        ("dataset_metrics", dataset_metrics),
        ("tracker_executed", tracker_executed),
    ):
        _finite_metrics(data)
        _save_json(output_dir / f"{name}.json", data)
    _write_jsonl(output_dir / "sample_metrics.jsonl", sample_rows)
    np.savez_compressed(
        output_dir / "generated_qpos.npz",
        qpos_36=generated_qpos.numpy().astype(np.float32, copy=False),
        body_pos_w=generated_body_pos.numpy().astype(np.float32, copy=False),
        fps=fps.numpy().astype(np.float32, copy=False),
        captions=np.asarray(captions, dtype=np.str_),
        dataset=np.asarray(dataset_names, dtype=np.str_),
        dataset_index=np.asarray(dataset_indices, dtype=np.int32),
        source_generated_qpos=np.asarray([str(Path(args.generated_qpos).resolve())], dtype=np.str_),
    )
    np.savez_compressed(
        output_dir / "reference_qpos.npz",
        qpos_36=reference_qpos.numpy().astype(np.float32, copy=False),
        body_pos_w=reference_body_pos.numpy().astype(np.float32, copy=False),
        valid=payload["reference_valid"].numpy().astype(np.bool_, copy=False),
        fps=fps.numpy().astype(np.float32, copy=False),
        captions=np.asarray(captions, dtype=np.str_),
        dataset=np.asarray(dataset_names, dtype=np.str_),
        dataset_index=np.asarray(dataset_indices, dtype=np.int32),
    )
    benchmark = {
        "benchmark": "artifact_humanref",
        "label": args.label,
        "generated_qpos": str(Path(args.generated_qpos).resolve()),
        "reference_qpos": str((output_dir / "reference_qpos.npz").resolve()),
        "samples_path": None if args.samples_path is None else str(Path(args.samples_path).resolve()),
        "exp": args.exp,
        "split": args.split,
        "num_samples": int(generated_qpos.shape[0]),
        "num_frames": int(generated_qpos.shape[1]),
        "embedding_metrics": embedding_metrics,
        "tracking_metrics": tracking_metrics,
        "physical_metrics": physical_metrics,
        "transition_metrics": transition_metrics,
        "tracker_executed": tracker_executed,
        "dataset_metrics": dataset_metrics,
        "sample_metrics_path": str((output_dir / "sample_metrics.jsonl").resolve()),
    }
    _save_json(output_dir / "benchmark.json", benchmark)
    _save_json(output_dir / "metrics.json", {"benchmark": benchmark, "label": args.label})
    _write_summary(output_dir, [_humanref_summary_row(benchmark, label=args.label)], {"benchmark": "artifact_humanref", "label": args.label, "exp": args.exp, "split": args.split})
    return BenchmarkResult(benchmark=benchmark)

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark externally generated G1 qpos artifacts.",
        allow_abbrev=False,
    )
    parser.add_argument("--mode", choices=["text", "audio", "humanref"], required=True)
    parser.add_argument("--generated_qpos", required=True)
    parser.add_argument("--reference_qpos", default=None)
    parser.add_argument(
        "--samples_path",
        required=True,
        help="Pinned omg.benchmark.sample.v2 manifest matching generated_qpos order.",
    )
    parser.add_argument("--exp", required=True)
    parser.add_argument("--data", default="omg_data_lerobot_omnimodal")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="Generated artifact")
    parser.add_argument("--motion_key", choices=["qpos_36", "body_link_pos_local", "body_pos_local"], default="qpos_36")
    parser.add_argument("--evaluator_checkpoint", default=None)
    parser.add_argument("--contact_height_threshold", type=float, default=0.12)
    parser.add_argument("--contact_penetration_tolerance", type=float, default=0.02)
    parser.add_argument("--transition-metrics", "--transition_metrics", dest="transition_metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transition-chunk-length", "--transition_chunk_length", dest="transition_chunk_length", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--enable_text_metrics", action="store_true")
    parser.add_argument("--t5_3b_model", default=None)
    parser.add_argument("--text_cache_dir", default=None)
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--beat_threshold", type=float, default=0.5)
    parser.add_argument("--sigma_frames", type=float, default=3.0)
    parser.add_argument("--min_motion_beat_distance_seconds", type=float, default=0.25)
    parser.add_argument("--beat_direction", choices=["music_to_motion", "motion_to_music"], default="music_to_motion")
    parser.add_argument("--aistpp_up_axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--pfc_root_index", type=int, default=0)
    parser.add_argument("--pfc_left_foot_indices", type=int, nargs="+", default=[5, 6])
    parser.add_argument("--pfc_right_foot_indices", type=int, nargs="+", default=[11, 12])
    parser.add_argument("--root_index", type=int, default=0)
    add_tracker_executed_args(parser)
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides.")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.num_frames is not None and args.num_frames <= 1:
        raise ValueError("--num_frames must be greater than 1")
    if args.mode == "text" and args.evaluator_checkpoint is None:
        raise ValueError("text artifact benchmark requires --evaluator_checkpoint")
    if args.evaluator_checkpoint is not None and not Path(args.evaluator_checkpoint).exists():
        raise FileNotFoundError(f"--evaluator_checkpoint does not exist: {args.evaluator_checkpoint}")
    if args.enable_text_metrics and (args.t5_3b_model is None or args.text_cache_dir is None):
        raise ValueError("--enable_text_metrics requires both --t5_3b_model and --text_cache_dir")
    if args.mode == "text" and args.enable_text_metrics:
        generated_count = int(_load_npz(args.generated_qpos)["qpos_36"].shape[0])
        if generated_count % TEXT_RETRIEVAL_BATCH_SIZE != 0:
            raise ValueError(f"--enable_text_metrics uses fixed batch{TEXT_RETRIEVAL_BATCH_SIZE}; generated_qpos has {generated_count} samples")
    if args.mode == "audio" and (not args.pfc_left_foot_indices or not args.pfc_right_foot_indices):
        raise ValueError("PFC foot index lists must be non-empty")
    validate_tracker_executed_args(args)
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = _device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_arrays = _load_npz(args.generated_qpos)
    _validate_generated(generated_arrays)
    records = _load_sample_records(Path(args.samples_path))
    cfg = _compose_cfg(args)
    include_datasets = args.datasets
    if include_datasets is None:
        include_datasets = _dataset_names_from_records(records)
    datasets = _build_datasets(
        cfg,
        args.split,
        include=include_datasets,
        num_frames=args.num_frames,
    )
    records = _resolve_sample_records(records, datasets)
    _validate_sample_records(records, datasets)
    _assert_records_match_artifact(records, generated_arrays)
    reference_np = valid_np = reference_fps_np = None
    if args.reference_qpos is not None:
        reference_np, valid_np, reference_fps_np = _load_reference_artifact(args.reference_qpos, num_samples=len(records))
    frames = _frame_count(generated_arrays["qpos_36"], reference_np, args.num_frames)
    payload: dict[str, Any] = {
        "generated_qpos": torch.from_numpy(np.asarray(generated_arrays["qpos_36"][:, :frames], dtype=np.float32)),
        "fps": torch.from_numpy(np.asarray(generated_arrays["fps"], dtype=np.float32)),
        "captions": [str(value) for value in generated_arrays["captions"].tolist()],
        "dataset_names": [str(value) for value in generated_arrays["dataset"].tolist()],
        "dataset_indices": [int(value) for value in generated_arrays["dataset_index"].tolist()],
    }
    if reference_np is None:
        payload.update(
            _collect_reference_from_datasets(
                datasets=datasets,
                records=records,
                frames=frames,
                batch_size=args.batch_size,
                mode=args.mode,
            )
        )
    else:
        payload["reference_qpos"] = torch.from_numpy(np.asarray(reference_np[:, :frames], dtype=np.float32))
        payload["reference_valid"] = torch.from_numpy(np.asarray(valid_np[:, :frames], dtype=np.bool_))
        if args.mode in {"audio", "humanref"}:
            condition_payload = _collect_reference_from_datasets(
                datasets=datasets,
                records=records,
                frames=frames,
                batch_size=args.batch_size,
                mode=args.mode,
            )
            for key in ("reference_body_pos", "audio_features"):
                if key in condition_payload:
                    payload[key] = condition_payload[key]
    if payload["reference_qpos"].shape[:2] != payload["generated_qpos"].shape[:2]:
        raise ValueError(f"reference/generation shape mismatch: {payload['reference_qpos'].shape} vs {payload['generated_qpos'].shape}")
    _write_jsonl(output_dir / "artifact_samples.jsonl", [
        {"sample_index": idx, "dataset": payload["dataset_names"][idx], "dataset_index": int(payload["dataset_indices"][idx]), "caption": payload["captions"][idx]}
        for idx in range(len(payload["captions"]))
    ])
    _write_sample_records(output_dir / "samples.jsonl", records, datasets)
    np.savez_compressed(output_dir / "generated_qpos.npz", qpos_36=payload["generated_qpos"].numpy().astype(np.float32), fps=payload["fps"].numpy().astype(np.float32), captions=np.asarray(payload["captions"], dtype=np.str_), dataset=np.asarray(payload["dataset_names"], dtype=np.str_), dataset_index=np.asarray(payload["dataset_indices"], dtype=np.int32), source_generated_qpos=np.asarray([str(Path(args.generated_qpos).resolve())], dtype=np.str_))
    np.savez_compressed(output_dir / "reference_qpos.npz", qpos_36=payload["reference_qpos"].numpy().astype(np.float32), valid=payload["reference_valid"].numpy().astype(np.bool_), fps=(reference_fps_np.astype(np.float32) if reference_fps_np is not None else payload["fps"].numpy().astype(np.float32)), captions=np.asarray(payload["captions"], dtype=np.str_), dataset=np.asarray(payload["dataset_names"], dtype=np.str_), dataset_index=np.asarray(payload["dataset_indices"], dtype=np.int32))
    model = _metric_model(cfg, device)
    if args.mode == "text":
        result = _run_text_artifact(args, output_dir, model, device, payload)
    elif args.mode == "audio":
        result = _run_audio_artifact(args, output_dir, model, device, payload)
    else:
        result = _run_humanref_artifact(args, output_dir, model, device, payload)
    print(json.dumps(result.benchmark, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
