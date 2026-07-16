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

from omg.benchmarks.evaluator.text_encoder import TextEncoder
from omg.benchmarks.metrics import (
    diversity,
    matching_score,
    multimodality,
    motion_fid,
    motion_kid,
    transition_metric_summary,
    transition_metric_values,
)
from omg.benchmarks.runners.common import (
    BenchmarkResult,
    SampleRecord,
    _build_datasets,
    _cfg_output_name,
    _cfg_scale_json,
    _config_dir,
    _dataset_filter_tokens_from_records,
    _dataset_indices,
    _device,
    _embedding_distribution_metrics,
    _encode_motion_embeddings,
    _finite_metrics,
    _load_evaluator_motion_encoder,
    _load_model,
    _load_sample_records,
    _motion_for_evaluator,
    _parse_cfg_scale_value,
    _physical_summary_from_values,
    _physical_values,
    _qpos_to_body_positions,
    _sample_file_path,
    _save_json,
    _summary_stats,
    _validate_sample_records,
    _write_jsonl,
    _write_sample_records,
)
from omg.benchmarks.runners.tracker_executed import (
    add_tracker_executed_args,
    run_tracker_executed_benchmark,
    validate_tracker_executed_args,
)
from omg.benchmarks.report import generation_benchmark_markdown
from omg.data.datamodule import motion_collate_fn


TEXT_RETRIEVAL_BATCH_SIZE = 32
DEFAULT_T5_3B_MODEL = os.environ.get("OMG_T5_3B_MODEL", "t5-3b")
DEFAULT_TEXT_CACHE_DIR = os.environ.get(
    "OMG_TEXT_CACHE_DIR",
    "outputs/evaluator_text_cache/filtered_original_g1_tmr_bodypos_60f",
)


def _has_caption(dataset: Any, index: int) -> bool:
    samples = getattr(dataset, "samples", None)
    if samples is not None and index < len(samples):
        return str(samples[index].get("segment_caption", "")).strip() != ""
    return str(dataset[index].get("caption", "")).strip() != ""


def _select_sample_records(
    datasets: dict[str, Any],
    num_samples: int,
    seed: int,
    *,
    require_min_two: bool = True,
) -> list[SampleRecord]:
    if require_min_two and num_samples < 2:
        raise ValueError("--num-texts must be at least 2 for FID/KID/diversity")
    names = list(datasets.keys())
    candidate_spans: list[tuple[str, int, int, int]] = []
    global_index = 0
    per_dataset_counts = {}
    print("[INFO] Indexing caption-bearing samples…")
    for name in names:
        dataset = datasets[name]
        count = 0
        captions = getattr(dataset, "captions", None)
        window_offsets = getattr(dataset, "window_offsets", None)
        if captions is not None and window_offsets is not None and len(window_offsets) == len(captions) + 1:
            for episode_index, caption in enumerate(captions):
                if not str(caption).strip():
                    continue
                start = int(window_offsets[episode_index])
                end = min(int(window_offsets[episode_index + 1]), len(dataset))
                if end <= start:
                    continue
                span_count = end - start
                candidate_spans.append((name, start, global_index + start, span_count))
                count += span_count
        else:
            for index in tqdm(
                range(len(dataset)),
                desc=f"Captions {name}",
                unit="idx",
                leave=False,
            ):
                if _has_caption(dataset, index):
                    candidate_spans.append((name, index, global_index + index, 1))
                    count += 1
        global_index += len(dataset)
        per_dataset_counts[name] = count
    if not candidate_spans:
        raise ValueError("No caption-bearing samples found in selected datasets")
    candidate_count = sum(span[3] for span in candidate_spans)
    if int(num_samples) > candidate_count:
        print(
            f"[WARN] Requested {num_samples} texts but only found {candidate_count} caption-bearing samples; "
            "evaluating all available samples."
        )
        num_samples = candidate_count
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(candidate_count, size=int(num_samples), replace=False))
    span_ends = np.cumsum(np.asarray([span[3] for span in candidate_spans], dtype=np.int64))
    selected_spans = np.searchsorted(span_ends, selected, side="right")
    records: list[SampleRecord] = []
    selected_counts: dict[str, int] = {}
    for candidate_index, span_index in zip(selected, selected_spans, strict=True):
        dataset_name, span_start, span_global_start, _ = candidate_spans[int(span_index)]
        previous_end = 0 if int(span_index) == 0 else int(span_ends[int(span_index) - 1])
        offset = int(candidate_index) - previous_end
        index = span_start + offset
        candidate_global = span_global_start + offset
        selected_counts[dataset_name] = selected_counts.get(dataset_name, 0) + 1
        records.append(SampleRecord(dataset=dataset_name, index=index, global_index=candidate_global))
    print(f"[INFO] caption-bearing samples by dataset: {per_dataset_counts}")
    print(f"[INFO] selected benchmark samples by dataset: {selected_counts}")
    return records


def _encode_text_embeddings(
    checkpoint: dict[str, Any],
    captions: list[str],
    *,
    model_name: str,
    cache_dir: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    from omg.benchmarks.evaluator.text_cache import encode_texts_with_cache

    config = checkpoint.get("config") or {}
    output_dim = int(config.get("embedding_dim", 512))
    max_length = int(config.get("text_max_length", 100))
    text_encoder = TextEncoder(output_dim=output_dim, model_name=model_name, max_length=max_length).to(device)
    text_encoder.proj.load_state_dict(checkpoint["text_proj_state_dict"], strict=True)
    text_encoder.eval()
    chunks = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(captions), batch_size),
            desc="Text embeddings",
            unit="batch",
            leave=False,
        ):
            chunk = captions[start : start + batch_size]
            encoded = encode_texts_with_cache(text_encoder, chunk, device=device, cache_root=cache_dir)
            chunks.append(encoded.detach().cpu())
    return torch.cat(chunks, dim=0).numpy()


def _transition_chunk_length(model: Any, args: argparse.Namespace) -> int:
    return int(args.transition_chunk_length or getattr(model.representation, "sequence_length"))


def _compute_transition_metrics(
    *,
    model: Any,
    generated_qpos: torch.Tensor,
    reference_qpos: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any], torch.Tensor | None]:
    chunk_length = _transition_chunk_length(model, args)
    if not args.transition_metrics:
        return {}, {"enabled": False, "reason": "disabled", "chunk_length": chunk_length}, None
    if generated_qpos.shape[1] <= chunk_length:
        return (
            {},
            {
                "enabled": False,
                "reason": "num_frames must be greater than chunk_length",
                "num_frames": int(generated_qpos.shape[1]),
                "chunk_length": chunk_length,
            },
            None,
        )
    body_pos = _qpos_to_body_positions(model, generated_qpos, batch_size=args.batch_size, device=device)
    reference_body_pos = _qpos_to_body_positions(model, reference_qpos, batch_size=args.batch_size, device=device)
    values = transition_metric_values(
        qpos=generated_qpos.numpy(),
        reference_qpos=reference_qpos.numpy(),
        body_pos=body_pos.numpy(),
        reference_body_pos=reference_body_pos.numpy(),
        chunk_length=chunk_length,
    )
    summary = transition_metric_summary(values, chunk_length=chunk_length, num_frames=int(generated_qpos.shape[1]))
    summary["enabled"] = True
    return values, summary, body_pos


def _real_motion_metrics(
    distribution_reference_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
    text_embeddings: np.ndarray | None,
) -> dict[str, Any]:
    distribution_reference = np.asarray(distribution_reference_embeddings)
    eval_motion = np.asarray(eval_embeddings)
    result: dict[str, Any] = {
        "num_samples": int(eval_motion.shape[0]),
        "num_reference_samples": int(distribution_reference.shape[0]),
        "motion_fid_split_half": None,
        "motion_kid_split_half": None,
        "motion_fid_reference_eval": None,
        "motion_kid_reference_eval": None,
        "diversity_real": diversity(eval_motion) if eval_motion.shape[0] >= 2 else None,
        "multimodality": "-",
    }
    if distribution_reference.shape[0] >= 2 and eval_motion.shape[0] >= 2:
        fid = motion_fid(distribution_reference, eval_motion)
        kid = motion_kid(distribution_reference, eval_motion)
        result["motion_fid_reference_eval"] = fid
        result["motion_kid_reference_eval"] = kid
        result["motion_fid_split_half"] = fid
        result["motion_kid_split_half"] = kid
    if text_embeddings is not None:
        reference_text = _text_retrieval_summary(eval_motion, text_embeddings)
        result.update(
            {
                "matching_score_real": reference_text["matching_score"],
                "matching_distance_real": reference_text["matching_distance"],
                "r_precision_real": reference_text["r_precision"],
                "text_real": reference_text,
            }
        )
    return result


def _pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    squared = (
        np.square(left).sum(axis=1, keepdims=True)
        - 2.0 * left.dot(right.T)
        + np.square(right).sum(axis=1)[None, :]
    )
    return np.sqrt(np.maximum(squared, 0.0))


def _retrieval_distances_and_ranks(
    motion_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = _pairwise_distances(motion_embeddings, text_embeddings)
    if distances.shape[0] != distances.shape[1]:
        raise ValueError("retrieval metrics require paired motion/text embeddings with equal length")
    matched = np.diag(distances)
    ranks = (distances < matched[:, None]).sum(axis=1).astype(np.int64) + 1
    return matched.astype(np.float64), ranks


def _stratified_retrieval_order(
    dataset_names: list[str] | None,
    num_samples: int,
    *,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    if dataset_names is None:
        return np.arange(num_samples, dtype=np.int64)
    if len(dataset_names) != num_samples:
        raise ValueError(f"dataset name count {len(dataset_names)} != sample count {num_samples}")
    rng = np.random.default_rng(int(seed))
    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(dataset_names):
        groups.setdefault(str(name), []).append(idx)
    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])
    order: list[int] = []
    cursors = {key: 0 for key in keys}
    rotate = 0
    while len(order) < num_samples:
        active = [key for key in keys if cursors[key] < len(groups[key])]
        if not active:
            break
        rotated = active[rotate % len(active) :] + active[: rotate % len(active)]
        batch: list[int] = []
        while len(batch) < batch_size and active:
            progressed = False
            for key in rotated:
                if cursors[key] >= len(groups[key]):
                    continue
                batch.append(groups[key][cursors[key]])
                cursors[key] += 1
                progressed = True
                if len(batch) == batch_size:
                    break
            if not progressed:
                break
            active = [key for key in keys if cursors[key] < len(groups[key])]
            rotated = active[rotate % len(active) :] + active[: rotate % len(active)] if active else []
        order.extend(batch)
        rotate += 1
    return np.asarray(order, dtype=np.int64)


def _retrieval_batch_stats(indices: np.ndarray, dataset_names: list[str] | None, batch_size: int) -> dict[str, Any]:
    if dataset_names is None or indices.size == 0:
        return {"unique_datasets_per_batch": None}
    counts = []
    for start in range(0, indices.size, batch_size):
        batch_indices = indices[start : start + batch_size]
        if batch_indices.size != batch_size:
            continue
        counts.append(len({str(dataset_names[int(index)]) for index in batch_indices}))
    if not counts:
        return {"unique_datasets_per_batch": None}
    return {"unique_datasets_per_batch": _summary_stats(np.asarray(counts, dtype=np.float64))}


def _batched_retrieval_metrics(
    motion_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    *,
    dataset_names: list[str] | None = None,
    batch_size: int = TEXT_RETRIEVAL_BATCH_SIZE,
    require_full_batches: bool = True,
    stratified: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    motion = np.asarray(motion_embeddings)
    text = np.asarray(text_embeddings)
    if motion.shape[0] != text.shape[0]:
        raise ValueError("retrieval metrics require paired motion/text embeddings with equal length")
    if batch_size <= 0:
        raise ValueError("retrieval batch size must be positive")
    if dataset_names is not None and len(dataset_names) != motion.shape[0]:
        raise ValueError(f"dataset name count {len(dataset_names)} != sample count {motion.shape[0]}")
    empty_float = np.asarray([], dtype=np.float64)
    empty_int = np.asarray([], dtype=np.int64)
    if motion.shape[0] < batch_size:
        return {
            "matched_distances": empty_float,
            "ranks": empty_int,
            "used_indices": empty_int,
            "order_policy": "stratified_by_dataset" if stratified and dataset_names is not None else "manifest_order",
            "batch_stats": {"unique_datasets_per_batch": None},
        }
    usable = (motion.shape[0] // batch_size) * batch_size
    if require_full_batches and usable != motion.shape[0]:
        raise ValueError(
            f"text retrieval metrics use fixed batch{batch_size}; "
            f"got {motion.shape[0]} samples, which is not divisible by {batch_size}"
        )
    order = _stratified_retrieval_order(dataset_names, motion.shape[0], batch_size=batch_size, seed=seed) if stratified else np.arange(motion.shape[0], dtype=np.int64)
    order = order[:usable]
    all_distances = []
    all_ranks = []
    for start in range(0, usable, batch_size):
        batch_indices = order[start : start + batch_size]
        distances = _pairwise_distances(motion[batch_indices], text[batch_indices])
        matched = np.diag(distances)
        ranks = (distances < matched[:, None]).sum(axis=1).astype(np.int64) + 1
        all_distances.append(matched.astype(np.float64))
        all_ranks.append(ranks)
    return {
        "matched_distances": np.concatenate(all_distances),
        "ranks": np.concatenate(all_ranks),
        "used_indices": order,
        "order_policy": "stratified_by_dataset" if stratified and dataset_names is not None else "manifest_order",
        "batch_stats": _retrieval_batch_stats(order, dataset_names, batch_size),
    }


def _batched_retrieval_distances_and_ranks(
    motion_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    *,
    batch_size: int = TEXT_RETRIEVAL_BATCH_SIZE,
    require_full_batches: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metrics = _batched_retrieval_metrics(
        motion_embeddings,
        text_embeddings,
        batch_size=batch_size,
        require_full_batches=require_full_batches,
    )
    return metrics["matched_distances"], metrics["ranks"], metrics["used_indices"]


def _r_precision_from_ranks(ranks: np.ndarray, top_k: int = 3) -> list[float] | None:
    if ranks.size == 0:
        return None
    return [float((ranks <= k).mean()) for k in range(1, top_k + 1)]


def _compute_multimodality_metrics(
    *,
    model: Any,
    datasets: dict[str, Any],
    records: list[SampleRecord],
    motion_encoder: MotionEncoder,
    args: argparse.Namespace,
    cfg_scale: float | None,
    embedding_frames: int,
    device: torch.device,
) -> dict[str, Any] | None:
    if args.multimodality_repeats <= 1:
        return None
    repeat_embeddings = []
    print(
        f"[INFO] Multimodality generation: repeats={args.multimodality_repeats} "
        f"num_texts={len(records)} pairs={args.multimodality_pairs}"
    )
    for repeat_idx in range(int(args.multimodality_repeats)):
        generated_qpos, *_ = _generate_qpos(
            model=model,
            datasets=datasets,
            records=records,
            args=args,
            cfg_scale=cfg_scale,
            desc=f"Multimodality generate {repeat_idx + 1}/{args.multimodality_repeats}",
        )
        generated_for_embedding = generated_qpos[:, :embedding_frames]
        generated_valid = torch.ones(generated_for_embedding.shape[:2], dtype=torch.bool)
        generated_motion_for_embedding = _motion_for_evaluator(
            generated_for_embedding,
            args.motion_key,
            kinematics=model.representation.kinematics,
        )
        embeddings = _encode_motion_embeddings(
            motion_encoder,
            generated_motion_for_embedding,
            generated_valid,
            batch_size=args.batch_size,
            device=device,
        )
        repeat_embeddings.append(embeddings)
    stacked = np.stack(repeat_embeddings, axis=1)
    return multimodality(
        stacked,
        num_pairs=int(args.multimodality_pairs),
        seed=int(args.seed),
    )


def _text_retrieval_summary(
    motion_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    *,
    dataset_names: list[str] | None = None,
    batch_size: int = TEXT_RETRIEVAL_BATCH_SIZE,
    require_full_batches: bool = True,
    stratified: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    paired_distances = matching_score(motion_embeddings, text_embeddings, reduction="none")
    retrieval = _batched_retrieval_metrics(
        motion_embeddings,
        text_embeddings,
        dataset_names=dataset_names,
        batch_size=batch_size,
        require_full_batches=require_full_batches,
        stratified=stratified,
        seed=seed,
    )
    ranks = retrieval["ranks"]
    used_indices = retrieval["used_indices"]
    summary: dict[str, Any] = {
        "matching_score": matching_score(motion_embeddings, text_embeddings),
        "matching_distance": _summary_stats(paired_distances),
        "retrieval_batch_size": int(batch_size),
        "retrieval_num_batches": int(len(used_indices) // batch_size),
        "retrieval_num_samples": int(len(used_indices)),
        "retrieval_order_policy": retrieval["order_policy"],
        "retrieval_unique_datasets_per_batch": retrieval["batch_stats"]["unique_datasets_per_batch"],
        "mean_text_rank": None,
        "median_text_rank": None,
        "r_precision": None,
    }
    if ranks.size > 0:
        summary.update(
            {
                "mean_text_rank": float(ranks.mean()),
                "median_text_rank": float(np.median(ranks)),
                "r_precision": _r_precision_from_ranks(ranks),
            }
        )
    return summary


def _sample_metric_rows(
    *,
    captions: list[str],
    dataset_names: list[str],
    dataset_indices: list[int],
    physical_values: dict[str, np.ndarray],
    generated_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    text_embeddings: np.ndarray | None,
    transition_values: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    generated_paired_distances = reference_paired_distances = None
    generated_rank_by_index: dict[int, int] = {}
    reference_rank_by_index: dict[int, int] = {}
    if text_embeddings is not None:
        generated_paired_distances = matching_score(generated_embeddings, text_embeddings, reduction="none")
        reference_paired_distances = matching_score(reference_embeddings, text_embeddings, reduction="none")
        generated_retrieval = _batched_retrieval_metrics(
            generated_embeddings,
            text_embeddings,
            dataset_names=dataset_names,
            stratified=True,
        )
        reference_retrieval = _batched_retrieval_metrics(
            reference_embeddings,
            text_embeddings,
            dataset_names=dataset_names,
            stratified=True,
        )
        generated_ranks = generated_retrieval["ranks"]
        reference_ranks = reference_retrieval["ranks"]
        generated_rank_indices = generated_retrieval["used_indices"]
        reference_rank_indices = reference_retrieval["used_indices"]
        generated_rank_by_index = {int(index): int(rank) for index, rank in zip(generated_rank_indices, generated_ranks)}
        reference_rank_by_index = {int(index): int(rank) for index, rank in zip(reference_rank_indices, reference_ranks)}

    rows = []
    for idx, caption in enumerate(captions):
        row: dict[str, Any] = {
            "sample_index": idx,
            "dataset": dataset_names[idx],
            "dataset_index": int(dataset_indices[idx]),
            "caption": caption,
        }
        for key, values in physical_values.items():
            row[key] = float(values[idx])
        if transition_values:
            for key, values in transition_values.items():
                array = np.asarray(values)
                row[key] = float(array if array.ndim == 0 else array[idx])
        if text_embeddings is not None:
            assert generated_paired_distances is not None and reference_paired_distances is not None
            generated_rank = generated_rank_by_index[idx]
            reference_rank = reference_rank_by_index[idx]
            row.update(
                {
                    "text_retrieval_batch_size": TEXT_RETRIEVAL_BATCH_SIZE,
                    "generated_matching_distance": float(generated_paired_distances[idx]),
                    "generated_text_rank": int(generated_rank),
                    "generated_r_at_1": bool(generated_rank <= 1),
                    "generated_r_at_2": bool(generated_rank <= 2),
                    "generated_r_at_3": bool(generated_rank <= 3),
                    "reference_matching_distance": float(reference_paired_distances[idx]),
                    "reference_text_rank": int(reference_rank),
                    "reference_r_at_1": bool(reference_rank <= 1),
                    "reference_r_at_2": bool(reference_rank <= 2),
                    "reference_r_at_3": bool(reference_rank <= 3),
                    "generated_reference_distance_gap": float(generated_paired_distances[idx] - reference_paired_distances[idx]),
                }
            )
        rows.append(row)
    return rows


def _compact_sample(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "sample_index",
        "dataset",
        "dataset_index",
        "caption",
        "generated_matching_distance",
        "generated_text_rank",
        "reference_matching_distance",
        "reference_text_rank",
        "generated_reference_distance_gap",
    ]
    return {key: row[key] for key in keys if key in row}


def _sample_rankings(rows: list[dict[str, Any]], limit: int = 20) -> dict[str, Any]:
    if not rows or "generated_text_rank" not in rows[0]:
        return {"num_samples": len(rows)}
    best_generated = sorted(rows, key=lambda row: (row["generated_text_rank"], row["generated_matching_distance"]))
    worst_generated = sorted(
        rows,
        key=lambda row: (row["generated_text_rank"], row["generated_matching_distance"]),
        reverse=True,
    )
    best_reference = sorted(rows, key=lambda row: (row["reference_text_rank"], row["reference_matching_distance"]))
    worst_reference = sorted(
        rows,
        key=lambda row: (row["reference_text_rank"], row["reference_matching_distance"]),
        reverse=True,
    )
    largest_gap = sorted(rows, key=lambda row: row["generated_reference_distance_gap"], reverse=True)
    return {
        "num_samples": len(rows),
        "best_generated": [_compact_sample(row) for row in best_generated[:limit]],
        "worst_generated": [_compact_sample(row) for row in worst_generated[:limit]],
        "best_reference": [_compact_sample(row) for row in best_reference[:limit]],
        "worst_reference": [_compact_sample(row) for row in worst_reference[:limit]],
        "largest_generated_reference_distance_gap": [_compact_sample(row) for row in largest_gap[:limit]],
    }


def _dataset_metrics(
    *,
    dataset_names: list[str],
    reference_embeddings: np.ndarray,
    generated_embeddings: np.ndarray,
    physical_values: dict[str, np.ndarray],
    text_embeddings: np.ndarray | None,
    distribution_reference_embeddings: np.ndarray | None = None,
    transition_values: dict[str, np.ndarray] | None = None,
    transition_chunk_length: int | None = None,
    transition_num_frames: int | None = None,
) -> dict[str, Any]:
    metrics = {}
    for name, indices in _dataset_indices(dataset_names).items():
        distribution_reference = (
            reference_embeddings[indices]
            if distribution_reference_embeddings is None
            else distribution_reference_embeddings
        )
        item: dict[str, Any] = {
            "num_samples": int(len(indices)),
            "embedding": _embedding_distribution_metrics(distribution_reference, generated_embeddings[indices]),
            "physical": _physical_summary_from_values(physical_values, indices),
        }
        if text_embeddings is not None:
            subset_dataset_names = [dataset_names[int(index)] for index in indices]
            item["text"] = {
                "generated": _text_retrieval_summary(
                    generated_embeddings[indices],
                    text_embeddings[indices],
                    dataset_names=subset_dataset_names,
                    require_full_batches=False,
                    stratified=False,
                ),
                "reference": _text_retrieval_summary(
                    reference_embeddings[indices],
                    text_embeddings[indices],
                    dataset_names=subset_dataset_names,
                    require_full_batches=False,
                    stratified=False,
                ),
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


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path("outputs") / "benchmark" / str(args.exp)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a generation checkpoint on fixed G1 motion samples.",
        allow_abbrev=False,
    )
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--ckpts", nargs="+", default=None)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--data", default="omg_data")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--reference-split",
        "--reference_split",
        dest="reference_split",
        choices=["train", "val", "test"],
        default="test",
        help="Real-motion split used as the distribution baseline for FID/KID; defaults to the test split.",
    )
    parser.add_argument("--num_samples", type=int, default=None, help="Deprecated alias for --num-texts.")
    parser.add_argument("--num-texts", "--num_texts", dest="num_texts", type=int, default=1024)
    parser.add_argument(
        "--num-reference-motions",
        "--num_reference_motions",
        dest="num_reference_motions",
        type=int,
        default=None,
        help="Number of real motions sampled from --reference-split for FID/KID; defaults to --num-texts.",
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument(
        "--cfg_scales",
        nargs="+",
        default=None,
        help="Run a CFG sweep. Use 'default' to use the diffusion config value.",
    )
    parser.add_argument(
        "--motion_key",
        choices=["qpos_36", "motion_features", "body_link_pos_local", "body_pos_local"],
        default="body_pos_local",
    )
    parser.add_argument("--evaluator_checkpoint", required=True)
    parser.add_argument("--samples_path", default=None)
    parser.add_argument("--contact_height_threshold", type=float, default=0.12)
    parser.add_argument("--contact_penetration_tolerance", type=float, default=0.02)
    parser.add_argument("--transition-metrics", "--transition_metrics", dest="transition_metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transition-chunk-length", "--transition_chunk_length", dest="transition_chunk_length", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--enable_text_metrics", action="store_true")
    parser.add_argument("--t5_3b_model", default=DEFAULT_T5_3B_MODEL)
    parser.add_argument("--text_cache_dir", default=DEFAULT_TEXT_CACHE_DIR)
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument(
        "--text-only",
        "--text_only",
        dest="text_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable frame-level audio/human conditions after checkpoint loading; useful for pure text-to-motion evaluation of multimodal checkpoints. Use --no-text-only to keep all conditions enabled.",
    )
    parser.add_argument("--multimodality-repeats", "--multimodality_repeats", dest="multimodality_repeats", type=int, default=1)
    parser.add_argument("--multimodality-pairs", "--multimodality_pairs", dest="multimodality_pairs", type=int, default=10)
    add_tracker_executed_args(parser)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides, e.g. model.text_encoder.model_name=models/t5-base-local",
    )
    args = parser.parse_args(argv)
    if args.ckpts is None:
        if args.ckpt_path is None:
            raise ValueError("Specify --ckpts or --ckpt_path")
        args.ckpts = [args.ckpt_path]
    if args.ckpt_path is None:
        args.ckpt_path = args.ckpts[0]
    if args.num_samples is not None:
        args.num_texts = int(args.num_samples)
    if args.num_reference_motions is None:
        args.num_reference_motions = int(args.num_texts)
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.num_reference_motions < 2:
        raise ValueError("--num_reference_motions must be at least 2 for FID/KID")
    if args.num_frames <= 0:
        raise ValueError("--num_frames must be positive")
    if args.cfg_scale is not None and not math.isfinite(float(args.cfg_scale)):
        raise ValueError("--cfg_scale must be finite")
    if args.transition_chunk_length is not None and args.transition_chunk_length <= 1:
        raise ValueError("--transition_chunk_length must be greater than 1")
    if args.multimodality_repeats <= 0:
        raise ValueError("--multimodality_repeats must be positive")
    if args.multimodality_pairs <= 0:
        raise ValueError("--multimodality_pairs must be positive")
    if args.cfg_scales is None:
        args.resolved_cfg_scales = [None if args.cfg_scale is None else float(args.cfg_scale)]
    else:
        args.resolved_cfg_scales = [_parse_cfg_scale_value(value) for value in args.cfg_scales]
    if args.enable_text_metrics and (args.t5_3b_model is None or args.text_cache_dir is None):
        raise ValueError("--enable_text_metrics requires both --t5_3b_model and --text_cache_dir")
    if not Path(args.evaluator_checkpoint).exists():
        raise FileNotFoundError(
            f"--evaluator_checkpoint does not exist: {args.evaluator_checkpoint}. "
            "Metrics are not silently skipped; provide a trained benchmark evaluator checkpoint."
        )
    validate_tracker_executed_args(args)
    return args


def _apply_text_only_mode(model: Any, args: argparse.Namespace) -> None:
    if not bool(getattr(args, "text_only", False)):
        return
    disabled = []
    if bool(getattr(model, "use_audio", False)):
        model.use_audio = False
        disabled.append("audio")
    if bool(getattr(model, "use_human_motion", False)):
        model.use_human_motion = False
        disabled.append("human_motion")
    if disabled:
        print(f"[INFO] Text-only benchmark: disabled frame-level conditions after checkpoint load: {', '.join(disabled)}")


def _generate_qpos(
    *,
    model: Any,
    datasets: dict[str, Any],
    records: list[SampleRecord],
    args: argparse.Namespace,
    cfg_scale: float | None,
    desc: str = "Benchmark generate",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str], list[int]]:
    generated_chunks = []
    reference_chunks = []
    reference_valid_chunks = []
    fps_chunks = []
    captions: list[str] = []
    dataset_names: list[str] = []
    dataset_indices: list[int] = []

    for start in tqdm(
        range(0, len(records), args.batch_size),
        desc=desc,
        unit="batch",
    ):
        batch_records = records[start : start + args.batch_size]
        items = [datasets[record.dataset][record.index] for record in batch_records]
        batch = motion_collate_fn(items)
        batch_size = len(items)
        batch["has_text"] = torch.ones(batch_size, dtype=torch.bool)
        with torch.inference_mode():
            sample = model.generate(batch, num_frames=int(args.num_frames), cfg_text_scale=cfg_scale)
        generated_chunks.append(sample["qpos_36"].detach().cpu())
        reference_chunks.append(batch["qpos_36"].detach().cpu())
        reference_valid_chunks.append(batch["mask"]["valid"].detach().cpu().bool())
        fps_chunks.append(batch["fps"].detach().cpu().float())
        captions.extend(str(item.get("caption", "")) for item in items)
        dataset_names.extend(record.dataset for record in batch_records)
        dataset_indices.extend(int(record.index) for record in batch_records)

    return (
        torch.cat(generated_chunks, dim=0),
        torch.cat(reference_chunks, dim=0),
        torch.cat(reference_valid_chunks, dim=0),
        torch.cat(fps_chunks, dim=0),
        captions,
        dataset_names,
        dataset_indices,
    )


def _encode_record_motion_embeddings(
    *,
    datasets: dict[str, Any],
    records: list[SampleRecord],
    motion_encoder: MotionEncoder,
    model: Any,
    args: argparse.Namespace,
    device: torch.device,
    desc: str,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    motion_chunks = []
    valid_chunks = []
    captions: list[str] = []
    dataset_names: list[str] = []
    dataset_indices: list[int] = []
    for start in tqdm(range(0, len(records), args.batch_size), desc=desc, unit="batch"):
        batch_records = records[start : start + args.batch_size]
        items = [datasets[record.dataset][record.index] for record in batch_records]
        batch = motion_collate_fn(items)
        qpos = batch["qpos_36"][:, : args.num_frames].detach().cpu()
        valid = batch["mask"]["valid"][:, : args.num_frames].detach().cpu().bool()
        motion = _motion_for_evaluator(qpos, args.motion_key, kinematics=model.representation.kinematics)
        motion_chunks.append(motion)
        valid_chunks.append(valid)
        captions.extend(str(item.get("caption", "")) for item in items)
        dataset_names.extend(record.dataset for record in batch_records)
        dataset_indices.extend(int(record.index) for record in batch_records)
    embeddings = _encode_motion_embeddings(
        motion_encoder,
        torch.cat(motion_chunks, dim=0),
        torch.cat(valid_chunks, dim=0),
        batch_size=args.batch_size,
        device=device,
    )
    return embeddings, captions, dataset_names, dataset_indices


def _run_single_benchmark(
    *,
    output_dir: Path,
    ckpt_path: str,
    cfg_scale: float | None,
    datasets: dict[str, Any],
    reference_datasets: dict[str, Any],
    records: list[SampleRecord],
    reference_records: list[SampleRecord],
    model: Any,
    motion_encoder: MotionEncoder,
    evaluator_checkpoint: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    text_embeddings: np.ndarray | None,
) -> BenchmarkResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_sample_records(output_dir / "samples.jsonl", records, datasets)
    _write_sample_records(output_dir / "reference_samples.jsonl", reference_records, reference_datasets)
    print(
        f"[INFO] Text benchmark run: num_samples={len(records)} device={device} "
        f"output_dir={output_dir.resolve()} cfg_scale={_cfg_scale_json(cfg_scale)}"
    )

    (
        generated_qpos,
        reference_qpos,
        reference_valid,
        fps,
        captions,
        dataset_names,
        dataset_indices,
    ) = _generate_qpos(model=model, datasets=datasets, records=records, args=args, cfg_scale=cfg_scale)

    if generated_qpos.shape[1] < reference_qpos.shape[1]:
        raise ValueError(
            f"Generated motion has fewer frames than reference: {generated_qpos.shape[1]} < {reference_qpos.shape[1]}"
        )
    embedding_frames = int(reference_qpos.shape[1])
    generated_for_embedding = generated_qpos[:, :embedding_frames]
    generated_valid = torch.ones(generated_for_embedding.shape[:2], dtype=torch.bool)
    transition_values, transition_metrics, generated_body_pos = _compute_transition_metrics(
        model=model,
        generated_qpos=generated_qpos,
        reference_qpos=reference_qpos,
        args=args,
        device=device,
    )

    print("[INFO] Saving motion npz bundles…")
    np.savez_compressed(
        output_dir / "generated_qpos.npz",
        qpos_36=generated_qpos.numpy().astype(np.float32, copy=False),
        fps=fps.numpy().astype(np.float32, copy=False),
        captions=np.asarray(captions, dtype=np.str_),
        dataset=np.asarray(dataset_names, dtype=np.str_),
        dataset_index=np.asarray(dataset_indices, dtype=np.int32),
        ckpt_path=np.asarray([str(Path(ckpt_path).resolve())], dtype=np.str_),
        **({"body_pos_w": generated_body_pos.numpy().astype(np.float32, copy=False)} if generated_body_pos is not None else {}),
    )
    np.savez_compressed(
        output_dir / "reference_qpos.npz",
        qpos_36=reference_qpos.numpy().astype(np.float32, copy=False),
        valid=reference_valid.numpy().astype(np.bool_, copy=False),
        fps=fps.numpy().astype(np.float32, copy=False),
        captions=np.asarray(captions, dtype=np.str_),
        dataset=np.asarray(dataset_names, dtype=np.str_),
        dataset_index=np.asarray(dataset_indices, dtype=np.int32),
    )

    print("[INFO] Motion embeddings, physical metrics, optional text metrics…")
    reference_motion_for_embedding = _motion_for_evaluator(
        reference_qpos,
        args.motion_key,
        kinematics=model.representation.kinematics,
    )
    generated_motion_for_embedding = _motion_for_evaluator(
        generated_for_embedding,
        args.motion_key,
        kinematics=model.representation.kinematics,
    )
    reference_embeddings = _encode_motion_embeddings(
        motion_encoder,
        reference_motion_for_embedding,
        reference_valid,
        batch_size=args.batch_size,
        device=device,
    )
    generated_embeddings = _encode_motion_embeddings(
        motion_encoder,
        generated_motion_for_embedding,
        generated_valid,
        batch_size=args.batch_size,
        device=device,
    )
    distribution_reference_embeddings, _, distribution_dataset_names, distribution_dataset_indices = _encode_record_motion_embeddings(
        datasets=reference_datasets,
        records=reference_records,
        motion_encoder=motion_encoder,
        model=model,
        args=args,
        device=device,
        desc="Distribution-reference motion embeddings",
    )
    physical_values = _physical_values(
        generated_qpos,
        fps,
        representation=model.representation,
        device=device,
        contact_height_threshold=args.contact_height_threshold,
        contact_penetration_tolerance=args.contact_penetration_tolerance,
    )
    multimodality_metrics = _compute_multimodality_metrics(
        model=model,
        datasets=datasets,
        records=records,
        motion_encoder=motion_encoder,
        args=args,
        cfg_scale=cfg_scale,
        embedding_frames=embedding_frames,
        device=device,
    )
    if args.enable_text_metrics and text_embeddings is None:
        text_embeddings = _encode_text_embeddings(
            evaluator_checkpoint,
            captions,
            model_name=args.t5_3b_model,
            cache_dir=args.text_cache_dir,
            batch_size=args.text_batch_size,
            device=device,
        )
    if text_embeddings is not None and text_embeddings.shape[0] != generated_embeddings.shape[0]:
        raise ValueError(
            f"Text embedding count mismatch: {text_embeddings.shape[0]} != {generated_embeddings.shape[0]}"
        )
    real_motion_metrics = _real_motion_metrics(distribution_reference_embeddings, reference_embeddings, text_embeddings)

    embedding_metrics = _embedding_distribution_metrics(distribution_reference_embeddings, generated_embeddings)
    embedding_metrics.update(
        {
            "embedding_frames": embedding_frames,
            "motion_key": args.motion_key,
            "evaluator_checkpoint": str(Path(args.evaluator_checkpoint).resolve()),
            "multimodality": multimodality_metrics,
            "distribution_reference": "real_motion",
            "distribution_reference_split": args.reference_split,
            "distribution_reference_num_samples": int(distribution_reference_embeddings.shape[0]),
        }
    )
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
        embedding_metrics.update(
            {
                "matching_score_generated": generated_text["matching_score"],
                "matching_score_reference": reference_text["matching_score"],
                "r_precision_generated": generated_text["r_precision"],
                "r_precision_reference": reference_text["r_precision"],
                "text_generated": generated_text,
                "text_reference": reference_text,
                "text_encoder_model": str(args.t5_3b_model),
                "text_cache_dir": str(args.text_cache_dir),
                "text_retrieval_batch_size": TEXT_RETRIEVAL_BATCH_SIZE,
            }
        )

    physical_metrics = _physical_summary_from_values(physical_values)
    physical_metrics["frames"] = int(generated_qpos.shape[1])
    physical_metrics["contact_height_threshold"] = float(args.contact_height_threshold)
    physical_metrics["contact_penetration_tolerance"] = float(args.contact_penetration_tolerance)

    dataset_metrics = _dataset_metrics(
        dataset_names=dataset_names,
        reference_embeddings=reference_embeddings,
        generated_embeddings=generated_embeddings,
        distribution_reference_embeddings=distribution_reference_embeddings,
        physical_values=physical_values,
        text_embeddings=text_embeddings,
        transition_values=transition_values,
        transition_chunk_length=transition_metrics.get("chunk_length"),
        transition_num_frames=int(generated_qpos.shape[1]),
    )
    sample_rows = _sample_metric_rows(
        captions=captions,
        dataset_names=dataset_names,
        dataset_indices=dataset_indices,
        physical_values=physical_values,
        generated_embeddings=generated_embeddings,
        reference_embeddings=reference_embeddings,
        text_embeddings=text_embeddings,
        transition_values=transition_values,
    )
    sample_rankings = _sample_rankings(sample_rows)
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

    print("[INFO] Saving benchmark metrics and embeddings…")
    _finite_metrics(embedding_metrics)
    _finite_metrics(real_motion_metrics)
    _finite_metrics(physical_metrics)
    _finite_metrics(transition_metrics)
    _finite_metrics(dataset_metrics)
    _finite_metrics(sample_rankings)
    _finite_metrics(tracker_executed)
    _save_json(output_dir / "embedding_metrics.json", embedding_metrics)
    _save_json(output_dir / "real_motion_metrics.json", real_motion_metrics)
    _save_json(output_dir / "physical_metrics.json", physical_metrics)
    _save_json(output_dir / "transition_metrics.json", transition_metrics)
    _save_json(output_dir / "dataset_metrics.json", dataset_metrics)
    _save_json(output_dir / "sample_rankings.json", sample_rankings)
    _write_jsonl(output_dir / "sample_metrics.jsonl", sample_rows)

    embeddings_payload: dict[str, Any] = {
        "reference_embeddings": reference_embeddings.astype(np.float32, copy=False),
        "distribution_reference_embeddings": distribution_reference_embeddings.astype(np.float32, copy=False),
        "generated_embeddings": generated_embeddings.astype(np.float32, copy=False),
    }
    if text_embeddings is not None:
        embeddings_payload["text_embeddings"] = text_embeddings.astype(np.float32, copy=False)
    np.savez_compressed(output_dir / "embeddings.npz", **embeddings_payload)

    benchmark = {
        "ckpt_path": str(Path(ckpt_path).resolve()),
        "exp": args.exp,
        "split": args.split,
        "num_samples": len(records),
        "num_frames": int(args.num_frames),
        "reference_frames": int(reference_qpos.shape[1]),
        "embedding_frames": embedding_frames,
        "reference_split": args.reference_split,
        "reference_num_samples": int(distribution_reference_embeddings.shape[0]),
        "reference_datasets": sorted(set(distribution_dataset_names)),
        "reference_dataset_indices": [int(index) for index in distribution_dataset_indices],
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "cfg_scale": _cfg_scale_json(cfg_scale),
        "samples_path": str((output_dir / "samples.jsonl").resolve()),
        "reference_samples_path": str((output_dir / "reference_samples.jsonl").resolve()),
        "generated_qpos": str((output_dir / "generated_qpos.npz").resolve()),
        "reference_qpos": str((output_dir / "reference_qpos.npz").resolve()),
        "embedding_metrics": embedding_metrics,
        "real_motion_metrics": real_motion_metrics,
        "physical_metrics": physical_metrics,
        "transition_metrics": transition_metrics,
        "tracker_executed": tracker_executed,
        "dataset_metrics": dataset_metrics,
        "dataset_metrics_path": str((output_dir / "dataset_metrics.json").resolve()),
        "sample_metrics_path": str((output_dir / "sample_metrics.jsonl").resolve()),
        "sample_rankings_path": str((output_dir / "sample_rankings.json").resolve()),
    }
    _save_json(output_dir / "benchmark.json", benchmark)
    return BenchmarkResult(benchmark=benchmark, text_embeddings=text_embeddings)


def _summary_row(benchmark: dict[str, Any], label: str | None = None) -> dict[str, Any]:
    embedding = benchmark["embedding_metrics"]
    physical = benchmark["physical_metrics"]
    row: dict[str, Any] = {
        "ckpt": label or Path(benchmark["ckpt_path"]).name,
        "FID(seq)": embedding.get("motion_fid"),
        "KID(seq)": embedding.get("motion_kid"),
        "Diversity": embedding.get("diversity_generated"),
        "MultiModality": embedding.get("multimodality") or "-",
        "contact_sliding": physical.get("contact_sliding_speed"),
        "foot_ground_error": physical.get("foot_ground_error"),
        "body_jerk": physical.get("body_jerk_mean"),
    }
    tracker = benchmark.get("tracker_executed", {})
    tracker_metrics = tracker.get("metrics", {}) if tracker.get("enabled") else {}
    row["Tracker g-MPJPE"] = tracker_metrics.get("g_mpjpe")
    row["Tracker MPJPE"] = tracker_metrics.get("mpjpe")
    row["Tracker E_vel"] = tracker_metrics.get("e_vel")
    row["Tracker E_acc"] = tracker_metrics.get("e_acc")
    transition = benchmark.get("transition_metrics", {})
    if transition.get("enabled"):
        row["PJ"] = transition.get("pj")
        row["AUJ"] = transition.get("auj")
    else:
        row["PJ"] = None
        row["AUJ"] = None
    if "matching_score_generated" in embedding:
        row["Matching"] = embedding.get("matching_score_generated")
        row["MultiModal Dist"] = embedding.get("text_generated", {}).get("matching_distance")
    else:
        row["Matching"] = None
        row["MultiModal Dist"] = None
    r_precision_values = embedding.get("r_precision_generated")
    if r_precision_values is not None:
        for idx, value in enumerate(r_precision_values[:3], start=1):
            row[f"R@{idx}"] = value
    else:
        row["R@1"] = None
        row["R@2"] = None
        row["R@3"] = None
    return row


def _real_summary_row(benchmark: dict[str, Any]) -> dict[str, Any]:
    real = benchmark["real_motion_metrics"]
    row: dict[str, Any] = {
        "ckpt": "Real motions",
        "FID(seq)": real.get("motion_fid_split_half"),
        "KID(seq)": real.get("motion_kid_split_half"),
        "Diversity": real.get("diversity_real"),
        "MultiModality": real.get("multimodality", "-"),
        "Matching": real.get("matching_score_real"),
        "MultiModal Dist": real.get("matching_distance_real"),
        "contact_sliding": None,
        "foot_ground_error": None,
        "body_jerk": None,
        "Tracker g-MPJPE": None,
        "Tracker MPJPE": None,
        "Tracker E_vel": None,
        "Tracker E_acc": None,
        "PJ": None,
        "AUJ": None,
    }
    r_precision_values = real.get("r_precision_real")
    if r_precision_values is not None:
        for idx, value in enumerate(r_precision_values[:3], start=1):
            row[f"R@{idx}"] = value
    else:
        row["R@1"] = None
        row["R@2"] = None
        row["R@3"] = None
    return row


def _write_generation_summary(
    output_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    datasets: dict[str, Any],
) -> None:
    metric_directions = {
        "FID(seq)": "min",
        "KID(seq)": "min",
        "R@1": "max",
        "R@2": "max",
        "R@3": "max",
        "Diversity": "max",
        "Matching": "min",
        "MultiModal Dist": "min",
        "MultiModality": "max",
        "contact_sliding": "min",
        "foot_ground_error": "min",
        "body_jerk": "min",
        "Tracker g-MPJPE": "min",
        "Tracker MPJPE": "min",
        "Tracker E_vel": "min",
        "Tracker E_acc": "min",
        "PJ": "min",
        "AUJ": "min",
    }
    markdown = generation_benchmark_markdown(
        rows=rows,
        metric_directions=metric_directions,
        metadata={
            "ckpts": [str(Path(path).resolve()) for path in args.ckpts],
            "datasets": list(datasets.keys()),
            "num_texts": args.num_texts,
            "reference_split": args.reference_split,
            "num_reference_motions": args.num_reference_motions,
            "batch_size": args.batch_size,
            "split": args.split,
            "evaluator_checkpoint": str(Path(args.evaluator_checkpoint).resolve()),
            "enable_text_metrics": bool(args.enable_text_metrics),
            "t5_3b_model": str(Path(args.t5_3b_model).resolve()) if args.t5_3b_model else None,
            "text_cache_dir": str(Path(args.text_cache_dir).resolve()) if args.text_cache_dir else None,
            "text_batch_size": args.text_batch_size,
            "metric_definition": (
                "FID(seq) and KID(seq) compare sequence-level evaluator embeddings against real motions "
                "sampled from --reference-split, which defaults to the training split. "
                "The Real motions row compares eval real motions against this reference baseline; "
                "model rows compare generated motions against the same reference baseline."
            ),
            "text_metrics": (
                "R-precision is computed with fixed batch32 retrieval candidate sets. "
                "Retrieval batches are stratified by dataset when dataset metadata is available. "
                "Matching and MultiModal Dist are paired embedding distances. Text metrics are computed when "
                "--enable_text_metrics is provided; --t5_3b_model and --text_cache_dir default to the paths above."
            ),
            "multimodality": (
                "Computed only when --multimodality_repeats > 1 by sampling each text multiple times "
                "and averaging same-text motion embedding distances."
            ),
            "transition_metrics": "TextOp-style PJ/AUJ on body/link jerk over transition windows centered at autoregressive chunk seams.",
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = _device(args.device)
    output_dir = _output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=str(_config_dir()), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"exp={args.exp}",
                f"data={args.data}",
                "logger=none",
                "trainer=1gpu",
                *args.overrides,
            ],
        )

    sample_path = _sample_file_path(output_dir, args.samples_path)
    if args.samples_path is not None and not sample_path.exists():
        raise FileNotFoundError(f"--samples_path does not exist: {sample_path}")
    records = None
    if sample_path.exists():
        records = _load_sample_records(sample_path)
        print(f"[INFO] Loaded {len(records)} sample records from {sample_path.resolve()}")
    dataset_include = args.datasets if args.datasets is not None else (
        _dataset_filter_tokens_from_records(records) if records is not None else None
    )
    datasets = _build_datasets(cfg, args.split, include=dataset_include)
    reference_datasets = _build_datasets(cfg, args.reference_split, include=dataset_include)
    if records is None:
        records = _select_sample_records(datasets, args.num_texts, args.seed)
    reference_records = _select_sample_records(
        reference_datasets,
        int(args.num_reference_motions),
        int(args.seed),
    )
    _validate_sample_records(records, datasets)
    _validate_sample_records(reference_records, reference_datasets)
    if args.enable_text_metrics and len(records) % TEXT_RETRIEVAL_BATCH_SIZE != 0:
        raise ValueError(
            f"--enable_text_metrics uses fixed batch{TEXT_RETRIEVAL_BATCH_SIZE} R-Precision; "
            f"selected {len(records)} samples, which is not divisible by {TEXT_RETRIEVAL_BATCH_SIZE}"
        )
    _write_sample_records(output_dir / "samples.jsonl", records, datasets)
    print(
        f"[INFO] Benchmark setup: num_records={len(records)} device={device} "
        f"split={args.split} datasets={list(datasets.keys())} "
        f"reference_split={args.reference_split} reference_records={len(reference_records)} "
        f"reference_datasets={list(reference_datasets.keys())} output_dir={output_dir.resolve()}"
    )

    motion_encoder, evaluator_checkpoint = _load_evaluator_motion_encoder(
        args.evaluator_checkpoint,
        args.motion_key,
        device,
    )

    cfg_scales = list(args.resolved_cfg_scales)
    shared_text_embeddings: np.ndarray | None = None
    benchmark_rows = []
    real_row: dict[str, Any] | None = None
    real_motion_metrics: dict[str, Any] | None = None
    all_runs = {}
    if len(args.ckpts) == 1 and len(cfg_scales) == 1:
        ckpt_path = args.ckpts[0]
        print(f"[INFO] Running benchmark ckpt={ckpt_path} num_texts={len(records)} batch_size={args.batch_size} output_dir={output_dir}")
        model = _load_model(cfg, ckpt_path, device)
        _apply_text_only_mode(model, args)
        result = _run_single_benchmark(
            output_dir=output_dir,
            ckpt_path=ckpt_path,
            cfg_scale=cfg_scales[0],
            datasets=datasets,
            reference_datasets=reference_datasets,
            records=records,
            reference_records=reference_records,
            model=model,
            motion_encoder=motion_encoder,
            evaluator_checkpoint=evaluator_checkpoint,
            args=args,
            device=device,
            text_embeddings=shared_text_embeddings,
        )
        benchmark_rows.append(_real_summary_row(result.benchmark))
        benchmark_rows.append(_summary_row(result.benchmark))
        _save_json(
            output_dir / "metrics.json",
            {
                "ckpts": [str(Path(ckpt_path).resolve())],
                "exp": args.exp,
                "split": args.split,
                "num_texts": len(records),
                "reference_split": args.reference_split,
                "num_reference_motions": len(reference_records),
                "batch_size": int(args.batch_size),
                "datasets": list(datasets.keys()),
                "reference_datasets": list(reference_datasets.keys()),
                "samples_path": str((output_dir / "samples.jsonl").resolve()),
                "reference_samples_path": str((output_dir / "reference_samples.jsonl").resolve()),
                "benchmark": result.benchmark,
            },
        )
        _write_generation_summary(output_dir, benchmark_rows, args, datasets)
        print(json.dumps(result.benchmark, indent=2, sort_keys=True))
        return

    for ckpt_path in args.ckpts:
        model = _load_model(cfg, ckpt_path, device)
        _apply_text_only_mode(model, args)
        ckpt_label = Path(ckpt_path).stem
        for cfg_scale in cfg_scales:
            label = ckpt_label if len(cfg_scales) == 1 else f"{ckpt_label}_{_cfg_output_name(cfg_scale)}"
            run_dir = output_dir / label
            print(f"[INFO] Running benchmark ckpt={ckpt_path} cfg_scale={cfg_scale} output_dir={run_dir}")
            result = _run_single_benchmark(
                output_dir=run_dir,
                ckpt_path=ckpt_path,
                cfg_scale=cfg_scale,
                datasets=datasets,
                reference_datasets=reference_datasets,
                records=records,
                reference_records=reference_records,
                model=model,
                motion_encoder=motion_encoder,
                evaluator_checkpoint=evaluator_checkpoint,
                args=args,
                device=device,
                text_embeddings=shared_text_embeddings,
            )
            if shared_text_embeddings is None:
                shared_text_embeddings = result.text_embeddings
            if real_row is None:
                real_row = _real_summary_row(result.benchmark)
                real_motion_metrics = result.benchmark["real_motion_metrics"]
            all_runs[label] = {
                "ckpt_path": str(Path(ckpt_path).resolve()),
                "cfg_scale": _cfg_scale_json(cfg_scale),
                "output_dir": str(run_dir.resolve()),
                "benchmark_json": str((run_dir / "benchmark.json").resolve()),
                "embedding_metrics": result.benchmark["embedding_metrics"],
                "physical_metrics": result.benchmark["physical_metrics"],
                "transition_metrics": result.benchmark["transition_metrics"],
                "tracker_executed": result.benchmark["tracker_executed"],
            }
            benchmark_rows.append(_summary_row(result.benchmark, label=label))
    if real_row is not None:
        benchmark_rows = [real_row, *benchmark_rows]
    summary_payload = {
        "ckpts": [str(Path(path).resolve()) for path in args.ckpts],
        "exp": args.exp,
        "split": args.split,
        "num_texts": len(records),
        "reference_split": args.reference_split,
        "num_reference_motions": len(reference_records),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "datasets": list(datasets.keys()),
        "reference_datasets": list(reference_datasets.keys()),
        "samples_path": str((output_dir / "samples.jsonl").resolve()),
        "reference_samples_path": str((output_dir / "reference_samples.jsonl").resolve()),
        "real_motion_metrics": real_motion_metrics,
        "runs": all_runs,
    }
    _finite_metrics(summary_payload)
    _save_json(output_dir / "metrics.json", summary_payload)
    _write_generation_summary(output_dir, benchmark_rows, args, datasets)
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
