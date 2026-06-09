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
    conditioned_dataset_metrics,
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
    _embedding_distribution_metrics,
    _encode_motion_embeddings,
    _finite_metrics,
    _load_evaluator_motion_encoder,
    _motion_for_evaluator,
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
    e_acc,
    e_vel,
    g_mpjpe,
    mpjpe,
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

END_EFFECTOR_LINKS = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a human-reference-conditioned generation checkpoint.")
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
    parser.add_argument("--cfg_scale", "--cfg_human_scale", dest="cfg_human_scale", type=float, default=None)
    parser.add_argument("--cfg_scales", "--cfg_human_scales", dest="cfg_human_scales", nargs="+", default=None)
    parser.add_argument("--motion_key", choices=["qpos_36", "body_pos_local", "body_link_pos_local"], default="body_pos_local")
    parser.add_argument("--evaluator_checkpoint", default=None)
    parser.add_argument("--samples_path", default=None)
    parser.add_argument("--contact_height_threshold", type=float, default=0.12)
    parser.add_argument("--contact_penetration_tolerance", type=float, default=0.02)
    parser.add_argument("--transition-metrics", "--transition_metrics", dest="transition_metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transition-chunk-length", "--transition_chunk_length", dest="transition_chunk_length", type=int, default=None)
    parser.add_argument("--root_index", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    add_tracker_executed_args(parser)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides, e.g. model.use_human_motion=true model.human_motion_dim=66",
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
    if args.num_frames <= 2:
        raise ValueError("--num_frames must be greater than 2 for velocity and acceleration metrics")
    if args.cfg_human_scale is not None and not math.isfinite(float(args.cfg_human_scale)):
        raise ValueError("--cfg_human_scale must be finite")
    if args.cfg_human_scales is None:
        args.resolved_cfg_scales = [None if args.cfg_human_scale is None else float(args.cfg_human_scale)]
    else:
        args.resolved_cfg_scales = [_parse_cfg_scale_value(value) for value in args.cfg_human_scales]
    if args.transition_chunk_length is not None and args.transition_chunk_length <= 1:
        raise ValueError("--transition_chunk_length must be greater than 1")
    if args.evaluator_checkpoint is not None and not Path(args.evaluator_checkpoint).exists():
        raise FileNotFoundError(f"--evaluator_checkpoint does not exist: {args.evaluator_checkpoint}")
    validate_tracker_executed_args(args)
    return args


def _generate_humanref_qpos(
    *,
    model: Any,
    datasets: dict[str, Any],
    records: list[SampleRecord],
    args: argparse.Namespace,
    cfg_human_scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str], list[int]]:
    generated_chunks = []
    reference_chunks = []
    reference_body_chunks = []
    valid_chunks = []
    fps_chunks = []
    human_motion_chunks = []
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
                cfg_audio_scale=0.0,
                cfg_human_scale=cfg_human_scale,
            )
        generated_chunks.append(sample["qpos_36"].detach().cpu())
        reference_chunks.append(batch["qpos_36"][:, : args.num_frames].detach().cpu())
        reference_body_chunks.append(batch["body_pos_w"][:, : args.num_frames].detach().cpu())
        valid_chunks.append(finite_valid(batch["mask"]["valid"], args.num_frames))
        fps_chunks.append(batch["fps"].detach().cpu().float())
        human_motion_chunks.append(batch["human_motion"][:, : args.num_frames].detach().cpu())
        captions.extend(str(item.get("caption", "")) for item in items)
        dataset_names.extend(record.dataset for record in batch_records)
        dataset_indices.extend(int(record.index) for record in batch_records)

    return (
        torch.cat(generated_chunks, dim=0),
        torch.cat(reference_chunks, dim=0),
        torch.cat(reference_body_chunks, dim=0),
        torch.cat(valid_chunks, dim=0),
        torch.cat(fps_chunks, dim=0),
        torch.cat(human_motion_chunks, dim=0),
        captions,
        dataset_names,
        dataset_indices,
    )


def _tracking_values(
    *,
    generated_body_pos: torch.Tensor,
    reference_body_pos: torch.Tensor,
    root_index: int,
    end_effector_indices: tuple[int, ...],
) -> dict[str, np.ndarray]:
    generated_np = generated_body_pos.numpy()
    reference_np = reference_body_pos.numpy()
    values = {"g_mpjpe": [], "mpjpe": [], "ee_error": [], "e_vel": [], "e_acc": []}
    ee_indices = np.asarray(end_effector_indices, dtype=np.int64)
    for idx in tqdm(range(generated_np.shape[0]), desc="Tracking metrics", unit="sample", leave=False):
        values["g_mpjpe"].append(g_mpjpe(generated_np[idx], reference_np[idx]))
        values["mpjpe"].append(mpjpe(generated_np[idx], reference_np[idx], root_index=int(root_index)))
        ee_delta = generated_np[idx, :, ee_indices] - reference_np[idx, :, ee_indices]
        values["ee_error"].append(float(np.linalg.norm(ee_delta, axis=-1).mean() * 1000.0))
        values["e_vel"].append(e_vel(generated_np[idx], reference_np[idx]))
        values["e_acc"].append(e_acc(generated_np[idx], reference_np[idx]))
    return {key: np.asarray(metric_values, dtype=np.float64) for key, metric_values in values.items()}


def _end_effector_indices(model: Any) -> tuple[int, ...]:
    name_to_index = model.representation.kinematics.body_name_to_index
    missing = [name for name in END_EFFECTOR_LINKS if name not in name_to_index]
    if missing:
        raise KeyError(f"Missing G1 end-effector links in kinematics: {missing}")
    return tuple(int(name_to_index[name]) for name in END_EFFECTOR_LINKS)


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


def _sample_rows(
    *,
    captions: list[str],
    dataset_names: list[str],
    dataset_indices: list[int],
    tracking_values: dict[str, np.ndarray],
    physical_values: dict[str, np.ndarray],
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
        for key, values in tracking_values.items():
            row[key] = float(values[idx])
        for key, values in physical_values.items():
            row[key] = float(values[idx])
        if transition_values:
            for key, values in transition_values.items():
                array = np.asarray(values)
                row[key] = float(array if array.ndim == 0 else array[idx])
        rows.append(row)
    return rows


def _summary_row(benchmark: dict[str, Any], label: str | None = None) -> dict[str, Any]:
    embedding = benchmark["embedding_metrics"]
    tracking = benchmark["tracking_metrics"]
    physical = benchmark["physical_metrics"]
    row = {
        "ckpt": label or Path(benchmark["ckpt_path"]).name,
        "g-MPJPE": tracking["g_mpjpe"],
        "MPJPE": tracking["mpjpe"],
        "EE error": tracking["ee_error"],
        "E_vel": tracking["e_vel"],
        "E_acc": tracking["e_acc"],
        "FID(seq)": embedding.get("motion_fid"),
        "KID(seq)": embedding.get("motion_kid"),
        "Diversity": embedding.get("diversity_generated"),
        "contact_sliding": physical.get("contact_sliding_speed"),
        "foot_ground_error": physical.get("foot_ground_error"),
        "body_jerk": physical.get("body_jerk_mean"),
    }
    transition = benchmark.get("transition_metrics", {})
    if transition.get("enabled"):
        row["PJ"] = transition.get("pj")
        row["AUJ"] = transition.get("auj")
    else:
        row["PJ"] = None
        row["AUJ"] = None
    return row


def _write_humanref_summary(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace, datasets: dict[str, Any]) -> None:
    markdown = generation_benchmark_markdown(
        rows=rows,
        metric_directions={
            "g-MPJPE": "min",
            "MPJPE": "min",
            "EE error": "min",
            "E_vel": "min",
            "E_acc": "min",
            "FID(seq)": "min",
            "KID(seq)": "min",
            "Diversity": "max",
            "contact_sliding": "min",
            "foot_ground_error": "min",
            "body_jerk": "min",
            "PJ": "min",
            "AUJ": "min",
        },
        metadata={
            "benchmark": "humanref",
            "ckpts": [str(Path(path).resolve()) for path in args.ckpts],
            "datasets": list(datasets.keys()),
            "num_samples": args.num_samples,
            "num_frames": args.num_frames,
            "batch_size": args.batch_size,
            "split": args.split,
            "data": args.data,
            "evaluator_checkpoint": None if args.evaluator_checkpoint is None else str(Path(args.evaluator_checkpoint).resolve()),
            "root_index": args.root_index,
            "transition_metrics": "TextOp-style PJ/AUJ on body/link jerk over transition windows centered at autoregressive chunk seams.",
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")


def _run_single_benchmark(
    *,
    output_dir: Path,
    ckpt_path: str,
    cfg_human_scale: float | None,
    datasets: dict[str, Any],
    records: list[SampleRecord],
    model: Any,
    motion_encoder: Any | None,
    args: argparse.Namespace,
    device: torch.device,
) -> BenchmarkResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_sample_records(output_dir / "samples.jsonl", records, datasets)
    print(
        f"[INFO] Humanref benchmark run: num_samples={len(records)} device={device} "
        f"output_dir={output_dir.resolve()}"
    )

    (
        generated_qpos,
        reference_qpos,
        reference_body_pos,
        reference_valid,
        fps,
        human_motion,
        captions,
        dataset_names,
        dataset_indices,
    ) = _generate_humanref_qpos(model=model, datasets=datasets, records=records, args=args, cfg_human_scale=cfg_human_scale)

    print("[INFO] Forward kinematics on generated motion, motion embeddings, physical and tracking metrics…")
    generated_body_pos = qpos_to_body_positions(model, generated_qpos, batch_size=args.batch_size, device=device)
    generated_valid = torch.ones(generated_qpos.shape[:2], dtype=torch.bool)
    reference_embeddings = None
    generated_embeddings = None
    if motion_encoder is not None:
        reference_motion = _motion_for_evaluator(reference_qpos, args.motion_key, kinematics=model.representation.kinematics)
        generated_motion = _motion_for_evaluator(generated_qpos, args.motion_key, kinematics=model.representation.kinematics)
        reference_embeddings = _encode_motion_embeddings(
            motion_encoder,
            reference_motion,
            reference_valid,
            batch_size=args.batch_size,
            device=device,
        )
        generated_embeddings = _encode_motion_embeddings(
            motion_encoder,
            generated_motion,
            generated_valid,
            batch_size=args.batch_size,
            device=device,
        )
    physical_values = _physical_values(
        generated_qpos,
        fps,
        representation=model.representation,
        device=device,
        contact_height_threshold=args.contact_height_threshold,
        contact_penetration_tolerance=args.contact_penetration_tolerance,
    )
    transition_values, transition_metrics = _compute_transition_metrics(
        model=model,
        generated_qpos=generated_qpos,
        reference_qpos=reference_qpos,
        generated_body_pos=generated_body_pos,
        reference_body_pos=reference_body_pos,
        args=args,
    )
    tracking_values = _tracking_values(
        generated_body_pos=generated_body_pos,
        reference_body_pos=reference_body_pos,
        root_index=args.root_index,
        end_effector_indices=_end_effector_indices(model),
    )

    embedding_metrics = _embedding_distribution_metrics(reference_embeddings, generated_embeddings)
    embedding_metrics.update(
        {
            "embedding_frames": int(generated_qpos.shape[1]),
            "motion_key": args.motion_key,
            "evaluator_checkpoint": None if args.evaluator_checkpoint is None else str(Path(args.evaluator_checkpoint).resolve()),
        }
    )
    tracking_metrics = summarize_values(tracking_values)
    tracking_metrics.update({"frames": int(generated_qpos.shape[1]), "root_index": int(args.root_index)})
    physical_metrics = _physical_summary_from_values(physical_values)
    physical_metrics["frames"] = int(generated_qpos.shape[1])
    physical_metrics["contact_height_threshold"] = float(args.contact_height_threshold)
    physical_metrics["contact_penetration_tolerance"] = float(args.contact_penetration_tolerance)
    if reference_embeddings is not None and generated_embeddings is not None:
        dataset_metrics = conditioned_dataset_metrics(
            dataset_names=dataset_names,
            reference_embeddings=reference_embeddings,
            generated_embeddings=generated_embeddings,
            physical_values=physical_values,
            condition_values=tracking_values,
            condition_key="tracking",
        )
    else:
        dataset_metrics = {}
        by_name: dict[str, list[int]] = {}
        for idx, name in enumerate(dataset_names):
            by_name.setdefault(str(name), []).append(idx)
        for name, indices in sorted(by_name.items()):
            idx = np.asarray(indices, dtype=np.int64)
            dataset_metrics[name] = {
                "num_samples": int(len(indices)),
                "physical": _physical_summary_from_values({key: np.asarray(values)[idx] for key, values in physical_values.items()}),
                "tracking": summarize_values({key: np.asarray(values)[idx] for key, values in tracking_values.items()}),
            }
    if transition_values:
        for name, indices in _dataset_indices(dataset_names).items():
            dataset_metrics[name]["transition"] = transition_metric_summary(
                transition_values,
                chunk_length=int(transition_metrics["chunk_length"]),
                num_frames=int(generated_qpos.shape[1]),
                indices=indices,
            )
    sample_rows = _sample_rows(
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

    print("[INFO] Saving metrics and motion artifacts…")
    _finite_metrics(embedding_metrics)
    _finite_metrics(tracking_metrics)
    _finite_metrics(physical_metrics)
    _finite_metrics(transition_metrics)
    _finite_metrics(dataset_metrics)
    _finite_metrics(tracker_executed)
    _save_json(output_dir / "embedding_metrics.json", embedding_metrics)
    _save_json(output_dir / "tracking_metrics.json", tracking_metrics)
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
        output_dir / "human_motion.npz",
        human_motion=human_motion.numpy().astype(np.float32, copy=False),
        fps=fps.numpy().astype(np.float32, copy=False),
        dataset=np.asarray(dataset_names, dtype=np.str_),
        dataset_index=np.asarray(dataset_indices, dtype=np.int32),
    )
    if reference_embeddings is not None and generated_embeddings is not None:
        np.savez_compressed(
            output_dir / "embeddings.npz",
            reference_embeddings=reference_embeddings.astype(np.float32, copy=False),
            generated_embeddings=generated_embeddings.astype(np.float32, copy=False),
        )

    benchmark = {
        "benchmark": "humanref",
        "ckpt_path": str(Path(ckpt_path).resolve()),
        "exp": args.exp,
        "split": args.split,
        "num_samples": len(records),
        "num_frames": int(args.num_frames),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "cfg_human_scale": _cfg_scale_json(cfg_human_scale),
        "samples_path": str((output_dir / "samples.jsonl").resolve()),
        "generated_qpos": str((output_dir / "generated_qpos.npz").resolve()),
        "reference_qpos": str((output_dir / "reference_qpos.npz").resolve()),
        "human_motion": str((output_dir / "human_motion.npz").resolve()),
        "embedding_metrics": embedding_metrics,
        "tracking_metrics": tracking_metrics,
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
    root_output_dir = conditioned_output_dir(args, "humanref")
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
            tensor_key="human_motion",
            mask_key="has_human_motion",
            label="human-reference",
        )
    _validate_sample_records(records, datasets)
    _write_sample_records(root_output_dir / "samples.jsonl", records, datasets)
    print(
        f"[INFO] Benchmark setup: num_records={len(records)} device={device} "
        f"split={args.split} datasets={list(datasets.keys())} output_dir={root_output_dir.resolve()}"
    )

    motion_encoder = None
    if args.evaluator_checkpoint is not None:
        motion_encoder, _ = _load_evaluator_motion_encoder(args.evaluator_checkpoint, args.motion_key, device)
    rows = []
    runs = {}
    for ckpt_path in args.ckpts:
        model = load_condition_model(cfg, ckpt_path, device)
        if not bool(getattr(model, "use_human_motion", False)):
            raise ValueError("Human-reference benchmark requires a checkpoint/config with model.use_human_motion=true")
        ckpt_label = Path(ckpt_path).stem
        for cfg_human_scale in args.resolved_cfg_scales:
            label = ckpt_label if len(args.resolved_cfg_scales) == 1 else f"{ckpt_label}_{_cfg_output_name(cfg_human_scale)}"
            run_dir = root_output_dir if len(args.ckpts) == 1 and len(args.resolved_cfg_scales) == 1 else root_output_dir / label
            print(f"[INFO] Running humanref benchmark ckpt={ckpt_path} cfg_human_scale={cfg_human_scale} output_dir={run_dir}")
            result = _run_single_benchmark(
                output_dir=run_dir,
                ckpt_path=ckpt_path,
                cfg_human_scale=cfg_human_scale,
                datasets=datasets,
                records=records,
                model=model,
                motion_encoder=motion_encoder,
                args=args,
                device=device,
            )
            rows.append(_summary_row(result.benchmark, label=label))
            runs[label] = {
                "ckpt_path": str(Path(ckpt_path).resolve()),
                "cfg_human_scale": _cfg_scale_json(cfg_human_scale),
                "output_dir": str(run_dir.resolve()),
                "benchmark_json": str((run_dir / "benchmark.json").resolve()),
                "embedding_metrics": result.benchmark["embedding_metrics"],
                "tracking_metrics": result.benchmark["tracking_metrics"],
                "physical_metrics": result.benchmark["physical_metrics"],
                "transition_metrics": result.benchmark["transition_metrics"],
                "tracker_executed": result.benchmark["tracker_executed"],
            }
    summary_payload = {
        "benchmark": "humanref",
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
    _write_humanref_summary(root_output_dir, rows, args, datasets)
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
