from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from tqdm import tqdm

from omg.benchmarks.runners.common import (
    SampleRecord,
    _config_dir,
    _with_representation_rotation,
    _jsonable,
    item_has_condition,
)
from omg.benchmarks.runners.text import _has_caption


CONDITION_SPECS = {
    "text": {"use_flag": "use_text", "output_name": "text_test_1024.jsonl"},
    "audio": {
        "use_flag": "use_audio",
        "tensor_key": "audio_features",
        "mask_key": "has_audio",
        "output_name": "audio_test_512.jsonl",
    },
    "humanref": {
        "use_flag": "use_human_motion",
        "tensor_key": "human_motion",
        "mask_key": "has_human_motion",
        "output_name": "humanref_test_512.jsonl",
    },
}


def _condition_dataset_names(cfg: Any, *, split: str, condition: str) -> list[str]:
    spec = CONDITION_SPECS[condition]
    names = []
    for name, dataset_cfg in cfg.data.dataset_opts[split].items():
        if bool(dataset_cfg.get(spec["use_flag"], False)):
            names.append(str(name))
    if not names:
        raise ValueError(f"No {condition} datasets found in split {split!r}")
    return names


def _build_selected_datasets(cfg: Any, *, split: str, names: list[str], num_frames: int) -> dict[str, Any]:
    dataset_opts = cfg.data.dataset_opts
    rotation_representation = cfg.representation.get("rotation_representation") if "representation" in cfg else None
    datasets = {}
    for name in names:
        print(f"[INFO] Instantiating {split} dataset {name}", flush=True)
        dataset_cfg = dataset_opts[split][name].copy()
        fps = float(dataset_cfg.get("fps", 30.0))
        dataset_cfg["sequence_duration"] = float(num_frames) / fps
        dataset = instantiate(_with_representation_rotation(dataset_cfg, rotation_representation))
        if len(dataset) <= 0:
            raise ValueError(f"Dataset {name!r} for split {split!r} is empty")
        datasets[str(name)] = dataset
    return datasets


def _item_matches_condition(dataset: Any, index: int, *, condition: str, num_frames: int) -> bool:
    item = dataset[index]
    if condition == "text":
        if not _has_caption(dataset, index):
            return False
        qpos = item.get("qpos_36")
        if qpos is None:
            qpos = item.get("qpos")
        if qpos is None or int(qpos.shape[0]) < int(num_frames):
            return False
        valid = item.get("mask", {}).get("valid") if isinstance(item.get("mask"), dict) else None
        if valid is not None:
            valid_prefix = valid[: int(num_frames)]
            if hasattr(valid_prefix, "detach"):
                return bool(valid_prefix.detach().bool().all().item())
            return bool(np.asarray(valid_prefix, dtype=bool).all())
        return True
    spec = CONDITION_SPECS[condition]
    return item_has_condition(
        item,
        tensor_key=str(spec["tensor_key"]),
        mask_key=str(spec["mask_key"]),
        num_frames=num_frames,
    )


def _candidate_records(
    datasets: dict[str, Any],
    *,
    dataset_names: list[str],
    condition: str,
    num_frames: int,
) -> dict[str, list[SampleRecord]]:
    by_dataset: dict[str, list[SampleRecord]] = defaultdict(list)
    global_index = 0
    selected_names = set(dataset_names)
    for dataset_name, dataset in datasets.items():
        is_selected = dataset_name in selected_names
        for index in tqdm(range(len(dataset)), desc=f"Scan {condition} {dataset_name}", unit="idx", leave=False):
            if is_selected and _item_matches_condition(dataset, index, condition=condition, num_frames=num_frames):
                by_dataset[dataset_name].append(SampleRecord(dataset=dataset_name, index=index, global_index=global_index))
            global_index += 1
    missing = [name for name in dataset_names if not by_dataset.get(name)]
    if missing:
        print(f"[WARN] No valid {condition} samples found for datasets: {missing}", flush=True)
    if not by_dataset:
        raise ValueError(f"No valid {condition} samples found in selected datasets")
    return dict(by_dataset)


def _balanced_counts(candidate_counts: dict[str, int], *, total: int) -> dict[str, int]:
    if total < 1:
        raise ValueError("total must be positive")
    available = {name: int(count) for name, count in candidate_counts.items() if int(count) > 0}
    if not available:
        raise ValueError("No candidate samples available")
    if total > sum(available.values()):
        raise ValueError(f"Requested {total} samples but only {sum(available.values())} are available")

    names = sorted(available)
    base = total // len(names)
    counts = {name: min(base, available[name]) for name in names}
    remaining = total - sum(counts.values())
    while remaining > 0:
        progressed = False
        for name in names:
            if counts[name] >= available[name]:
                continue
            counts[name] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError(f"Failed to allocate all samples; remaining={remaining}")
    return counts


def _select_balanced_records(
    candidates: dict[str, list[SampleRecord]],
    *,
    num_samples: int,
    seed: int,
) -> tuple[list[SampleRecord], dict[str, int], dict[str, int]]:
    candidate_counts = {name: len(records) for name, records in candidates.items()}
    selected_counts = _balanced_counts(candidate_counts, total=num_samples)
    rng = np.random.default_rng(int(seed))
    selected: list[SampleRecord] = []
    for dataset_name in sorted(candidates):
        records = candidates[dataset_name]
        count = selected_counts.get(dataset_name, 0)
        if count <= 0:
            continue
        indices = np.sort(rng.choice(len(records), size=count, replace=False))
        selected.extend(records[int(i)] for i in indices)
    selected.sort(key=lambda record: (record.dataset, record.index))
    return selected, candidate_counts, selected_counts


def _write_sample_records(path: Path, records: list[SampleRecord], datasets: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            item = datasets[record.dataset][record.index]
            payload = asdict(record)
            payload["caption"] = str(item.get("caption", ""))
            payload["meta"] = _jsonable(item.get("meta", {}))
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_manifest_summary(path: Path, summaries: dict[str, dict[str, Any]], args: argparse.Namespace) -> None:
    payload = {
        "benchmark_sample_set": "mixed_modalities_all_v1",
        "data": args.data,
        "exp": args.exp,
        "split": args.split,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "conditions": summaries,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _output_name(condition: str, *, args: argparse.Namespace, num_samples: int) -> str:
    if args.output_name_suffix:
        return f"{condition}_{args.split}_{int(num_samples)}{args.output_name_suffix}.jsonl"
    return str(CONDITION_SPECS[condition]["output_name"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed mixed-modality benchmark sample manifests.")
    parser.add_argument("--output_dir", default="outputs/benchmark_samples/mixed_modalities_all_v1")
    parser.add_argument("--exp", default="100m")
    parser.add_argument("--data", default="omg_data")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text_samples", type=int, default=1024)
    parser.add_argument("--audio_samples", type=int, default=512)
    parser.add_argument("--humanref_samples", type=int, default=512)
    parser.add_argument("--conditions", nargs="+", choices=sorted(CONDITION_SPECS), default=list(CONDITION_SPECS))
    parser.add_argument("--output-name-suffix", default="", help="Optional suffix for generated sample filenames, e.g. _120f.")
    parser.add_argument("--summary-name", default="manifest_summary.json", help="Summary JSON filename inside output_dir.")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = {"text": int(args.text_samples), "audio": int(args.audio_samples), "humanref": int(args.humanref_samples)}

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

    summaries: dict[str, dict[str, Any]] = {}
    for condition in args.conditions:
        spec = CONDITION_SPECS[condition]
        dataset_names = _condition_dataset_names(cfg, split=args.split, condition=condition)
        datasets = _build_selected_datasets(cfg, split=args.split, names=dataset_names, num_frames=args.num_frames)
        candidates = _candidate_records(datasets, dataset_names=dataset_names, condition=condition, num_frames=args.num_frames)
        records, candidate_counts, selected_counts = _select_balanced_records(
            candidates,
            num_samples=requested[condition],
            seed=args.seed,
        )
        path = output_dir / _output_name(condition, args=args, num_samples=requested[condition])
        _write_sample_records(path, records, datasets)
        summaries[condition] = {
            "path": str(path.resolve()),
            "num_samples": len(records),
            "candidate_counts": candidate_counts,
            "selected_counts": selected_counts,
        }
        print(f"[INFO] wrote {condition} samples: {path} selected={selected_counts}", flush=True)

    summary_path = output_dir / str(args.summary_name)
    _write_manifest_summary(summary_path, summaries, args)
    print(f"[INFO] wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
