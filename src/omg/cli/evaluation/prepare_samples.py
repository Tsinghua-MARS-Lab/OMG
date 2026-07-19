from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from tqdm import tqdm

from omg.benchmarks.runners.common import (
    SampleRecord,
    _build_datasets,
    _config_dir,
    _write_sample_records,
)


CONDITION_SPECS = {
    "text": {"output_name": "text_test_1024.jsonl"},
    "audio": {"output_name": "audio_test_512.jsonl"},
    "humanref": {"output_name": "humanref_test_512.jsonl"},
}

BENCHMARK_COHORT_FAMILIES = {
    "text": {
        name: (name,)
        for name in (
            "100style",
            "amass",
            "bones_seed",
            "fitness",
            "humanml",
            "idea400",
            "lafan1",
            "motiongv",
            "motionllama",
            "omomo",
            "permo",
            "snapmogen",
        )
    },
    "audio": {
        name: (name,)
        for name in (
            "aioz_gdance",
            "aistpp",
            "compas3d",
            "finedance",
            "opendance",
        )
    },
    "humanref": {
        **{
            name: (name,)
            for name in (
                "aistpp",
                "amass",
                "finedance",
                "fitness",
                "humanml",
                "idea400",
                "motiongv",
                "motionllama",
                "permo",
                "snapmogen",
            )
        },
        "beat2": (
            "beat2_chinese",
            "beat2_english",
            "beat2_japanese",
            "beat2_spanish",
        ),
    },
}


def _condition_cohorts(condition: str, split: str) -> dict[str, tuple[str, ...]]:
    return {
        cohort: tuple(f"{family}_{split}" for family in families)
        for cohort, families in BENCHMARK_COHORT_FAMILIES[condition].items()
    }


def _item_matches_condition(dataset: Any, index: int, *, condition: str, num_frames: int) -> bool:
    if not hasattr(dataset, "sample_has_condition"):
        raise TypeError("Benchmark sample preparation requires LeRobot benchmark views")
    return bool(dataset.sample_has_condition(index, condition, num_frames=int(num_frames)))


def _candidate_records(
    datasets: dict[str, Any],
    *,
    cohorts: dict[str, tuple[str, ...]],
    condition: str,
    num_frames: int,
) -> dict[str, list[SampleRecord]]:
    by_cohort: dict[str, list[SampleRecord]] = defaultdict(list)
    source_to_cohort: dict[str, str] = {}
    for cohort, source_datasets in cohorts.items():
        for source_dataset in source_datasets:
            if source_dataset not in datasets:
                raise KeyError(f"Benchmark cohort {cohort!r} requires missing source dataset {source_dataset!r}")
            if source_dataset in source_to_cohort:
                raise ValueError(f"Source dataset {source_dataset!r} belongs to multiple benchmark cohorts")
            source_to_cohort[source_dataset] = cohort
    for dataset_name, dataset in datasets.items():
        cohort = source_to_cohort.get(dataset_name)
        if cohort is None:
            continue
        for index in tqdm(range(len(dataset)), desc=f"Scan {condition} {dataset_name}", unit="idx", leave=False):
            if _item_matches_condition(dataset, index, condition=condition, num_frames=num_frames):
                source_index = dataset.global_index(index)
                by_cohort[cohort].append(
                    SampleRecord(dataset=dataset_name, index=index, global_index=source_index)
                )
    missing = [cohort for cohort in cohorts if not by_cohort.get(cohort)]
    if missing:
        raise ValueError(f"No valid {condition} samples found for benchmark cohorts: {missing}")
    if not by_cohort:
        raise ValueError(f"No valid {condition} samples found in selected datasets")
    return dict(by_cohort)


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


def _write_manifest_summary(
    path: Path,
    summaries: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    dataset_identity: dict[str, str],
) -> None:
    payload = {
        "benchmark_sample_set": "mixed_modalities_all_v2",
        "sample_schema": "omg.benchmark.sample.v2",
        **dataset_identity,
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
    parser.add_argument("--output_dir", default="outputs/benchmark_samples/mixed_modalities_all_v2")
    parser.add_argument("--exp", default="100m")
    parser.add_argument("--data", default="omg_data_lerobot_omnimodal")
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

    datasets = _build_datasets(cfg, args.split, num_frames=args.num_frames)
    first_dataset = next(iter(datasets.values()))
    dataset_identity = {"repo_id": str(first_dataset.repo_id), "revision": str(first_dataset.revision)}
    summaries: dict[str, dict[str, Any]] = {}
    for condition in args.conditions:
        cohorts = _condition_cohorts(condition, args.split)
        candidates = _candidate_records(
            datasets,
            cohorts=cohorts,
            condition=condition,
            num_frames=args.num_frames,
        )
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
            "cohorts": {name: list(source_datasets) for name, source_datasets in cohorts.items()},
        }
        print(f"[INFO] wrote {condition} samples: {path} selected={selected_counts}", flush=True)

    summary_path = output_dir / str(args.summary_name)
    _write_manifest_summary(summary_path, summaries, args, dataset_identity)
    print(f"[INFO] wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
