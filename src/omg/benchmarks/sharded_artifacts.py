"""Utilities for sharding benchmark sample manifests and merging qpos artifacts.

Artifact generation jobs are sample independent. This module keeps that
independence explicit: shards only split ``samples.jsonl`` rows, and merge only
concatenates generated artifacts back into the original manifest order. Metric
computation remains unchanged and should run on the merged artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from omg.benchmarks.artifacts import GENERATED_KEYS, verify_schema


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object, got {type(value).__name__}")
        rows.append(value)
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def _shard_ranges(num_items: int, num_shards: int) -> list[tuple[int, int]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if num_items < num_shards:
        raise ValueError(f"cannot split {num_items} samples into {num_shards} non-empty shards")
    ranges: list[tuple[int, int]] = []
    base, extra = divmod(num_items, num_shards)
    start = 0
    for shard_index in range(num_shards):
        length = base + (1 if shard_index < extra else 0)
        end = start + length
        ranges.append((start, end))
        start = end
    return ranges


def split_samples(
    *,
    samples_path: str | Path,
    output_dir: str | Path,
    num_shards: int,
    shard_prefix: str = "shard",
) -> dict[str, Any]:
    """Split a benchmark ``samples.jsonl`` into contiguous non-empty shards."""

    src = Path(samples_path).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()
    rows = _read_jsonl(src)
    ranges = _shard_ranges(len(rows), int(num_shards))

    shard_rows: list[dict[str, Any]] = []
    for shard_index, (start, end) in enumerate(ranges):
        shard_dir = out_root / f"{shard_prefix}_{shard_index:03d}"
        shard_samples = shard_dir / "samples.jsonl"
        subset = rows[start:end]
        _write_jsonl(shard_samples, subset)
        shard_rows.append(
            {
                "shard_index": int(shard_index),
                "start": int(start),
                "end": int(end),
                "num_samples": int(len(subset)),
                "samples_path": str(shard_samples),
                "output_dir": str(shard_dir),
            }
        )

    manifest = {
        "samples_path": str(src),
        "output_dir": str(out_root),
        "num_samples": int(len(rows)),
        "num_shards": int(num_shards),
        "shards": shard_rows,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "shards.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _load_generated(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"generated_qpos artifact does not exist: {path}")
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    verify_schema(arrays, GENERATED_KEYS, label=str(path))
    return arrays


def _sample_caption(row: dict[str, Any]) -> str:
    return str(row.get("caption") or "")


def _validate_against_samples(payload: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> None:
    count = int(payload["qpos_36"].shape[0])
    if count != len(rows):
        raise ValueError(f"merged artifact has {count} samples but samples manifest has {len(rows)} rows")
    datasets = [str(value) for value in payload["dataset"].tolist()]
    dataset_indices = [int(value) for value in payload["dataset_index"].tolist()]
    captions = [str(value) for value in payload["captions"].tolist()]
    for idx, row in enumerate(rows):
        expected_dataset = str(row["dataset"])
        expected_index = int(row["index"])
        if datasets[idx] != expected_dataset or dataset_indices[idx] != expected_index:
            raise ValueError(
                "merged artifact order does not match samples manifest at "
                f"row {idx}: artifact=({datasets[idx]}, {dataset_indices[idx]}) "
                f"sample=({expected_dataset}, {expected_index})"
            )


def _resolve_shard_artifacts(
    *,
    shard_dirs: list[str] | None,
    shards_root: str | Path | None,
    artifact_name: str,
) -> list[Path]:
    if shard_dirs and shards_root is not None:
        raise ValueError("pass either --shard_dirs or --shards_root, not both")
    if shard_dirs:
        dirs = [Path(value).expanduser().resolve() for value in shard_dirs]
    elif shards_root is not None:
        root = Path(shards_root).expanduser().resolve()
        dirs = sorted(path for path in root.glob("shard_*") if path.is_dir())
    else:
        raise ValueError("merge-generated-qpos requires --shard_dirs or --shards_root")
    if not dirs:
        raise ValueError("no shard directories found")
    return [directory / artifact_name for directory in dirs]


def merge_generated_qpos(
    *,
    samples_path: str | Path,
    output_path: str | Path,
    shard_dirs: list[str] | None = None,
    shards_root: str | Path | None = None,
    artifact_name: str = "generated_qpos.npz",
) -> dict[str, Any]:
    """Concatenate shard ``generated_qpos.npz`` files and validate manifest order."""

    rows = _read_jsonl(Path(samples_path).expanduser().resolve())
    artifact_paths = _resolve_shard_artifacts(
        shard_dirs=shard_dirs,
        shards_root=shards_root,
        artifact_name=artifact_name,
    )
    artifacts = [_load_generated(path) for path in artifact_paths]

    payload = {
        "qpos_36": np.concatenate([item["qpos_36"] for item in artifacts], axis=0).astype(np.float32, copy=False),
        "fps": np.concatenate([item["fps"] for item in artifacts], axis=0).astype(np.float32, copy=False),
        "captions": np.concatenate([item["captions"] for item in artifacts], axis=0).astype(np.str_, copy=False),
        "dataset": np.concatenate([item["dataset"] for item in artifacts], axis=0).astype(np.str_, copy=False),
        "dataset_index": np.concatenate([item["dataset_index"] for item in artifacts], axis=0).astype(np.int32, copy=False),
        "source_generated_qpos": np.asarray([str(path) for path in artifact_paths], dtype=np.str_),
    }
    verify_schema(payload, GENERATED_KEYS, label="merged generated_qpos")
    _validate_against_samples(payload, rows)

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    summary = {
        "samples_path": str(Path(samples_path).expanduser().resolve()),
        "output_path": str(out),
        "num_samples": int(payload["qpos_36"].shape[0]),
        "num_frames": int(payload["qpos_36"].shape[1]),
        "num_shards": int(len(artifact_paths)),
        "shard_artifacts": [str(path) for path in artifact_paths],
    }
    (out.parent / "merge_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shard benchmark samples and merge generated qpos artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_split = subparsers.add_parser("split-samples", help="Split samples.jsonl into contiguous shards.")
    p_split.add_argument("--samples_path", required=True)
    p_split.add_argument("--output_dir", required=True)
    p_split.add_argument("--num_shards", type=int, required=True)
    p_split.add_argument("--shard_prefix", default="shard")

    p_merge = subparsers.add_parser("merge-generated-qpos", help="Merge shard generated_qpos.npz files.")
    p_merge.add_argument("--samples_path", required=True)
    p_merge.add_argument("--output_path", required=True)
    p_merge.add_argument("--shards_root", default=None)
    p_merge.add_argument("--shard_dirs", nargs="+", default=None)
    p_merge.add_argument("--artifact_name", default="generated_qpos.npz")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "split-samples":
        result = split_samples(
            samples_path=args.samples_path,
            output_dir=args.output_dir,
            num_shards=int(args.num_shards),
            shard_prefix=str(args.shard_prefix),
        )
    elif args.command == "merge-generated-qpos":
        result = merge_generated_qpos(
            samples_path=args.samples_path,
            output_path=args.output_path,
            shard_dirs=args.shard_dirs,
            shards_root=args.shards_root,
            artifact_name=str(args.artifact_name),
        )
    else:
        raise SystemExit(f"unknown command: {args.command}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
