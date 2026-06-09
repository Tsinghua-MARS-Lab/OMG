from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from omg.data.split_yaml import dataset_motion_npz_path, flatten_dataset_split_paths


SEGMENT_INDEX_FORMAT = "omg_segment_index_v5_compact_pairs_clamped"


def _window_count(segment_start: int, segment_end: int, window_size: int) -> int:
    return max(0, segment_end - segment_start - window_size + 1)


def _clamp_segment_frames(start_frame: int, end_frame: int, total_len: int) -> tuple[int, int] | None:
    if total_len <= 0:
        return None
    start_frame = max(0, min(start_frame, total_len - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_len))
    if end_frame <= start_frame:
        return None
    return start_frame, end_frame


def _resolve_split(split: str, info: dict, *, info_path: str | Path) -> str:
    if split in info:
        return split
    if split == "valid" and "val" in info:
        return "val"
    raise ValueError(f"Split `{split}` not found in {info_path}")


def _label_lookup(labels_root: Path) -> dict[str, Path | None]:
    lookup: dict[str, Path | None] = {}
    for path in labels_root.rglob("*.json"):
        stem = path.stem
        lookup[stem] = path
        for index, char in enumerate(stem):
            if char != "_":
                continue
            suffix_key = f"__suffix__::{stem[index + 1:]}"
            existing = lookup.get(suffix_key)
            if existing is None and suffix_key in lookup:
                continue
            if existing is not None and existing != path:
                lookup[suffix_key] = None
            else:
                lookup[suffix_key] = path
    return lookup


def _strip_motion_suffix(stem: str) -> str:
    for suffix in (
        "_retarget",
        "_poses_120_jpos",
        "_poses_100_jpos",
        "_poses_60_jpos",
        "_poses_30_jpos",
        "_120",
        "_100",
        "_60",
        "_30",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _filename_variants(stem: str) -> list[str]:
    base = _strip_motion_suffix(stem)
    variants = [
        stem,
        base,
        base.replace(" ", "").replace("-", ""),
        base.replace(" ", ""),
        base.replace(" ", "-"),
        base.replace(" ", "_"),
    ]
    out = []
    seen = set()
    for variant in variants:
        if variant and variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out


def _iter_entry_label_candidates(labels_root: Path, entry: str, sequence_name: str) -> list[Path]:
    entry_path = Path(entry)
    candidates: list[Path] = []
    stem_variants = _filename_variants(sequence_name)

    candidates.append(labels_root / entry_path.with_suffix(".json"))
    candidates.extend(labels_root / entry_path.parent / f"{stem}.json" for stem in stem_variants)

    # AMASS stores entries like BMLhandball/S01_Expert/Trial_... as
    # labels/BMLhandball/S01/Expert_Trial_....json. Keep this resolver
    # deterministic so the precomputed index matches the AMASS dataloader.
    if entry_path.parent.name.endswith("_c3d") and len(entry_path.parts) >= 3:
        label_parent = entry_path.parent.name[: -len("_c3d")]
        label_dir = labels_root / entry_path.parent.parent / label_parent
        candidates.extend(label_dir / f"c3d_{stem}.json" for stem in stem_variants)

    parent_name = entry_path.parent.name
    if "_" in parent_name and len(entry_path.parts) >= 2:
        subject, label_prefix = parent_name.split("_", 1)
        label_dir = labels_root / entry_path.parent.parent / subject
        candidates.extend(label_dir / f"{label_prefix}_{stem}.json" for stem in stem_variants)

    if entry_path.parts and "_" in entry_path.parts[0] and len(entry_path.parts) >= 3:
        label_dir = labels_root.joinpath(*entry_path.parts[0].split("_"))
        subject_prefix = entry_path.parts[1]
        candidates.extend(label_dir / f"{subject_prefix}_{stem}.json" for stem in stem_variants)

    out = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _resolve_label_path(labels_root: Path, entry: str, sequence_name: str, lookup: dict[str, Path | None]) -> Path | None:
    for variant in _filename_variants(sequence_name):
        exact = lookup.get(variant)
        if exact is not None:
            return exact

    for candidate in _iter_entry_label_candidates(labels_root, entry, sequence_name):
        if candidate.exists():
            return candidate

    for variant in _filename_variants(sequence_name):
        suffix = lookup.get(f"__suffix__::{variant}")
        if suffix is not None:
            return suffix
    return None


def build_segment_index(
    *,
    dataset_root: str | Path,
    info_path: str | Path,
    labels_root: str | Path,
    split: str,
    window_size: int,
    segment_sample_ratio: float = 0.2,
    seed: int = 1234,
    default_fps: float = 30.0,
    dataset_name: str | None = None,
) -> tuple[list[dict], dict]:
    dataset_root = Path(dataset_root)
    labels_root = Path(labels_root)
    dataset_name = dataset_name or dataset_root.parent.name or dataset_root.name
    with Path(info_path).open("r", encoding="utf-8") as f:
        info = yaml.safe_load(f)
    requested_split = split
    split = _resolve_split(split, info, info_path=info_path)

    pairs: list[dict] = []
    motion_stats: list[dict] = []
    missing_label_count = 0
    total_window_count = 0
    label_lookup: dict[str, Path | None] | None = None
    entries = flatten_dataset_split_paths(info[split])
    for entry in entries:
        path = dataset_motion_npz_path(dataset_root, entry)
        if not path.exists():
            raise FileNotFoundError(f"Missing sample for split `{split}`: {path}")

        sequence_name = Path(entry).name
        if label_lookup is None:
            label_lookup = _label_lookup(labels_root)
        label_path = _resolve_label_path(labels_root, entry, sequence_name, label_lookup)
        if label_path is None or not label_path.exists():
            missing_label_count += 1
            if missing_label_count <= 10:
                print(f"warning: missing label file for {sequence_name}; searched under {labels_root}")
            continue

        with np.load(path, mmap_mode="r") as npz:
            total_len = int(npz["qpos"].shape[0])
            fps = float(npz["fps"]) if "fps" in npz else float(default_fps)

        with label_path.open("r", encoding="utf-8") as f:
            label_data = json.load(f)
        segments = label_data.get("segments", [])
        if len(segments) == 0:
            continue

        motion_pair_count = 0
        motion_window_count = 0
        for segment_index, segment in enumerate(segments):
            start_frame = int(round(float(segment["start_time"]) * fps))
            end_frame = int(round(float(segment["end_time"]) * fps))
            clamped = _clamp_segment_frames(start_frame, end_frame, total_len)
            if clamped is None:
                continue
            start_frame, end_frame = clamped
            window_count = _window_count(start_frame, end_frame, window_size)
            if window_count <= 0:
                continue
            action_text = str(segment.get("action", "")).strip()
            pairs.append(
                {
                    "dataset_name": dataset_name,
                    "entry": entry,
                    "motion_id": entry,
                    "path": str(path),
                    "motion_path": str(path),
                    "sequence_name": sequence_name,
                    "fps": fps,
                    "label_path": str(label_path),
                    "annotation_id": f"{entry}:{segment_index}",
                    "segment_index": segment_index,
                    "action_text": action_text,
                    "segment_frame_start": start_frame,
                    "segment_frame_end": end_frame,
                    "action_start_frame": start_frame,
                    "action_end_frame": end_frame,
                    "motion_num_frames": total_len,
                    "segment_length": int(window_size),
                    "num_segments": window_count,
                    "window_count": window_count,
                }
            )
            motion_pair_count += 1
            motion_window_count += window_count
        total_window_count += motion_window_count
        motion_stats.append(
            {
                "motion_id": entry,
                "motion_length": total_len,
                "text_motion_pair_count": motion_pair_count,
                "total_segment_count": motion_window_count,
                "cached_segment_count": motion_window_count,
                "segment_sample_ratio": float(segment_sample_ratio),
            }
        )

    metadata = {
        "dataset_root": str(dataset_root),
        "dataset_name": dataset_name,
        "info_path": str(info_path),
        "labels_root": str(labels_root),
        "split": split,
        "requested_split": requested_split,
        "window_size": int(window_size),
        "segment_sample_ratio": float(segment_sample_ratio),
        "seed": int(seed),
        "default_fps": float(default_fps),
        "num_entries": len(entries),
        "num_pairs": len(pairs),
        "num_windows": total_window_count,
        "num_samples": total_window_count,
        "missing_label_count": missing_label_count,
        "motion_segment_stats": motion_stats,
        "format": SEGMENT_INDEX_FORMAT,
    }
    return pairs, metadata


def save_segment_index(pairs: list[dict], metadata: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pairs_jsonl=np.array([json.dumps(pair, ensure_ascii=False) for pair in pairs]),
        metadata_json=np.array(json.dumps(metadata, indent=2, ensure_ascii=False)),
    )
    return output_path


def load_segment_index(index_path: str | Path) -> tuple[list[dict], dict]:
    with np.load(index_path, allow_pickle=False) as npz:
        samples = [json.loads(str(pair_json)) for pair_json in npz["pairs_jsonl"]]
        metadata = json.loads(str(npz["metadata_json"].item()))
    return samples, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute fixed text-motion segment windows for OMG datasets.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--info-path", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--segment-length", type=int, default=60)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--segment-sample-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_size = int(args.segment_length)
    if window_size <= 0:
        raise ValueError(f"Invalid segment length: {args.segment_length}")

    output_dir = Path(args.output_dir)
    for split in args.splits:
        seed = args.seed if split == "train" else args.eval_seed
        output_path = output_dir / f"segment_index_{split}.npz"
        if output_path.exists() and not args.overwrite:
            print(f"[INFO] Found existing segment index: {output_path}")
            print("[INFO] Loading existing segment index, skip precompute. Use --overwrite to rebuild.")
            pairs, metadata = load_segment_index(output_path)
            print(
                f"[INFO] {split}: {metadata.get('num_pairs', len(pairs))} text-motion pairs, "
                f"{metadata.get('num_windows', metadata.get('num_samples', 0))} windows"
            )
            continue
        if output_path.exists() and args.overwrite:
            print(f"[INFO] --overwrite is enabled. Rebuilding segment index: {output_path}")

        print(f"[INFO] Precomputing compact segment index for split={split} with [start, end) windows")
        pairs, metadata = build_segment_index(
            dataset_root=args.dataset_root,
            info_path=args.info_path,
            labels_root=args.labels_root,
            split=split,
            window_size=window_size,
            segment_sample_ratio=args.segment_sample_ratio,
            seed=seed,
            default_fps=args.fps,
            dataset_name=args.dataset_name,
        )
        output_path = save_segment_index(pairs, metadata, output_path)
        print(
            f"[INFO] {split}: {metadata['num_pairs']} text-motion pairs, "
            f"{metadata['num_windows']} windows -> {output_path}"
        )
        if metadata.get("missing_label_count", 0) > 0:
            print(f"[INFO] {split}: skipped {metadata['missing_label_count']} entries without labels")
        for stat in metadata["motion_segment_stats"][:5]:
            print(
                "[INFO]   "
                f"motion_id: {stat['motion_id']} "
                f"motion_length: {stat['motion_length']} "
                f"total_segment_count: {stat['total_segment_count']} "
                f"cached_segment_count: {stat['cached_segment_count']} "
                f"segment_sample_ratio: {stat['segment_sample_ratio']}"
            )


if __name__ == "__main__":
    main()
