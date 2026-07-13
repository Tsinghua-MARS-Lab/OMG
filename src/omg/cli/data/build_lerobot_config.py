from __future__ import annotations

import argparse
from pathlib import Path

import yaml


DEFAULT_EXCLUDE_PREFIXES = ("beat2_",)
DEFAULT_EXCLUDE_NAMES = ("amass_finetune",)


def _is_excluded(name: str, exclude_names: set[str], exclude_prefixes: tuple[str, ...]) -> bool:
    return name in exclude_names or any(name.startswith(prefix) for prefix in exclude_prefixes)


def build_config(args: argparse.Namespace) -> dict:
    source_root = Path(args.source_root).expanduser()
    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")
    exclude_names = set(args.exclude_names or [])
    exclude_prefixes = tuple(args.exclude_prefixes or [])
    if args.default_excludes:
        exclude_names.update(DEFAULT_EXCLUDE_NAMES)
        exclude_prefixes = (*exclude_prefixes, *DEFAULT_EXCLUDE_PREFIXES)
    dataset_opts = {split: {} for split in ("train", "val", "test")}
    skipped = {}
    for dataset_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        name = dataset_dir.name
        reason = None
        if _is_excluded(name, exclude_names, exclude_prefixes):
            reason = "excluded"
        elif not (dataset_dir / "g1").is_dir():
            reason = "missing_g1"
        elif not (dataset_dir / "info.yaml").is_file():
            reason = "missing_info"
        if reason is not None:
            skipped[name] = reason
            continue
        has_labels = (dataset_dir / "labels").is_dir()
        if args.require_labels and not has_labels:
            skipped[name] = "missing_labels"
            continue
        with (dataset_dir / "info.yaml").open("r", encoding="utf-8") as f:
            info = yaml.safe_load(f) or {}
        available_splits = [split for split in dataset_opts if split in info]
        if not available_splits:
            skipped[name] = "missing_splits"
            continue
        for split in available_splits:
            cfg = {
                "_target_": "omg.data.g1_motion.G1MotionDataset",
                "dataset_root": str(dataset_dir / "g1"),
                "info_path": str(dataset_dir / "info.yaml"),
                "split": split,
                "sequence_duration": float(args.sequence_duration),
                "fps": float(args.fps),
                "sample_by_segment": True,
                "include_style_in_caption": True,
                "skip_missing_labels": bool(args.skip_missing_labels),
                "use_text": bool(has_labels),
                "use_audio": False,
                "audio_dim": 35,
                "use_human_motion": False,
                "human_motion_dim": 66,
            }
            if has_labels:
                cfg["labels_root"] = str(dataset_dir / "labels")
            dataset_opts[split][f"{name}_{split}"] = cfg
    dataset_names = {key.rsplit("_", 1)[0] for configs in dataset_opts.values() for key in configs}
    return {
        "metadata": {
            "source_root": str(source_root),
            "default_excludes": bool(args.default_excludes),
            "exclude_names": sorted(exclude_names),
            "exclude_prefixes": list(exclude_prefixes),
            "require_labels": bool(args.require_labels),
            "datasets": len(dataset_names),
            "split_datasets": {split: len(configs) for split, configs in dataset_opts.items()},
            "skipped": skipped,
        },
        "dataset_opts": dataset_opts,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a LeRobot export data config from a unified OMG source root.")
    parser.add_argument("--source-root", type=Path, required=True, help="Directory containing dataset subdirectories.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-duration", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--skip-missing-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--default-excludes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-names", nargs="*", default=[])
    parser.add_argument("--exclude-prefixes", nargs="*", default=[])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = build_config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(config["metadata"], sort_keys=False))


if __name__ == "__main__":
    main()
