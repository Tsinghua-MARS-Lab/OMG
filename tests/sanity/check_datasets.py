from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _config_dir() -> Path:
    return _repo_root() / "configs" / "generation"


def _default_datasets_root() -> Path:
    data_root = Path(os.environ.get("OMG_DATA_ROOT", "data/OMG-Data"))
    return Path(os.environ.get("OMG_DATASETS_ROOT", str(data_root / "datasets")))


def _shape(value: Any) -> str:
    if torch.is_tensor(value):
        return str(tuple(value.shape))
    if value is None:
        return "None"
    return type(value).__name__


def _mask_ratio(item: dict[str, Any], key: str) -> str:
    value = item.get("mask", {}).get(key)
    if not torch.is_tensor(value):
        return "missing"
    if value.numel() == 0:
        return "empty"
    return f"{float(value.float().mean().item()):.4f}"


def _preview(text: Any, max_chars: int = 120) -> str:
    text = str(text or "").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _include_dataset(name: str, include: list[str] | None) -> bool:
    if include is None:
        return True
    lowered = name.lower()
    return any(token.lower() in lowered for token in include)


def _print_root_scan(dataset_root: Path) -> None:
    print(f"[INFO] datasets_root={dataset_root}")
    if not dataset_root.exists():
        print(f"[WARN] datasets_root does not exist on this machine: {dataset_root}")
        return
    names = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    print(f"[INFO] datasets_root_dirs={names}")


def _print_sample(dataset_name: str, idx: int, item: dict[str, Any]) -> None:
    meta = item.get("meta", {}) or {}
    valid = item.get("mask", {}).get("valid")
    valid_frames = int(valid.sum().item()) if torch.is_tensor(valid) else "missing"
    print(
        f"[INFO] sample dataset={dataset_name} index={idx} "
        f"sequence={meta.get('sequence_name')} valid_frames={valid_frames}"
    )
    print(
        f"[INFO]   motion_features={_shape(item.get('motion_features'))} "
        f"qpos_36={_shape(item.get('qpos_36'))} caption_present={bool(str(item.get('caption', '')).strip())}"
    )
    print(
        f"[INFO]   audio_features={_shape(item.get('audio_features'))} "
        f"has_audio_ratio={_mask_ratio(item, 'has_audio')}"
    )
    print(
        f"[INFO]   human_motion={_shape(item.get('human_motion'))} "
        f"has_human_motion_ratio={_mask_ratio(item, 'has_human_motion')}"
    )
    print(
        f"[INFO]   files source_file={meta.get('source_file')} "
        f"label_path={meta.get('label_path')} text_path={meta.get('text_path')}"
    )
    print(
        f"[INFO]   window start={meta.get('window_start')} end={meta.get('window_end')} "
        f"segment=[{meta.get('segment_frame_start')},{meta.get('segment_frame_end')}) "
        f"caption='{_preview(item.get('caption'))}'"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print OMG dataset modality sanity information. This script is read-only and does not "
            "fix schemas or guess missing side modality files."
        )
    )
    parser.add_argument("--data", default="omg_data", help="Hydra data config name.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--datasets", nargs="+", default=None, help="Optional dataset name substrings to include.")
    parser.add_argument("--num-samples", type=int, default=3, help="Samples to print per selected dataset.")
    parser.add_argument(
        "--dataset-root",
        default=str(_default_datasets_root()),
        help="Expected remote datasets root. Used for logging/root sanity only; Hydra config paths remain authoritative.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides, e.g. paths.dataset_root=/data/g1 data.dataset_opts.train.amass.use_human_motion=true",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(_repo_root() / "src"))
    try:
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
    except Exception as exc:
        print(f"[ERROR] Hydra is required on the remote machine: {exc}")
        return 2

    print("[INFO] Dataset loading sanity check")
    print(f"[INFO] config_dir={_config_dir()}")
    print(f"[INFO] data={args.data} split={args.split} datasets={args.datasets or 'ALL'}")
    _print_root_scan(Path(args.dataset_root))
    if args.overrides:
        print(f"[INFO] overrides={args.overrides}")

    with initialize_config_dir(config_dir=str(_config_dir()), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"data={args.data}", "logger=none", *args.overrides])

    dataset_opts = cfg.data.dataset_opts
    if args.split not in dataset_opts:
        raise KeyError(f"Data config has no split '{args.split}'")

    selected = [
        (str(name), dataset_cfg)
        for name, dataset_cfg in dataset_opts[args.split].items()
        if _include_dataset(str(name), args.datasets)
    ]
    if not selected:
        raise ValueError(f"No datasets selected for split={args.split}")

    for name, dataset_cfg in selected:
        print(f"[INFO] dataset={name} instantiate_start")
        dataset = instantiate(dataset_cfg)
        length = len(dataset)
        print(f"[INFO] dataset={name} length={length} class={type(dataset).__name__}")
        for attr in ("dataset_root", "labels_root", "text_root", "audio_dir", "human_motion_dir"):
            value = getattr(dataset, attr, None)
            if value is not None:
                print(f"[INFO] dataset={name} {attr}={value}")
        limit = min(int(args.num_samples), length)
        counts = {"text": 0, "audio": 0, "human_motion": 0}
        for idx in range(limit):
            item = dataset[idx]
            if str(item.get("caption", "")).strip():
                counts["text"] += 1
            has_audio = item.get("mask", {}).get("has_audio")
            has_human_motion = item.get("mask", {}).get("has_human_motion")
            if torch.is_tensor(has_audio) and bool(has_audio.any().item()):
                counts["audio"] += 1
            if torch.is_tensor(has_human_motion) and bool(has_human_motion.any().item()):
                counts["human_motion"] += 1
            _print_sample(name, idx, item)
        print(f"[INFO] dataset={name} checked={limit} counts={counts}")
    print("[INFO] Dataset loading sanity check finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
