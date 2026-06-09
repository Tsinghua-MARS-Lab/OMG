from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from omg.data.materialized_format import MASK_KEYS, TENSOR_KEYS

np = None
torch = None
yaml = None
instantiate = None
OmegaConf = None
tqdm = None


def _ensure_runtime_imports() -> None:
    global np, torch, yaml, instantiate, OmegaConf, tqdm
    if np is not None:
        return
    import numpy as _np
    import torch as _torch
    import yaml as _yaml
    from hydra.utils import instantiate as _instantiate
    from omegaconf import OmegaConf as _OmegaConf
    from tqdm import tqdm as _tqdm

    np = _np
    torch = _torch
    yaml = _yaml
    instantiate = _instantiate
    OmegaConf = _OmegaConf
    tqdm = _tqdm


def _load_yaml(path: Path) -> dict[str, Any]:
    _ensure_runtime_imports()
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return loaded


def _jsonable(value: Any) -> Any:
    _ensure_runtime_imports()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _tensor_to_numpy(value: Any, *, key: str) -> np.ndarray:
    _ensure_runtime_imports()
    if not torch.is_tensor(value):
        raise TypeError(f"Sample key {key!r} must be a torch.Tensor, got {type(value).__name__}")
    array = value.detach().cpu().numpy()
    if array.dtype == np.bool_:
        return array.astype(np.bool_, copy=True)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64, copy=True)
    return array.astype(np.float32, copy=True)


def _resolve_dataset_cfg(
    raw_cfg: dict[str, Any],
    *,
    representation: dict[str, Any],
    paths: dict[str, Any],
    split: str,
    train_window_stride: int,
) -> dict[str, Any]:
    _ensure_runtime_imports()
    root = OmegaConf.create({"paths": paths, "representation": representation, "dataset": raw_cfg})
    OmegaConf.resolve(root)
    resolved = OmegaConf.to_container(root.dataset, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("Resolved dataset config must be a mapping")
    if resolved.get("_target_") == "omg.data.g1_motion.G1MotionDataset":
        resolved.setdefault("rotation_representation", representation.get("rotation_representation", "quat"))
        if split == "train":
            resolved["train_window_policy"] = "exhaustive"
            resolved["train_window_stride"] = int(train_window_stride)
    return resolved


def _default_output_root(args: argparse.Namespace, representation: dict[str, Any], paths: dict[str, Any]) -> Path:
    if args.output_root is not None:
        return Path(args.output_root).expanduser()
    materialized_root = Path(str(paths["materialized_root"])).expanduser()
    rotation = str(representation.get("rotation_representation", "quat")).lower().replace("-", "")
    sequence_length = int(representation["sequence_length"])
    history = int(representation["num_prev_states"])
    return materialized_root / f"filtered_original_mixed_modalities_all_{rotation}_seq{sequence_length}_hist{history}_k{int(args.train_window_stride)}"


def _train_by_dataset_root(output_root: Path, requested: str | None) -> Path:
    if requested is not None:
        return Path(requested).expanduser()
    return output_root.parent / f"{output_root.name}_by_dataset"


class _ShardWriter:
    def __init__(self, root: Path, split: str, *, shard_size: int, overwrite: bool, compress: bool) -> None:
        if shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")
        self.root = root
        self.split = split
        self.split_root = self.root / self.split
        self.shard_size = int(shard_size)
        self.compress = bool(compress)
        self.sample_count = 0
        self.shard_count = 0
        self._arrays: dict[str, list[np.ndarray]] = {key: [] for key in TENSOR_KEYS}
        for key in MASK_KEYS:
            self._arrays[f"mask__{key}"] = []
        self._metadata: list[dict[str, Any]] = []
        self._index_lines: list[str] = []
        if self.split_root.exists():
            if not overwrite:
                raise FileExistsError(f"Materialized split already exists: {self.split_root}")
            shutil.rmtree(self.split_root)
        self.split_root.mkdir(parents=True, exist_ok=False)

    def add(self, sample: dict[str, Any]) -> None:
        for key in TENSOR_KEYS:
            self._arrays[key].append(_tensor_to_numpy(sample[key], key=key))
        mask = sample.get("mask")
        if not isinstance(mask, dict):
            raise TypeError("Sample mask must be a dict")
        for key in MASK_KEYS:
            self._arrays[f"mask__{key}"].append(_tensor_to_numpy(mask[key], key=f"mask.{key}"))
        self._metadata.append(
            {
                "caption": str(sample.get("caption", "")),
                "meta": _jsonable(sample.get("meta", {})),
            }
        )
        self.sample_count += 1
        if len(self._metadata) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._metadata:
            return
        shard_name = f"shard_{self.shard_count:05d}.npz"
        shard_path = self.split_root / shard_name
        arrays = {key: np.stack(values, axis=0) for key, values in self._arrays.items()}
        save = np.savez_compressed if self.compress else np.savez
        save(shard_path, **arrays)
        metadata_path = shard_path.with_suffix(".json")
        metadata_path.write_text(json.dumps(self._metadata, ensure_ascii=False) + "\n", encoding="utf-8")
        for offset, metadata in enumerate(self._metadata):
            self._index_lines.append(
                json.dumps(
                    {
                        "shard": shard_name,
                        "offset": offset,
                        "caption": metadata["caption"],
                        "meta": metadata["meta"],
                    },
                    ensure_ascii=False,
                )
            )
        self.shard_count += 1
        self._arrays = {key: [] for key in TENSOR_KEYS}
        for key in MASK_KEYS:
            self._arrays[f"mask__{key}"] = []
        self._metadata = []

    def close(self) -> dict[str, Any]:
        self.flush()
        (self.split_root / "index.jsonl").write_text("\n".join(self._index_lines) + ("\n" if self._index_lines else ""), encoding="utf-8")
        summary = {
            "format": "omg.materialized.g1_motion.v1",
            "split": self.split,
            "samples": self.sample_count,
            "shards": self.shard_count,
            "shard_size": self.shard_size,
            "tensor_keys": list(TENSOR_KEYS),
            "mask_keys": list(MASK_KEYS),
            "compressed": self.compress,
        }
        (self.split_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary


def _materialize_dataset(
    *,
    name: str,
    cfg: dict[str, Any],
    output_root: Path,
    split: str,
    shard_size: int,
    overwrite: bool,
    compress: bool,
    max_samples: int | None,
) -> dict[str, Any]:
    _ensure_runtime_imports()
    dataset = instantiate(OmegaConf.create(cfg))
    total = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    writer = _ShardWriter(output_root, split, shard_size=shard_size, overwrite=overwrite, compress=compress)
    for idx in tqdm(range(total), desc=f"{split}:{name}"):
        writer.add(dataset[idx])
    summary = writer.close()
    summary["dataset"] = name
    summary["root"] = str(output_root)
    return summary


def _materialize_combined_split(
    *,
    split: str,
    dataset_cfgs: dict[str, dict[str, Any]],
    output_root: Path,
    shard_size: int,
    overwrite: bool,
    compress: bool,
    max_samples_per_dataset: int | None,
) -> dict[str, Any]:
    _ensure_runtime_imports()
    writer = _ShardWriter(output_root, split, shard_size=shard_size, overwrite=overwrite, compress=compress)
    dataset_summaries = []
    for name, cfg in dataset_cfgs.items():
        dataset = instantiate(OmegaConf.create(cfg))
        total = len(dataset) if max_samples_per_dataset is None else min(len(dataset), int(max_samples_per_dataset))
        start = writer.sample_count
        for idx in tqdm(range(total), desc=f"{split}:{name}"):
            writer.add(dataset[idx])
        dataset_summaries.append({"dataset": name, "samples": writer.sample_count - start})
    summary = writer.close()
    summary["root"] = str(output_root)
    summary["datasets"] = dataset_summaries
    (output_root / split / "datasets.json").write_text(
        json.dumps(dataset_summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_runtime_imports()
    data_cfg = _load_yaml(args.data_config)
    repr_cfg = _load_yaml(args.representation_config)
    paths_cfg = _load_yaml(args.paths_config)
    root = OmegaConf.create({"paths": paths_cfg})
    OmegaConf.resolve(root)
    paths = OmegaConf.to_container(root.paths, resolve=True)
    if not isinstance(paths, dict):
        raise ValueError("Resolved paths config must be a mapping")

    output_root = _default_output_root(args, repr_cfg, paths)
    train_by_dataset_root = _train_by_dataset_root(output_root, args.train_by_dataset_root)
    requested_datasets = set(args.datasets or [])
    requested_splits = tuple(args.splits)
    all_summaries: dict[str, Any] = {
        "output_root": str(output_root),
        "train_by_dataset_root": str(train_by_dataset_root),
        "splits": {},
    }

    dataset_opts = data_cfg.get("dataset_opts")
    if not isinstance(dataset_opts, dict):
        raise ValueError("data config must contain dataset_opts mapping")

    for split in requested_splits:
        raw_split_cfgs = dataset_opts.get(split)
        if not isinstance(raw_split_cfgs, dict):
            raise KeyError(f"Split {split!r} not found in {args.data_config}")
        resolved_cfgs: dict[str, dict[str, Any]] = {}
        for name, raw_cfg in raw_split_cfgs.items():
            if requested_datasets and name not in requested_datasets:
                continue
            if not isinstance(raw_cfg, dict):
                raise ValueError(f"Dataset config for {split}:{name} must be a mapping")
            resolved_cfgs[name] = _resolve_dataset_cfg(
                raw_cfg,
                representation=repr_cfg,
                paths=paths,
                split=split,
                train_window_stride=int(args.train_window_stride),
            )
        if requested_datasets and not resolved_cfgs:
            raise ValueError(f"No requested datasets found in split {split}: {sorted(requested_datasets)}")

        if split == "train" and args.train_by_dataset:
            split_summaries = {}
            for name, cfg in resolved_cfgs.items():
                split_summaries[name] = _materialize_dataset(
                    name=name,
                    cfg=cfg,
                    output_root=train_by_dataset_root / name,
                    split=split,
                    shard_size=int(args.shard_size),
                    overwrite=bool(args.overwrite),
                    compress=bool(args.compress),
                    max_samples=args.max_samples_per_dataset,
                )
            all_summaries["splits"][split] = split_summaries
        else:
            all_summaries["splits"][split] = _materialize_combined_split(
                split=split,
                dataset_cfgs=resolved_cfgs,
                output_root=output_root,
                shard_size=int(args.shard_size),
                overwrite=bool(args.overwrite),
                compress=bool(args.compress),
                max_samples_per_dataset=args.max_samples_per_dataset,
            )

    summary_path = output_root / "materialize_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return all_summaries


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize unified OMG source datasets into fixed-window NPZ shards.")
    parser.add_argument("--data-config", type=Path, default=Path("configs/generation/data/omg_data.yaml"))
    parser.add_argument("--representation-config", type=Path, default=Path("configs/generation/representation/125d.yaml"))
    parser.add_argument("--paths-config", type=Path, default=Path("configs/generation/paths/default.yaml"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--train-by-dataset-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset keys to materialize.")
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--train-window-stride", type=int, default=1)
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    parser.add_argument("--train-by-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = materialize(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
