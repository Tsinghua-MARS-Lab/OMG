from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from omg.data.materialized_format import DEFAULT_TRAIN_TENSOR_KEYS, MASK_KEYS, TENSOR_KEYS


class MaterializedG1MotionDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        limit_size: int | None = None,
        preload_shards: bool = False,
        shard_cache_size: int = 1,
        tensor_keys: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = str(split)
        self.split_root = self.root / self.split
        summary_path = self.split_root / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Materialized summary not found: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.num_samples = int(summary["samples"])
        self.num_shards = int(summary["shards"])
        self.shard_size = int(summary["shard_size"])
        if limit_size is not None:
            self.num_samples = min(self.num_samples, int(limit_size))
        self.shard_cache_size = max(0, int(shard_cache_size))
        self.tensor_keys = tuple(tensor_keys) if tensor_keys is not None else DEFAULT_TRAIN_TENSOR_KEYS
        unknown_keys = sorted(set(self.tensor_keys) - set(TENSOR_KEYS))
        if unknown_keys:
            raise ValueError(f"Unknown materialized tensor_keys={unknown_keys}")
        self.uses_materialized_shards = True
        self.uses_exhaustive_train_windows = True
        self._shard_cache: OrderedDict[str, tuple[dict[str, np.ndarray], list[dict[str, Any]]]] = OrderedDict()
        if preload_shards:
            for shard_id in range(self.num_shards):
                self._load_shard(f"shard_{shard_id:05d}.npz")
        print(
            f"[INFO] MaterializedG1MotionDataset root={self.root} split={self.split} "
            f"samples={self.num_samples} shards={self.num_shards} "
            f"shard_size={self.shard_size} preload_shards={bool(preload_shards)} "
            f"shard_cache_size={self.shard_cache_size} tensor_keys={self.tensor_keys}"
        )

    def __len__(self) -> int:
        return self.num_samples

    def _load_shard(self, shard_name: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        cached = self._shard_cache.get(shard_name)
        if cached is not None:
            self._shard_cache.move_to_end(shard_name)
            return cached
        shard_path = self.split_root / shard_name
        meta_path = shard_path.with_suffix(".json")
        wanted = set(self.tensor_keys) | {f"mask__{key}" for key in MASK_KEYS}
        with np.load(shard_path) as npz:
            arrays = {key: np.asarray(npz[key]) for key in npz.files if key in wanted}
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        loaded = (arrays, metadata)
        if self.shard_cache_size > 0:
            self._shard_cache[shard_name] = loaded
            self._shard_cache.move_to_end(shard_name)
            while len(self._shard_cache) > self.shard_cache_size:
                self._shard_cache.popitem(last=False)
        return loaded

    def _locate(self, idx: int) -> tuple[str, int]:
        if idx < 0:
            idx += self.num_samples
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)
        shard_id = idx // self.shard_size
        offset = idx - shard_id * self.shard_size
        if shard_id >= self.num_shards:
            raise IndexError(idx)
        return f"shard_{shard_id:05d}.npz", offset

    def materialized_shard_spans(self, base_index: int = 0) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        remaining = self.num_samples
        for shard_id in range(self.num_shards):
            start = int(base_index) + shard_id * self.shard_size
            count = min(self.shard_size, remaining)
            if count <= 0:
                break
            spans.append((start, count))
            remaining -= count
        return spans

    def iter_stats_batches(
        self,
        *,
        batch_size: int,
        max_samples: int | None = None,
        device: torch.device | str = "cpu",
        episode_batch_frames: int = 65536,
        rank: int = 0,
        world_size: int = 1,
    ) -> Iterator[dict[str, torch.Tensor]]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if int(episode_batch_frames) <= 0:
            raise ValueError(f"episode_batch_frames must be positive, got {episode_batch_frames}")
        rank = int(rank)
        world_size = int(world_size)
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid rank/world_size: {rank}/{world_size}")
        effective_total = self.num_samples if max_samples is None else min(self.num_samples, int(max_samples))
        global_begin = effective_total * rank // world_size
        global_end = effective_total * (rank + 1) // world_size
        sample_limit = global_end - global_begin
        target_device = torch.device(device)
        emitted = 0
        first_shard = global_begin // self.shard_size
        last_shard = (global_end + self.shard_size - 1) // self.shard_size
        for shard_id in range(first_shard, min(last_shard, self.num_shards)):
            if emitted >= sample_limit:
                break
            shard_name = f"shard_{shard_id:05d}.npz"
            with np.load(self.split_root / shard_name) as npz:
                arrays = {
                    "motion_features": np.asarray(npz["motion_features"]),
                    "qpos_36": np.asarray(npz["qpos_36"]),
                    "mask__valid": np.asarray(npz["mask__valid"]),
                }
            shard_global_start = shard_id * self.shard_size
            offset_begin = max(0, global_begin - shard_global_start)
            offset_end = min(int(arrays["motion_features"].shape[0]), global_end - shard_global_start)
            for offset in range(offset_begin, offset_end, batch_size):
                end = min(offset + batch_size, offset_end)
                yield {
                    "motion_features": self._tensor(arrays["motion_features"][offset:end]).to(target_device),
                    "qpos_36": self._tensor(arrays["qpos_36"][offset:end]).to(target_device),
                    "valid_mask": self._tensor(arrays["mask__valid"][offset:end]).to(target_device),
                }
                emitted += end - offset

    @staticmethod
    def _tensor(value: np.ndarray) -> torch.Tensor:
        array = np.asarray(value)
        if array.dtype == np.bool_:
            return torch.from_numpy(array.astype(np.bool_, copy=True))
        if np.issubdtype(array.dtype, np.integer):
            return torch.from_numpy(array.astype(np.int64, copy=True))
        return torch.from_numpy(array.astype(np.float32, copy=True))

    @staticmethod
    def _has_valid_human_motion(human_motion: np.ndarray, valid_mask: np.ndarray | None = None) -> bool:
        human = np.asarray(human_motion)
        if valid_mask is not None:
            human = human[np.asarray(valid_mask, dtype=np.bool_)]
        return bool(human.size > 0 and np.isfinite(human).all() and np.count_nonzero(human) > 0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_name, offset = self._locate(int(idx))
        arrays, metadata = self._load_shard(shard_name)
        sample: dict[str, Any] = {}
        for key in self.tensor_keys:
            sample[key] = self._tensor(arrays[key][offset])
        sample["mask"] = {}
        for key in MASK_KEYS:
            array_key = f"mask__{key}"
            if array_key in arrays:
                sample["mask"][key] = self._tensor(arrays[array_key][offset])
            else:
                sample["mask"][key] = torch.zeros_like(sample["mask"]["valid"], dtype=torch.bool)
        if bool(sample["mask"].get("has_human_motion", torch.tensor(False)).any().item()):
            human_np = np.asarray(arrays["human_motion"][offset])
            valid_np = np.asarray(arrays["mask__valid"][offset], dtype=np.bool_)
            if not self._has_valid_human_motion(human_np, valid_np):
                sample["mask"]["has_human_motion"] = torch.zeros_like(sample["mask"]["has_human_motion"], dtype=torch.bool)
        sample["caption"] = str(metadata[offset].get("caption", ""))
        sample["meta"] = dict(metadata[offset]["meta"])
        return sample
