from __future__ import annotations

import copy
import importlib
import math
import os
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset, default_collate

from omg.core.logging import Log

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    try:
        import lightning.pytorch as pl
    except ModuleNotFoundError:
        class _LightningDataModule:
            pass

        class _LightningModuleShim:
            LightningDataModule = _LightningDataModule

        pl = _LightningModuleShim()

try:
    from hydra.utils import instantiate as hydra_instantiate
except ModuleNotFoundError:
    hydra_instantiate = None


HUMAN_MOTION_DIM = 66


def _collate_optional_tensor(values: list[Any]) -> torch.Tensor | list[Any] | None:
    if all(value is None for value in values):
        return None
    if all(torch.is_tensor(value) for value in values):
        return default_collate(values)
    tensor_values = [value for value in values if torch.is_tensor(value)]
    if tensor_values:
        template = tensor_values[0]
        filled = [value if torch.is_tensor(value) else torch.zeros_like(template) for value in values]
        return default_collate(filled)
    return values


def _collate_human_motion(batch: list[dict[str, Any]]) -> torch.Tensor:
    values = [item.get("human_motion") for item in batch]
    tensor_values = [value for value in values if torch.is_tensor(value)]
    if tensor_values:
        template = tensor_values[0]
        filled = [value if torch.is_tensor(value) else torch.zeros_like(template) for value in values]
        return default_collate(filled)

    valid = batch[0].get("mask", {}).get("valid")
    if not torch.is_tensor(valid):
        raise ValueError("Cannot collate human_motion=None without mask[valid] to infer sequence length")
    return valid.new_zeros((len(batch), int(valid.shape[0]), HUMAN_MOTION_DIM), dtype=torch.float32)


def motion_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for item in batch for key in item.keys()})
    collated: dict[str, Any] = {"B": len(batch)}
    for key in keys:
        if key == "meta":
            collated[key] = [item[key] for item in batch]
        elif key == "mask":
            mask_keys = sorted({mask_key for item in batch for mask_key in item["mask"].keys()})
            collated[key] = {
                mask_key: default_collate([item["mask"][mask_key] for item in batch])
                for mask_key in mask_keys
            }
        elif key == "human_motion":
            collated[key] = _collate_human_motion(batch)
        elif key == "audio_features":
            collated[key] = _collate_optional_tensor([item.get(key) for item in batch])
        else:
            collated[key] = default_collate([item[key] for item in batch])
    return collated



def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


def _uses_exhaustive_train_windows(dataset: Dataset) -> bool:
    if bool(getattr(dataset, "uses_exhaustive_train_windows", False)):
        return True
    if isinstance(dataset, ConcatDataset):
        return any(_uses_exhaustive_train_windows(child) for child in dataset.datasets)
    if isinstance(dataset, Subset):
        return _uses_exhaustive_train_windows(dataset.dataset)
    return False


def _materialized_shard_spans(dataset: Dataset, base_index: int = 0) -> list[tuple[int, int]] | None:
    if isinstance(dataset, Subset):
        return None
    if hasattr(dataset, "materialized_shard_spans"):
        spans = dataset.materialized_shard_spans(base_index=base_index)
        return [(int(start), int(count)) for start, count in spans]
    if isinstance(dataset, ConcatDataset):
        spans: list[tuple[int, int]] = []
        previous = 0
        for child, cumulative_size in zip(dataset.datasets, dataset.cumulative_sizes):
            child_spans = _materialized_shard_spans(child, base_index + previous)
            if child_spans is None:
                return None
            spans.extend(child_spans)
            previous = int(cumulative_size)
        return spans
    return None


def _rank_world_size() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


class DistributedPermutationSampler(Sampler[int]):
    """Distributed deterministic pseudo-random permutation with O(1) sampler memory."""

    def __init__(self, num_samples: int, *, seed: int = 0) -> None:
        super().__init__()
        if int(num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        bits = max(2, (self.num_samples - 1).bit_length())
        if bits % 2:
            bits += 1
        self._half_bits = bits // 2
        self._half_mask = (1 << self._half_bits) - 1

    def _rank_num_samples(self) -> tuple[int, int, int]:
        rank, world_size = _rank_world_size()
        if world_size <= 0:
            raise ValueError(f"Invalid WORLD_SIZE={world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid RANK={rank} for WORLD_SIZE={world_size}")
        rank_num_samples = int(math.ceil(self.num_samples / float(world_size)))
        return rank, world_size, rank_num_samples

    def _permute(self, value: int) -> int:
        key = _splitmix64(self.seed + self.epoch * 0xD1B54A32D192ED03)
        value = int(value)
        while True:
            left = value >> self._half_bits
            right = value & self._half_mask
            for round_idx in range(4):
                round_key = key ^ (round_idx * 0x9E3779B97F4A7C15)
                mixed = _splitmix64(right ^ round_key) & self._half_mask
                left, right = right, left ^ mixed
            permuted = (left << self._half_bits) | right
            if permuted < self.num_samples:
                return permuted
            value = permuted

    def __iter__(self):
        rank, world_size, rank_num_samples = self._rank_num_samples()
        for local_index in range(rank_num_samples):
            global_position = (rank + local_index * world_size) % self.num_samples
            yield self._permute(global_position)

    def __len__(self) -> int:
        _, _, rank_num_samples = self._rank_num_samples()
        return rank_num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class DistributedMaterializedShardSampler(Sampler[int]):
    """Distributed shard-local shuffle for materialized datasets.

    The materialized format stores many samples per large shard. Fully random
    sample order causes every worker to repeatedly load unrelated shards. This
    sampler still shuffles globally at the shard level and locally within each
    shard, while keeping consecutive samples shard-local so a training batch can
    reuse the loaded shard.
    """

    def __init__(self, dataset: Dataset, *, seed: int = 0) -> None:
        super().__init__()
        spans = _materialized_shard_spans(dataset)
        if not spans:
            raise ValueError("DistributedMaterializedShardSampler requires materialized shard spans")
        self.spans = spans
        self.num_samples = int(sum(count for _, count in spans))
        self.seed = int(seed)
        self.epoch = 0

    def _rank_span_indices(self) -> list[int]:
        rank, world_size = _rank_world_size()
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 0x9E3779B1)
        order = torch.randperm(len(self.spans), generator=generator).tolist()
        return [span_index for block_position, span_index in enumerate(order) if block_position % world_size == rank]

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 0x9E3779B1 + 0xD1B54A32)
        for span_index in self._rank_span_indices():
            start, count = self.spans[span_index]
            offsets = torch.randperm(count, generator=generator).tolist()
            for offset in offsets:
                yield start + int(offset)

    def __len__(self) -> int:
        return int(sum(self.spans[span_index][1] for span_index in self._rank_span_indices()))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class DistributedWeightedSampler(Sampler[int]):
    def __init__(
        self,
        weights: list[float] | torch.Tensor,
        num_samples: int,
        *,
        replacement: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if weights_tensor.ndim != 1 or weights_tensor.numel() == 0:
            raise ValueError("weights must be a non-empty 1D sequence")
        if not torch.isfinite(weights_tensor).all():
            raise ValueError("weights must be finite")
        if (weights_tensor < 0).any():
            raise ValueError("weights must be non-negative")
        if float(weights_tensor.sum()) <= 0.0:
            raise ValueError("weights must sum to a positive value")
        if int(num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        self.weights = weights_tensor
        self.global_num_samples = int(num_samples)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0

    def _rank_num_samples(self) -> tuple[int, int, int]:
        rank, world_size = _rank_world_size()
        if world_size <= 0:
            raise ValueError(f"Invalid WORLD_SIZE={world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid RANK={rank} for WORLD_SIZE={world_size}")
        rank_num_samples = int(math.ceil(self.global_num_samples / float(world_size)))
        return rank, world_size, rank_num_samples

    def __iter__(self):
        rank, world_size, rank_num_samples = self._rank_num_samples()
        total_size = rank_num_samples * world_size
        generator = torch.Generator()
        if self.replacement:
            generator.manual_seed(self.seed + self.epoch * world_size + rank)
            indices = torch.multinomial(
                self.weights,
                rank_num_samples,
                replacement=True,
                generator=generator,
            )
        else:
            if total_size > self.weights.numel():
                raise ValueError(
                    "replacement=False requires num_samples rounded to all ranks to be <= len(weights)"
                )
            generator.manual_seed(self.seed + self.epoch)
            global_indices = torch.multinomial(
                self.weights,
                total_size,
                replacement=False,
                generator=generator,
            )
            indices = global_indices[rank:total_size:world_size]
        return iter(indices.tolist())

    def __len__(self) -> int:
        _, _, rank_num_samples = self._rank_num_samples()
        return rank_num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class GenerationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_opts: Any,
        loader_opts: Any,
        limit_each_trainset: int | None = None,
        limit_total_trainset: int | None = None,
        train_subset_seed: int = 1234,
        train_subset_ratio: float | None = None,
        train_subset_percent: float | None = None,
        rotation_representation: str = "quat",
        train_window_policy: str | None = None,
        train_window_stride: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_opts = dataset_opts
        self.loader_opts = loader_opts
        self.limit_each_trainset = limit_each_trainset
        self.limit_total_trainset = limit_total_trainset
        self.train_subset_seed = int(train_subset_seed)
        self.train_subset_ratio = train_subset_ratio
        self.train_subset_percent = train_subset_percent
        self.rotation_representation = rotation_representation
        self.train_window_policy = train_window_policy
        self.train_window_stride = train_window_stride
        self.trainset: Dataset | None = None
        self.valsets: list[Dataset] = []
        self.testsets: list[Dataset] = []

    def setup(self, stage: str | None = None) -> None:
        if self.trainset is None and "train" in self.dataset_opts and stage in {None, "fit"}:
            trainsets = []
            for idx, (name, cfg) in enumerate(self.dataset_opts.train.items()):
                start = time.perf_counter()
                Log.info(f"[Train Dataset][{idx + 1}/{len(self.dataset_opts.train)}]: {name} start")
                dataset_i = self._instantiate_dataset(cfg)
                raw_size = len(dataset_i)
                dataset_i = self._limit_trainset(dataset_i)
                trainsets.append(dataset_i)
                Log.info(
                    f"[Train Dataset][{idx + 1}/{len(self.dataset_opts.train)}]: "
                    f"{name} size={len(dataset_i)} raw_size={raw_size} elapsed_sec={time.perf_counter() - start:.3f}"
                )
            self.trainset = ConcatDataset(trainsets)
            self.trainset = self._limit_total_trainset_by_percent(self.trainset)
            self.trainset = self._limit_total_trainset(self.trainset)
            Log.info(f"[Train Dataset][All]: ConcatDataset size={len(self.trainset)}")

        # Keep fit-stage startup focused on train data; large eval sets are built lazily
        # when validation/test is actually requested.
        if not self.valsets and "val" in self.dataset_opts and stage == "validate":
            self.valsets = self._build_eval_sets("val")

        if not self.testsets and "test" in self.dataset_opts and stage == "test":
            self.testsets = self._build_eval_sets("test")

    def _dataset_target(self, cfg: Any) -> str | None:
        if isinstance(cfg, dict):
            return cfg.get("_target_")
        if hasattr(cfg, "get"):
            return cfg.get("_target_")
        return getattr(cfg, "_target_", None)

    def _with_generation_dataset_defaults(self, cfg: Any) -> Any:
        if self._dataset_target(cfg) != "omg.data.g1_motion.G1MotionDataset":
            return cfg
        updates: dict[str, Any] = {}
        if not hasattr(cfg, "get") or cfg.get("rotation_representation") is None:
            updates["rotation_representation"] = self.rotation_representation
        if self.train_window_policy is not None and (not hasattr(cfg, "get") or cfg.get("train_window_policy") is None):
            updates["train_window_policy"] = self.train_window_policy
        if self.train_window_stride is not None and (not hasattr(cfg, "get") or cfg.get("train_window_stride") is None):
            updates["train_window_stride"] = self.train_window_stride
        if not updates:
            return cfg

        patched = copy.deepcopy(cfg)
        try:
            for key, value in updates.items():
                patched[key] = value
        except Exception:
            from omegaconf import OmegaConf

            patched = OmegaConf.to_container(patched, resolve=True)
            for key, value in updates.items():
                patched[key] = value
        return patched

    def _instantiate_dataset(self, cfg: Any) -> Dataset:
        if isinstance(cfg, Dataset):
            return cfg
        cfg = self._with_generation_dataset_defaults(cfg)
        if hydra_instantiate is not None:
            return hydra_instantiate(cfg)
        target = cfg.get("_target_") if isinstance(cfg, dict) else getattr(cfg, "_target_", None)
        if target is None:
            raise ModuleNotFoundError("hydra is not installed and dataset config has no `_target_`")
        module_name, class_name = target.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        items = cfg.items() if isinstance(cfg, dict) else ((key, getattr(cfg, key)) for key in cfg.keys())
        kwargs = {key: value for key, value in items if key != "_target_"}
        return cls(**kwargs)

    def _limit_trainset(self, dataset: Dataset) -> Dataset:
        if self.limit_each_trainset is not None:
            upper = min(len(dataset), int(self.limit_each_trainset))
            dataset = Subset(dataset, list(range(upper)))
        if self.train_subset_ratio is not None:
            upper = max(1, int(len(dataset) * float(self.train_subset_ratio)))
            dataset = Subset(dataset, list(range(upper)))
        return dataset

    def _limit_total_trainset(self, dataset: Dataset) -> Dataset:
        if self.limit_total_trainset is None:
            return dataset
        upper = min(len(dataset), int(self.limit_total_trainset))
        if upper >= len(dataset):
            return dataset
        generator = torch.Generator()
        generator.manual_seed(self.train_subset_seed)
        indices = torch.randperm(len(dataset), generator=generator)[:upper].tolist()
        return Subset(dataset, indices)

    def _limit_total_trainset_by_percent(self, dataset: Dataset) -> Dataset:
        if self.train_subset_percent is None:
            return dataset
        percent = float(self.train_subset_percent)
        if percent <= 0.0 or percent > 100.0:
            raise ValueError(f"train_subset_percent must be in (0, 100], got {self.train_subset_percent}")
        upper = max(1, int(len(dataset) * percent / 100.0))
        if upper >= len(dataset):
            return dataset
        generator = torch.Generator()
        generator.manual_seed(self.train_subset_seed)
        indices = torch.randperm(len(dataset), generator=generator)[:upper].tolist()
        Log.info(
            f"[Train Dataset][All]: train_subset_percent={percent:g} "
            f"selected={upper} total={len(dataset)} seed={self.train_subset_seed}"
        )
        return Subset(dataset, indices)

    def _build_eval_sets(self, split: str) -> list[Dataset]:
        datasets = []
        split_opts = self.dataset_opts[split]
        for idx, (name, cfg) in enumerate(split_opts.items()):
            start = time.perf_counter()
            Log.info(f"[{split.capitalize()} Dataset][{idx + 1}/{len(split_opts)}]: {name} start")
            dataset_i = self._instantiate_dataset(cfg)
            datasets.append(dataset_i)
            Log.info(
                f"[{split.capitalize()} Dataset][{idx + 1}/{len(split_opts)}]: "
                f"{name} size={len(dataset_i)} elapsed_sec={time.perf_counter() - start:.3f}"
            )
        return datasets

    def _make_loader(self, dataset: Dataset, split: str, shuffle: bool) -> DataLoader:
        opts = self.loader_opts[split]
        drop_last = split == "train" and len(dataset) >= int(opts.batch_size)
        loader_kwargs = {
            "batch_size": int(opts.batch_size),
            "shuffle": shuffle,
            "num_workers": int(opts.num_workers),
            "persistent_workers": int(opts.num_workers) > 0 and split == "train",
            "drop_last": drop_last,
            "collate_fn": motion_collate_fn,
        }
        if split == "train" and shuffle:
            materialized_spans = _materialized_shard_spans(dataset)
            if materialized_spans is not None:
                loader_kwargs["shuffle"] = False
                loader_kwargs["sampler"] = DistributedMaterializedShardSampler(
                    dataset,
                    seed=self.train_subset_seed,
                )
            elif _uses_exhaustive_train_windows(dataset):
                loader_kwargs["shuffle"] = False
                loader_kwargs["sampler"] = DistributedPermutationSampler(
                    len(dataset),
                    seed=self.train_subset_seed,
                )
        if int(opts.num_workers) > 0 and opts.get("prefetch_factor") is not None:
            loader_kwargs["prefetch_factor"] = int(opts.prefetch_factor)
        return DataLoader(dataset, **loader_kwargs)

    def train_dataloader(self) -> DataLoader:
        if self.trainset is None:
            self.setup("fit")
        assert self.trainset is not None
        return self._make_loader(self.trainset, split="train", shuffle=True)

    def val_dataloader(self) -> list[DataLoader]:
        if not self.valsets:
            self.setup("validate")
        return [self._make_loader(dataset, split="val", shuffle=False) for dataset in self.valsets]

    def test_dataloader(self) -> list[DataLoader]:
        if not self.testsets:
            self.setup("test")
        return [self._make_loader(dataset, split="test", shuffle=False) for dataset in self.testsets]
