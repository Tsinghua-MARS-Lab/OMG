from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from omg.data.materialized_format import DEFAULT_TRAIN_TENSOR_KEYS


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_dataset_cfg(cfg: dict[str, Any], representation: dict[str, Any], paths: dict[str, Any]) -> dict[str, Any]:
    root = OmegaConf.create({"paths": paths, "representation": representation, "dataset": cfg})
    OmegaConf.resolve(root)
    resolved = OmegaConf.to_container(root.dataset, resolve=True)
    target = str(resolved.get("_target_", ""))
    if target in {
        "omg.data.g1_motion.G1MotionDataset",
        "omg.data.lerobot_dataset.LeRobotG1MotionDataset",
    }:
        resolved.setdefault("rotation_representation", representation.get("rotation_representation", "quat"))
    elif target == "omg.data.materialized.MaterializedG1MotionDataset":
        tensor_keys = list(resolved.get("tensor_keys") or DEFAULT_TRAIN_TENSOR_KEYS)
        for key in ("motion_features", "qpos_36"):
            if key not in tensor_keys:
                tensor_keys.append(key)
        resolved["tensor_keys"] = tensor_keys
    return resolved


def _update_moments(
    count: int,
    mean: torch.Tensor | None,
    m2: torch.Tensor | None,
    values: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    values = values.double()
    batch_count = int(values.shape[0])
    batch_mean = values.mean(dim=0)
    batch_m2 = ((values - batch_mean).pow(2)).sum(dim=0)
    if mean is None or m2 is None or count == 0:
        return batch_count, batch_mean, batch_m2
    total = count + batch_count
    delta = batch_mean - mean
    new_mean = mean + delta * (batch_count / total)
    new_m2 = m2 + batch_m2 + delta.pow(2) * count * batch_count / total
    return total, new_mean, new_m2


def compute_stats(args: argparse.Namespace) -> dict[str, Any]:
    data_cfg = _load_yaml(args.data_config)
    repr_cfg = _load_yaml(args.representation_config)
    paths_cfg = _load_yaml(args.paths_config)
    datasets = data_cfg["dataset_opts"][args.split]
    count = 0
    mean = None
    m2 = None
    qpos_sum = None
    qpos_count = 0

    for name, raw_cfg in datasets.items():
        cfg = _resolve_dataset_cfg(raw_cfg, repr_cfg, paths_cfg)
        dataset = instantiate(OmegaConf.create(cfg))
        if not hasattr(dataset, "iter_stats_batches"):
            raise TypeError(
                f"Dataset {type(dataset).__name__} must implement iter_stats_batches() for exact scalable stats"
            )
        batch_iterator = dataset.iter_stats_batches(
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=args.device,
            episode_batch_frames=args.episode_batch_frames,
            rank=args.rank,
            world_size=args.world_size,
        )
        iterator = tqdm(batch_iterator, desc=name, total=None, disable=args.rank != 0)
        for batch in iterator:
            valid = batch["valid_mask"].bool()
            features = batch["motion_features"][valid]
            qpos = batch["qpos_36"][valid]
            if features.numel() == 0:
                continue
            count, mean, m2 = _update_moments(count, mean, m2, features)
            if qpos_sum is None:
                qpos_sum = torch.zeros(36, dtype=torch.float64, device=qpos.device)
            qpos_sum += qpos.double().sum(dim=0)
            qpos_count += int(qpos.shape[0])

    if args.world_size > 1:
        feature_dim = int(repr_cfg["feat_dim"])
        moments_device = torch.device(args.device)
        if mean is None or m2 is None:
            mean = torch.zeros(feature_dim, dtype=torch.float64, device=moments_device)
            m2 = torch.zeros_like(mean)
        elif int(mean.numel()) != feature_dim:
            raise ValueError(f"Expected {feature_dim} feature dimensions, got {mean.numel()}")
        if qpos_sum is None:
            qpos_sum = torch.zeros(36, dtype=torch.float64, device=moments_device)
        aggregate = torch.cat(
            [
                torch.tensor([count, qpos_count], dtype=torch.float64, device=mean.device),
                mean.double() * count,
                m2.double() + mean.double().square() * count,
                qpos_sum,
            ]
        )
        dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
        count = int(aggregate[0].item())
        qpos_count = int(aggregate[1].item())
        if count == 0 or qpos_count == 0:
            raise RuntimeError("No valid frames were found across distributed statistics workers")
        feature_dim = int(mean.numel())
        feature_sum = aggregate[2 : 2 + feature_dim]
        feature_sum_sq = aggregate[2 + feature_dim : 2 + 2 * feature_dim]
        qpos_sum = aggregate[2 + 2 * feature_dim :]
        mean = feature_sum / count
        m2 = (feature_sum_sq - feature_sum.square() / count).clamp_min(0.0)
    elif mean is None or m2 is None or qpos_sum is None or count == 0 or qpos_count == 0:
        raise RuntimeError("No valid frames were found while computing stats")
    std = (m2 / max(count - 1, 1)).sqrt().clamp_min(float(args.std_min))
    default_qpos = (qpos_sum / qpos_count).float()
    default_root_quat = torch.nn.functional.normalize(default_qpos[3:7], dim=0)
    rotation_representation = str(repr_cfg.get("rotation_representation", "quat"))
    rot_key = rotation_representation.strip().lower().replace("-", "_")
    root_rot_dim = {
        "quat": 4,
        "quaternion": 4,
        "quat4": 4,
        "rot6d": 6,
        "rotation_6d": 6,
        "6d": 6,
    }[rot_key]
    return {
        "feature": f"root_pos_local+root_rot_local_{rotation_representation}_{root_rot_dim}+joint_dof_29+body_link_pos_local_29x3",
        "rotation_representation": rotation_representation,
        "split": args.split,
        "count": count,
        "num_prev_states": int(repr_cfg["num_prev_states"]),
        "canonical_frame_idx": int(repr_cfg["canonical_frame_idx"]),
        "sequence_length": int(repr_cfg["sequence_length"]),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "default_root_pos": default_qpos[:3].tolist(),
        "default_root_quat": default_root_quat.tolist(),
        "default_joint_dof": default_qpos[7:].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=Path("configs/generation/data/omg_data_lerobot.yaml"))
    parser.add_argument("--representation-config", type=Path, default=Path("configs/generation/representation/125d.yaml"))
    parser.add_argument("--paths-config", type=Path, default=Path("configs/generation/paths/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("assets/stats/g1_125d_stats.json"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episode-batch-frames", type=int, default=65536)
    parser.add_argument("--std-min", type=float, default=1e-6)
    args = parser.parse_args()

    args.rank = int(os.environ.get("RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.world_size > 1:
        if str(args.device).startswith("cuda"):
            args.device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="gloo")

    stats = compute_stats(args)
    if args.rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(stats, f)
            f.write("\n")
    if args.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
