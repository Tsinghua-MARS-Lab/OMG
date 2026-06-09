from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
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
    if target == "omg.data.g1_motion.G1MotionDataset":
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
    values = values.float()
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
    qpos_sum = torch.zeros(36, dtype=torch.float64)
    qpos_count = 0

    for name, raw_cfg in datasets.items():
        cfg = _resolve_dataset_cfg(raw_cfg, repr_cfg, paths_cfg)
        dataset = instantiate(OmegaConf.create(cfg))
        limit = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
        iterator = tqdm(range(limit), desc=name)
        for idx in iterator:
            sample = dataset[idx]
            valid = sample["mask"]["valid"].bool()
            features = sample["motion_features"][valid]
            qpos = sample["qpos_36"][valid]
            if features.numel() == 0:
                continue
            count, mean, m2 = _update_moments(count, mean, m2, features)
            qpos_sum += qpos.double().sum(dim=0)
            qpos_count += int(qpos.shape[0])

    if mean is None or m2 is None or count == 0 or qpos_count == 0:
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
    parser.add_argument("--data-config", type=Path, default=Path("configs/generation/data/omg_data.yaml"))
    parser.add_argument("--representation-config", type=Path, default=Path("configs/generation/representation/125d.yaml"))
    parser.add_argument("--paths-config", type=Path, default=Path("configs/generation/paths/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("assets/stats/g1_125d_stats.json"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--std-min", type=float, default=1e-6)
    args = parser.parse_args()

    stats = compute_stats(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(stats, f)
        f.write("\n")


if __name__ == "__main__":
    main()
