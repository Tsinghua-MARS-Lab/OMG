from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from omg.data.episode_cache import EpisodeCachedG1MotionDataset
from omg.data.lerobot_dataset import LeRobotG1MotionDataset


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return value


def _resolve_dataset(
    data_config: Path,
    representation_config: Path,
    paths_config: Path,
    split: str,
) -> LeRobotG1MotionDataset:
    data = _load_yaml(data_config)
    representation = _load_yaml(representation_config)
    paths = _load_yaml(paths_config)
    split_configs = data.get("dataset_opts", {}).get(split, {})
    if len(split_configs) != 1:
        raise ValueError(f"Episode-cache materialization requires exactly one dataset for {split!r}")
    raw_config = next(iter(split_configs.values()))
    root = OmegaConf.create({"paths": paths, "representation": representation, "dataset": raw_config})
    OmegaConf.resolve(root)
    config = OmegaConf.to_container(root.dataset, resolve=True)
    if not isinstance(config, dict) or config.get("_target_") != "omg.data.lerobot_dataset.LeRobotG1MotionDataset":
        raise TypeError("Episode-cache materialization requires LeRobotG1MotionDataset input")
    config["train_window_policy"] = "exhaustive"
    config["train_window_stride"] = int(config.get("train_window_stride", 1))
    config.setdefault("rotation_representation", representation.get("rotation_representation", "rot6d"))
    dataset = instantiate(OmegaConf.create(config))
    if not isinstance(dataset, LeRobotG1MotionDataset):
        raise TypeError(f"Expected LeRobotG1MotionDataset, got {type(dataset).__name__}")
    return dataset


def write_episode_cache(
    dataset: LeRobotG1MotionDataset,
    *,
    output_root: Path,
    split: str,
    max_frames_per_shard: int,
    device: str,
    overwrite: bool,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    source_identity = {
        "repo_id": str(dataset.repo_id),
        "revision": str(dataset.revision or ""),
    }
    if not source_identity["repo_id"] or not source_identity["revision"]:
        raise ValueError("Episode-cache materialization requires a pinned LeRobot repo_id and revision")
    split_root = output_root / split
    incomplete_root = output_root / f".{split}.incomplete"
    if split_root.exists():
        if not overwrite:
            raise FileExistsError(f"Episode cache already exists: {split_root}")
        shutil.rmtree(split_root)
    if overwrite and incomplete_root.exists():
        shutil.rmtree(incomplete_root)
    (incomplete_root / "shards").mkdir(parents=True, exist_ok=True)
    source_identity_path = incomplete_root / "source_identity.json"
    if source_identity_path.exists():
        existing_identity = json.loads(source_identity_path.read_text(encoding="utf-8"))
        if existing_identity != source_identity:
            raise ValueError(
                "Cannot resume episode cache from a different LeRobot source: "
                f"existing={existing_identity!r} requested={source_identity!r}"
            )
    elif any((incomplete_root / "shards").iterdir()):
        raise ValueError(
            f"Cannot resume unpinned episode-cache shards without {source_identity_path}; rebuild with --overwrite"
        )
    else:
        source_identity_path.write_text(
            json.dumps(source_identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    episode_shards: list[int] = []
    episode_frame_offsets: list[int] = []
    episode_lengths: list[int] = []
    window_offsets = [0]
    captions: list[str] = []
    frame_count = 0
    shard_count = 0
    for shard_root in sorted((incomplete_root / "shards").glob("shard_*")):
        manifest_path = shard_root / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Incomplete episode-cache shard without manifest: {shard_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lengths = [int(value) for value in manifest["episode_lengths"]]
        shard_captions = [str(value) for value in manifest["captions"]]
        if len(lengths) != len(shard_captions):
            raise ValueError(f"Episode lengths/captions mismatch in {manifest_path}")
        local_offset = 0
        for length, caption in zip(lengths, shard_captions, strict=True):
            max_start = length - dataset.window_size
            windows = 1 if max_start <= 0 else (max_start // dataset.train_window_stride) + 1
            episode_shards.append(shard_count)
            episode_frame_offsets.append(local_offset)
            episode_lengths.append(length)
            window_offsets.append(window_offsets[-1] + windows)
            captions.append(caption)
            local_offset += length
            frame_count += length
        shard_count += 1
    completed_episodes = len(episode_lengths)
    remaining_episodes = None if max_episodes is None else max(0, int(max_episodes) - completed_episodes)
    iterator = dataset.iter_episode_kinematics_groups(
        max_frames=max_frames_per_shard,
        device=device,
        max_episodes=remaining_episodes,
        episode_start=completed_episodes,
    )
    for group in tqdm(iterator, desc=f"episode-cache:{split}"):
        shard_root = incomplete_root / "shards" / f"shard_{shard_count:05d}"
        shard_incomplete = incomplete_root / "shards" / f".shard_{shard_count:05d}.incomplete"
        if shard_incomplete.exists():
            shutil.rmtree(shard_incomplete)
        shard_incomplete.mkdir()
        frame_keys = ["qpos_36", "body_pos_w", "body_quat_w"]
        frame_keys.extend(key for key in ("audio_features", "has_audio", "human_motion", "has_human_motion") if key in group)
        for key in frame_keys:
            value = group[key].detach().cpu().numpy().astype(np.float32, copy=False)
            if key.startswith("has_"):
                value = value.astype(np.bool_, copy=False)
            np.save(shard_incomplete / f"{key}.npy", value)
        local_frame_offset = 0
        shard_lengths = []
        shard_captions = []
        for episode in group["episodes"]:
            length = int(episode["data_end_row"]) - int(episode["data_start_row"])
            max_start = length - dataset.window_size
            windows = 1 if max_start <= 0 else (max_start // dataset.train_window_stride) + 1
            episode_shards.append(shard_count)
            episode_frame_offsets.append(local_frame_offset)
            episode_lengths.append(length)
            window_offsets.append(window_offsets[-1] + windows)
            captions.append(str(episode.get("segment_caption", "")))
            shard_lengths.append(length)
            shard_captions.append(str(episode.get("segment_caption", "")))
            local_frame_offset += length
            frame_count += length
        (shard_incomplete / "manifest.json").write_text(
            json.dumps(
                {
                    "shard": shard_count,
                    "frames": local_frame_offset,
                    "episode_lengths": shard_lengths,
                    "captions": shard_captions,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(shard_incomplete, shard_root)
        shard_count += 1

    np.savez(
        incomplete_root / "episodes.npz",
        shard=np.asarray(episode_shards, dtype=np.int32),
        frame_offset=np.asarray(episode_frame_offsets, dtype=np.int64),
        length=np.asarray(episode_lengths, dtype=np.int32),
        window_offset=np.asarray(window_offsets, dtype=np.int64),
    )
    (incomplete_root / "captions.json").write_text(
        json.dumps(captions, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "format": EpisodeCachedG1MotionDataset.FORMAT,
        "source_repo_id": source_identity["repo_id"],
        "source_revision": source_identity["revision"],
        "split": split,
        "fps": dataset.default_fps,
        "window_size": dataset.window_size,
        "num_prev_states": dataset.num_prev_states,
        "canonical_frame_idx": dataset.codec.canonical_frame_idx,
        "rotation_representation": dataset.codec.rotation_representation,
        "train_window_stride": dataset.train_window_stride,
        "episodes": len(episode_lengths),
        "frames": frame_count,
        "windows": window_offsets[-1],
        "shards": shard_count,
        "max_frames_per_shard": int(max_frames_per_shard),
        "use_audio": bool(getattr(dataset, "use_audio", False)),
        "audio_dim": int(getattr(dataset, "audio_dim", 35)),
        "use_human_motion": bool(getattr(dataset, "use_human_motion", False)),
        "human_motion_dim": int(getattr(dataset, "human_motion_dim", 66)),
    }
    (incomplete_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(incomplete_root, split_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize exact frame-level G1 episode kinematics caches.")
    parser.add_argument("--data-config", type=Path, default=Path("configs/generation/data/omg_data_lerobot.yaml"))
    parser.add_argument(
        "--representation-config",
        type=Path,
        default=Path("configs/generation/representation/125d.yaml"),
    )
    parser.add_argument("--paths-config", type=Path, default=Path("configs/generation/paths/default.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-frames-per-shard", type=int, default=262144)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summaries = {}
    for split in args.splits:
        dataset = _resolve_dataset(args.data_config, args.representation_config, args.paths_config, split)
        summaries[split] = write_episode_cache(
            dataset,
            output_root=args.output_root,
            split=split,
            max_frames_per_shard=args.max_frames_per_shard,
            device=args.device,
            overwrite=args.overwrite,
            max_episodes=args.max_episodes,
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
