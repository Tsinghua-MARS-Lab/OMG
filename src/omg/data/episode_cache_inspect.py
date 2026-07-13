from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from omg.data.episode_cache import EpisodeCachedG1MotionDataset


def inspect_episode_cache(root: str | Path, split: str) -> dict[str, Any]:
    root = Path(root)
    split_root = root / split
    errors: list[str] = []
    try:
        summary = json.loads((split_root / "summary.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return {"root": str(root.resolve()), "split": split, "valid": False, "errors": [str(exc)]}
    if summary.get("format") != EpisodeCachedG1MotionDataset.FORMAT:
        errors.append(f"unexpected format: {summary.get('format')!r}")
    try:
        with np.load(split_root / "episodes.npz") as episode_data:
            episode_shards = np.asarray(episode_data["shard"])
            frame_offsets = np.asarray(episode_data["frame_offset"])
            lengths = np.asarray(episode_data["length"])
            window_offsets = np.asarray(episode_data["window_offset"])
        captions = json.loads((split_root / "captions.json").read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(str(exc))
        episode_shards = frame_offsets = lengths = window_offsets = np.asarray([])
        captions = []
    episodes = int(lengths.shape[0])
    if episode_shards.shape != (episodes,) or frame_offsets.shape != (episodes,):
        errors.append("episode shard/frame-offset arrays have inconsistent shapes")
    if window_offsets.shape != (episodes + 1,):
        errors.append("window offsets do not contain episodes + 1 entries")
    if len(captions) != episodes:
        errors.append("caption count does not match episode count")
    if episodes and (lengths <= 0).any():
        errors.append("episode lengths must be positive")
    if window_offsets.size and ((np.diff(window_offsets) <= 0).any() or int(window_offsets[0]) != 0):
        errors.append("window offsets must be a strictly increasing prefix sum starting at zero")

    frame_count = 0
    shard_count = int(summary.get("shards", 0))
    observed_episode = 0
    for shard_id in range(shard_count):
        shard_root = split_root / "shards" / f"shard_{shard_id:05d}"
        try:
            manifest = json.loads((shard_root / "manifest.json").read_text(encoding="utf-8"))
            arrays = {
                key: np.load(shard_root / f"{key}.npy", mmap_mode="r")
                for key in ("qpos_36", "body_pos_w", "body_quat_w")
            }
        except Exception as exc:
            errors.append(f"shard {shard_id}: {exc}")
            continue
        rows = int(arrays["qpos_36"].shape[0])
        if arrays["qpos_36"].shape[1:] != (36,):
            errors.append(f"shard {shard_id}: qpos_36 shape {arrays['qpos_36'].shape}")
        if arrays["body_pos_w"].shape != (rows, 30, 3):
            errors.append(f"shard {shard_id}: body_pos_w shape {arrays['body_pos_w'].shape}")
        if arrays["body_quat_w"].shape != (rows, 30, 4):
            errors.append(f"shard {shard_id}: body_quat_w shape {arrays['body_quat_w'].shape}")
        if int(manifest.get("frames", -1)) != rows:
            errors.append(f"shard {shard_id}: manifest frames do not match arrays")
        shard_lengths = np.asarray(manifest.get("episode_lengths", []), dtype=np.int64)
        shard_captions = manifest.get("captions", [])
        if len(shard_captions) != shard_lengths.size:
            errors.append(f"shard {shard_id}: episode lengths/captions mismatch")
        if int(shard_lengths.sum()) != rows:
            errors.append(f"shard {shard_id}: episode lengths do not sum to frame count")
        shard_episode_count = int(shard_lengths.size)
        episode_end = observed_episode + shard_episode_count
        if episode_end > episodes:
            errors.append(f"shard {shard_id}: manifest contains more episodes than index")
        else:
            indexed_shards = episode_shards[observed_episode:episode_end]
            indexed_offsets = frame_offsets[observed_episode:episode_end]
            indexed_lengths = lengths[observed_episode:episode_end]
            expected_offsets = np.cumsum(np.r_[0, shard_lengths[:-1]])
            if not np.all(indexed_shards == shard_id):
                errors.append(f"shard {shard_id}: episode index points to another shard")
            if not np.array_equal(indexed_offsets, expected_offsets):
                errors.append(f"shard {shard_id}: episode frame offsets are not contiguous")
            if not np.array_equal(indexed_lengths, shard_lengths):
                errors.append(f"shard {shard_id}: episode lengths differ from manifest")
            if captions[observed_episode:episode_end] != [str(value) for value in shard_captions]:
                errors.append(f"shard {shard_id}: captions differ from manifest")
        observed_episode = episode_end

        optional_specs = []
        if bool(summary.get("use_audio", False)):
            optional_specs.extend(
                (("audio_features", (rows, int(summary.get("audio_dim", -1)))), ("has_audio", (rows,)))
            )
        if bool(summary.get("use_human_motion", False)):
            optional_specs.extend(
                (
                    ("human_motion", (rows, int(summary.get("human_motion_dim", -1)))),
                    ("has_human_motion", (rows,)),
                )
            )
        for key, expected_shape in optional_specs:
            try:
                value = np.load(shard_root / f"{key}.npy", mmap_mode="r")
            except Exception as exc:
                errors.append(f"shard {shard_id}: {key}: {exc}")
                continue
            if value.shape != expected_shape:
                errors.append(f"shard {shard_id}: {key} shape {value.shape}")
        frame_count += rows
    if observed_episode != episodes:
        errors.append(f"shard manifests contain {observed_episode} episodes, index contains {episodes}")
    windows = int(window_offsets[-1]) if window_offsets.size else 0
    for key, actual in (("episodes", episodes), ("frames", frame_count), ("windows", windows)):
        if int(summary.get(key, -1)) != actual:
            errors.append(f"summary {key}={summary.get(key)!r}, actual={actual}")
    if episodes:
        for episode_index in range(episodes):
            shard_id = int(episode_shards[episode_index])
            if shard_id < 0 or shard_id >= shard_count:
                errors.append(f"episode {episode_index}: invalid shard {shard_id}")
                break
    return {
        "root": str(root.resolve()),
        "split": split,
        "format": summary.get("format"),
        "episodes": episodes,
        "frames": frame_count,
        "windows": windows,
        "shards": shard_count,
        "valid": not errors,
        "errors": errors,
    }
