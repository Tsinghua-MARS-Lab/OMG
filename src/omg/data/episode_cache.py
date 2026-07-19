from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from omg.core.tensor import get_valid_mask, repeat_to_max_len
from omg.motion.feature_codec import G1MotionFeatureCodec
from omg.robots.g1.kinematics import G1Kinematics
from omg.utils.rotation_conversions import standardize_quaternion


class EpisodeCachedG1MotionDataset(Dataset):
    """Read exact windows from frame-level episode kinematics caches."""

    FORMAT = "omg.episode_cache.g1_motion.v2"

    def __init__(
        self,
        root: str | Path,
        split: str,
        source_repo_id: str,
        source_revision: str,
        limit_size: int | None = None,
        shard_cache_size: int = 4,
        episode_cache_size: int = 64,
        kinematics_path: str = "assets/robots/g1/g1_kinematics.json",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = str(split)
        self.split_root = self.root / self.split
        summary_path = self.split_root / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Episode-cache summary not found: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("format") != self.FORMAT:
            raise ValueError(f"Unsupported episode-cache format: {summary.get('format')!r}")
        self.source_repo_id = str(summary.get("source_repo_id", ""))
        self.source_revision = str(summary.get("source_revision", ""))
        expected_identity = (str(source_repo_id), str(source_revision))
        actual_identity = (self.source_repo_id, self.source_revision)
        if actual_identity != expected_identity:
            raise ValueError(
                "Episode cache LeRobot identity mismatch: "
                f"cache={actual_identity!r} expected={expected_identity!r}"
            )
        self.window_size = int(summary["window_size"])
        self.num_prev_states = int(summary["num_prev_states"])
        self.train_window_stride = int(summary["train_window_stride"])
        self.default_fps = float(summary["fps"])
        self.num_shards = int(summary["shards"])
        self.use_audio = bool(summary.get("use_audio", False))
        self.audio_dim = int(summary.get("audio_dim", 35))
        self.use_human_motion = bool(summary.get("use_human_motion", False))
        self.human_motion_dim = int(summary.get("human_motion_dim", 66))
        self.shard_cache_size = max(1, int(shard_cache_size))
        self.episode_cache_size = max(1, int(episode_cache_size))
        self.kinematics = G1Kinematics(kinematics_path=kinematics_path)
        self.codec = G1MotionFeatureCodec(
            self.kinematics,
            num_prev_states=self.num_prev_states,
            canonical_frame_idx=int(summary["canonical_frame_idx"]),
            rotation_representation=str(summary["rotation_representation"]),
        )

        with np.load(self.split_root / "episodes.npz") as episode_data:
            self.episode_shards = np.asarray(episode_data["shard"], dtype=np.int32)
            self.episode_frame_offsets = np.asarray(episode_data["frame_offset"], dtype=np.int64)
            self.episode_lengths = np.asarray(episode_data["length"], dtype=np.int32)
            self.window_offsets = np.asarray(episode_data["window_offset"], dtype=np.int64)
        if self.window_offsets.shape != (self.episode_lengths.shape[0] + 1,):
            raise ValueError("window_offset must contain one prefix-sum boundary per episode plus the final total")
        self.captions = json.loads((self.split_root / "captions.json").read_text(encoding="utf-8"))
        if len(self.captions) != int(self.episode_lengths.shape[0]):
            raise ValueError("captions and episode metadata have different lengths")
        self.num_samples = int(self.window_offsets[-1])
        if limit_size is not None:
            self.num_samples = min(self.num_samples, int(limit_size))
        self.uses_materialized_shards = True
        self.uses_exhaustive_train_windows = True
        self._shard_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._episode_cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return self.num_samples

    def _load_shard(self, shard_id: int) -> dict[str, np.ndarray]:
        cached = self._shard_cache.get(shard_id)
        if cached is not None:
            self._shard_cache.move_to_end(shard_id)
            return cached
        shard_root = self.split_root / "shards" / f"shard_{shard_id:05d}"
        loaded = {
            key: np.load(shard_root / f"{key}.npy", mmap_mode="r")
            for key in ("qpos_36", "body_pos_w", "body_quat_w")
        }
        optional_keys = []
        if self.use_audio:
            optional_keys.extend(("audio_features", "has_audio"))
        if self.use_human_motion:
            optional_keys.extend(("human_motion", "has_human_motion"))
        for key in optional_keys:
            loaded[key] = np.load(shard_root / f"{key}.npy", mmap_mode="r")
        self._shard_cache[shard_id] = loaded
        self._shard_cache.move_to_end(shard_id)
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return loaded

    def _load_episode(self, episode_index: int) -> dict[str, torch.Tensor]:
        cached = self._episode_cache.get(episode_index)
        if cached is not None:
            self._episode_cache.move_to_end(episode_index)
            return cached
        shard_id = int(self.episode_shards[episode_index])
        frame_offset = int(self.episode_frame_offsets[episode_index])
        total_len = int(self.episode_lengths[episode_index])
        shard = self._load_shard(shard_id)
        episode = {
            key: torch.from_numpy(np.array(value[frame_offset : frame_offset + total_len], copy=True))
            for key, value in shard.items()
        }
        episode["qpos_36"][..., 3:7] = standardize_quaternion(
            F.normalize(episode["qpos_36"][..., 3:7], dim=-1)
        )
        episode["body_quat_w"] = standardize_quaternion(F.normalize(episode["body_quat_w"], dim=-1))
        self._episode_cache[episode_index] = episode
        self._episode_cache.move_to_end(episode_index)
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return episode

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.num_samples
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        episode_index = bisect_right(self.window_offsets, index) - 1
        local_window = index - int(self.window_offsets[episode_index])
        return episode_index, local_window * self.train_window_stride

    def materialized_shard_spans(self, base_index: int = 0) -> list[tuple[int, int]]:
        spans = []
        for episode_index in range(self.episode_lengths.shape[0]):
            start = int(self.window_offsets[episode_index])
            end = min(int(self.window_offsets[episode_index + 1]), self.num_samples)
            if end > start:
                spans.append((int(base_index) + start, end - start))
            if end >= self.num_samples:
                break
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
        """Yield every cached window exactly once within the rank's contiguous partition."""
        del episode_batch_frames  # Episodes are already independently addressable in the cache.
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        rank = int(rank)
        world_size = int(world_size)
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid rank/world_size: {rank}/{world_size}")
        effective_total = self.num_samples if max_samples is None else min(self.num_samples, int(max_samples))
        rank_begin = effective_total * rank // world_size
        rank_end = effective_total * (rank + 1) // world_size
        if rank_begin == rank_end:
            return

        episode_begin = bisect_right(self.window_offsets, rank_begin) - 1
        episode_end = bisect_right(self.window_offsets, rank_end - 1)
        target_device = torch.device(device)
        frame_offsets = torch.arange(self.window_size, dtype=torch.long, device=target_device)
        history_offsets = (
            torch.arange(self.num_prev_states, dtype=torch.long, device=target_device) - self.num_prev_states
        )
        pending: dict[str, list[torch.Tensor]] = {
            "qpos_36": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "prev_qpos_36": [],
            "prev_body_pos_w": [],
            "prev_body_quat_w": [],
            "valid_mask": [],
        }
        pending_count = 0
        produced = 0
        sample_limit = rank_end - rank_begin

        def encode_pending() -> dict[str, torch.Tensor]:
            values = {key: torch.cat(parts, dim=0) for key, parts in pending.items()}
            fps = torch.full(
                (values["qpos_36"].shape[0],),
                self.default_fps,
                dtype=torch.float32,
                device=target_device,
            )
            _, canon_root_pos, canon_root_quat = self.codec.prev_state_features_from_history(
                values["prev_qpos_36"],
                values["prev_body_pos_w"],
                values["prev_body_quat_w"],
                fps=fps,
            )
            components = self.codec.canonicalize(
                values["qpos_36"],
                values["body_pos_w"],
                values["body_quat_w"],
                anchor_root_pos=canon_root_pos,
                anchor_root_quat=canon_root_quat,
                fps=fps,
                valid_mask=values["valid_mask"],
            )
            return {
                "motion_features": self.codec.assemble_features(components),
                "qpos_36": values["qpos_36"],
                "valid_mask": values["valid_mask"],
            }

        for episode_index in range(episode_begin, episode_end):
            episode = {
                key: value.to(target_device)
                for key, value in self._load_episode(episode_index).items()
                if key in {"qpos_36", "body_pos_w", "body_quat_w"}
            }
            total_len = int(self.episode_lengths[episode_index])
            window_count = int(self.window_offsets[episode_index + 1] - self.window_offsets[episode_index])
            starts = torch.arange(window_count, dtype=torch.long, device=target_device) * self.train_window_stride
            if episode_index == episode_begin:
                starts = starts[rank_begin - int(self.window_offsets[episode_index]) :]
            starts = starts[: sample_limit - produced - pending_count]
            start_cursor = 0
            while start_cursor < int(starts.numel()):
                take = min(batch_size - pending_count, int(starts.numel()) - start_cursor)
                batch_starts = starts[start_cursor : start_cursor + take]
                frame_indices = batch_starts[:, None] + frame_offsets[None, :]
                valid_mask = frame_indices < total_len
                frame_indices = frame_indices.clamp(max=total_len - 1)
                history_indices = (batch_starts[:, None] + history_offsets[None, :]).clamp(
                    min=0,
                    max=total_len - 1,
                )
                pending["qpos_36"].append(episode["qpos_36"][frame_indices])
                pending["body_pos_w"].append(episode["body_pos_w"][frame_indices])
                pending["body_quat_w"].append(episode["body_quat_w"][frame_indices])
                pending["prev_qpos_36"].append(episode["qpos_36"][history_indices])
                pending["prev_body_pos_w"].append(episode["body_pos_w"][history_indices])
                pending["prev_body_quat_w"].append(episode["body_quat_w"][history_indices])
                pending["valid_mask"].append(valid_mask)
                pending_count += take
                start_cursor += take
                if pending_count == batch_size:
                    yield encode_pending()
                    produced += pending_count
                    pending = {key: [] for key in pending}
                    pending_count = 0
            if produced + pending_count >= sample_limit:
                break

        if pending_count:
            yield encode_pending()

    @staticmethod
    def _pad(values: torch.Tensor, window_size: int) -> torch.Tensor:
        if values.shape[0] >= window_size:
            return values[:window_size]
        return repeat_to_max_len(values, window_size, dim=0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, start = self._locate(int(index))
        total_len = int(self.episode_lengths[episode_index])
        valid_len = min(self.window_size, total_len - start)
        end = start + valid_len
        episode = self._load_episode(episode_index)
        qpos_all = episode["qpos_36"]
        body_pos_all = episode["body_pos_w"]
        body_quat_all = episode["body_quat_w"]
        history_indices = torch.arange(start - self.num_prev_states, start, dtype=torch.long).clamp(
            min=0,
            max=total_len - 1,
        )
        qpos_36 = qpos_all[start:end]
        body_pos_w = body_pos_all[start:end]
        body_quat_w = body_quat_all[start:end]
        fps = torch.tensor([self.default_fps], dtype=torch.float32)
        history_features, canon_root_pos, canon_root_quat = self.codec.prev_state_features_from_history(
            qpos_all[history_indices].unsqueeze(0),
            body_pos_all[history_indices].unsqueeze(0),
            body_quat_all[history_indices].unsqueeze(0),
            fps=fps,
        )
        components = self.codec.canonicalize(
            qpos_36.unsqueeze(0),
            body_pos_w.unsqueeze(0),
            body_quat_w.unsqueeze(0),
            anchor_root_pos=canon_root_pos,
            anchor_root_quat=canon_root_quat,
            fps=fps,
            valid_mask=torch.ones(1, valid_len, dtype=torch.bool),
        )
        caption = str(self.captions[episode_index])
        valid_mask = get_valid_mask(self.window_size, valid_len)
        audio_features = torch.zeros(self.window_size, self.audio_dim, dtype=torch.float32)
        has_audio = torch.zeros(self.window_size, dtype=torch.bool)
        if self.use_audio:
            audio_features[:valid_len] = episode["audio_features"][start:end]
            has_audio[:valid_len] = episode["has_audio"][start:end]
        human_motion = torch.zeros(self.window_size, self.human_motion_dim, dtype=torch.float32)
        has_human_motion = torch.zeros(self.window_size, dtype=torch.bool)
        if self.use_human_motion:
            human_motion[:valid_len] = episode["human_motion"][start:end]
            has_human_motion[:valid_len] = episode["has_human_motion"][start:end]
        return {
            "length": torch.tensor(valid_len, dtype=torch.long),
            "fps": torch.tensor(self.default_fps, dtype=torch.float32),
            "qpos_36": self._pad(qpos_36, self.window_size),
            "body_pos_w": self._pad(body_pos_w, self.window_size),
            "body_quat_w": self._pad(body_quat_w, self.window_size),
            "audio_features": audio_features,
            "human_motion": human_motion,
            "motion_features": self._pad(self.codec.assemble_features(components)[0], self.window_size),
            "root_pos_local": self._pad(components.root_pos_local[0], self.window_size),
            "root_rot_local_quat": self._pad(components.root_rot_local_quat[0], self.window_size),
            "joint_dof": self._pad(components.joint_dof[0], self.window_size),
            "body_link_pos_local": self._pad(components.body_link_pos_local[0], self.window_size),
            "prev_state_features": history_features[0],
            "history_features": history_features[0],
            "canonical_frame_idx": torch.tensor(self.codec.canonical_frame_idx, dtype=torch.long),
            "canon_root_pos": canon_root_pos[0],
            "canon_root_quat": canon_root_quat[0],
            "caption": caption,
            "has_text": torch.tensor(bool(caption.strip()), dtype=torch.bool),
            "mask": {
                "valid": valid_mask,
                "has_audio": has_audio,
                "has_human_motion": has_human_motion,
            },
            "meta": {
                "episode_index": episode_index,
                "window_start": start,
                "split": self.split,
            },
        }
