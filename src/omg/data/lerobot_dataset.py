from __future__ import annotations

from pathlib import Path
import json
from collections.abc import Iterator
from bisect import bisect_left, bisect_right
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from omg.core.tensor import get_valid_mask, repeat_to_max_len
from omg.data.windowing import ExhaustiveWindowSampleView, is_exhaustive_train_window_policy
from omg.motion.feature_codec import G1MotionFeatureCodec
from omg.robots.g1.kinematics import G1Kinematics
from omg.utils.rotation_conversions import standardize_quaternion


LEROBOT_REPO_ID = "THU-MARS/OMG-Data"


def _load_dataset_runtime():
    try:
        from datasets import Dataset as HFDataset
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "LeRobot input requires the OMG data dependencies. Install with `pip install -e '.[data]'`."
        ) from exc
    return HFDataset, snapshot_download


def _resolve_root(dataset_root: str | Path | None, *, repo_id: str, revision: str | None) -> Path:
    if dataset_root is not None:
        root = Path(dataset_root).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"LeRobot dataset root does not exist: {root}")
        return root
    _, snapshot_download = _load_dataset_runtime()
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision))


def _parse_split_range(value: str) -> tuple[int, int]:
    parts = str(value).split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected LeRobot split range `start:end`, got {value!r}")
    return int(parts[0]), int(parts[1])


class LeRobotG1MotionDataset(Dataset):
    """Adapt an official LeRobotDataset v3 dataset to OMG training samples."""

    def __init__(
        self,
        dataset_root: str | Path | None,
        split: str,
        repo_id: str = LEROBOT_REPO_ID,
        revision: str | None = None,
        sequence_duration: float = 2.0,
        fps: float = 30.0,
        canonical_frame_idx: int | None = None,
        num_prev_states: int = 10,
        limit_size: int | None = None,
        kinematics_path: str = "assets/robots/g1/g1_kinematics.json",
        use_text: bool = True,
        use_audio: bool = False,
        audio_dim: int = 35,
        use_human_motion: bool = False,
        human_motion_dim: int = 66,
        rotation_representation: str = "rot6d",
        train_window_policy: str | None = None,
        train_window_stride: int | None = None,
        exhaustive_train_slicing: bool = True,
        eval_window_policy: str = "uniform",
        eval_num_windows: int = 3,
    ) -> None:
        self.repo_id = str(repo_id)
        self.revision = None if revision is None else str(revision)
        self.dataset_root = _resolve_root(dataset_root, repo_id=self.repo_id, revision=revision)
        self.split = str(split)
        self.sequence_duration = float(sequence_duration)
        self.default_fps = float(fps)
        self.num_prev_states = int(num_prev_states)
        self.limit_size = None if limit_size is None else int(limit_size)
        self.use_text = bool(use_text)
        self.use_audio = bool(use_audio)
        self.audio_dim = int(audio_dim)
        self.use_human_motion = bool(use_human_motion)
        self.human_motion_dim = int(human_motion_dim)
        self.training = self.split == "train"
        self.eval_window_policy = str(eval_window_policy)
        self.eval_num_windows = max(int(eval_num_windows), 1)
        self.window_size = int(round(self.sequence_duration * self.default_fps))
        if self.window_size <= 0:
            raise ValueError(f"Invalid sequence_duration={sequence_duration} for fps={fps}")
        if train_window_policy is None:
            train_window_policy = "exhaustive" if exhaustive_train_slicing else "random"
        self.train_window_policy = str(train_window_policy)
        self.train_window_stride = int(1 if train_window_stride is None else train_window_stride)
        if self.train_window_stride <= 0:
            raise ValueError(f"train_window_stride must be positive, got {self.train_window_stride}")

        info_path = self.dataset_root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        dataset_fps = float(info["fps"])
        if abs(dataset_fps - self.default_fps) > 1e-4:
            raise ValueError(f"Configured fps={self.default_fps} does not match LeRobot dataset fps={dataset_fps}")
        required_features = {"observation.state", "action"}
        missing = required_features - set(info["features"])
        if missing:
            raise ValueError(f"LeRobot dataset is missing required OMG features: {sorted(missing)}")
        state_shape = tuple(info["features"]["observation.state"]["shape"])
        if state_shape != (36,):
            raise ValueError(f"Expected observation.state shape (36,), got {state_shape}")
        if self.use_audio and "omg.audio.feature" not in info["features"]:
            raise ValueError("use_audio=true but LeRobot dataset has no `omg.audio.feature`")
        if self.use_human_motion and "omg.humanref.motion" not in info["features"]:
            raise ValueError("use_human_motion=true but LeRobot dataset has no `omg.humanref.motion`")

        HFDataset, _ = _load_dataset_runtime()
        data_files = sorted((self.dataset_root / "data").glob("chunk-*/*.parquet"))
        episode_files = sorted((self.dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        if not data_files or not episode_files:
            raise FileNotFoundError(f"Incomplete LeRobot v3 dataset under {self.dataset_root}")
        self.frame_dataset = HFDataset.from_parquet([str(path) for path in data_files])
        self.episode_dataset = HFDataset.from_parquet([str(path) for path in episode_files])

        self.kinematics = G1Kinematics(kinematics_path=kinematics_path)
        self.codec = G1MotionFeatureCodec(
            self.kinematics,
            num_prev_states=self.num_prev_states,
            canonical_frame_idx=canonical_frame_idx,
            rotation_representation=rotation_representation,
        )
        self.episodes = self._load_episodes(info)
        if self.training and is_exhaustive_train_window_policy(self.train_window_policy):
            self.samples = ExhaustiveWindowSampleView(
                self.episodes,
                window_size=self.window_size,
                stride=self.train_window_stride,
            )
            self.uses_exhaustive_train_windows = True
        else:
            self.samples = self._build_eval_samples(self.episodes) if not self.training else self.episodes
            self.uses_exhaustive_train_windows = False
        self._cached_episode_index: int | None = None
        self._cached_episode_data: dict[str, np.ndarray] | None = None
        self._cached_episode_kinematics: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._cached_episode_kinematics_device: torch.device | None = None
        print(
            f"[INFO] LeRobotG1MotionDataset repo_id={self.repo_id} root={self.dataset_root} "
            f"split={self.split} episodes={len(self.episodes)} samples={len(self.samples)} "
            f"train_window_policy={self.train_window_policy} train_window_stride={self.train_window_stride}"
        )

    def _load_episodes(self, info: dict[str, Any]) -> list[dict[str, Any]]:
        frame = self.episode_dataset.to_pandas()
        if "omg/split" in frame.columns:
            frame = frame.loc[frame["omg/split"] == self.split]
        else:
            split_ranges = info.get("splits", {})
            if self.split not in split_ranges:
                raise ValueError(f"Split {self.split!r} not found in LeRobot metadata")
            start, end = _parse_split_range(split_ranges[self.split])
            frame = frame.loc[(frame["episode_index"] >= start) & (frame["episode_index"] < end)]
        episodes: list[dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            tasks = row.get("tasks")
            if isinstance(tasks, np.ndarray):
                tasks = tasks.tolist()
            tasks = list(tasks or [])
            length = int(row["length"])
            episodes.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "data_start_row": int(row["dataset_from_index"]),
                    "data_end_row": int(row["dataset_to_index"]),
                    "segment_frame_start": 0,
                    "segment_frame_end": length,
                    "segment_caption": str(tasks[0]) if tasks else "",
                    "segment_action": str(tasks[0]) if tasks else "",
                    "segment_style": "",
                    "video_summary": str(tasks[0]) if tasks else "",
                    "sequence_name": str(row.get("omg/source_id", row["episode_index"])),
                    "source_dataset": str(row.get("omg/dataset", "")),
                    "segment_index": int(row.get("omg/segment_index", 0)),
                    "source_start_frame": int(row.get("omg/source_start_frame", 0)),
                    "source_end_frame": int(row.get("omg/source_end_frame", length)),
                    "has_text": bool(row.get("omg/has_text", bool(tasks and str(tasks[0]).strip()))),
                    "has_audio": bool(row.get("omg/has_audio", False)),
                    "has_humanref": bool(row.get("omg/has_humanref", False)),
                    "fps": self.default_fps,
                    "split": self.split,
                }
            )
        if not episodes:
            raise ValueError(f"No LeRobot episodes for split {self.split!r}")
        return episodes

    def _build_eval_samples(self, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        samples = []
        for episode in episodes:
            total_len = int(episode["segment_frame_end"])
            max_offset = total_len - self.window_size
            if max_offset <= 0 or self.eval_window_policy == "single":
                starts = [0]
            elif self.eval_window_policy == "uniform":
                starts = [0] if self.eval_num_windows == 1 else [
                    int(round(value)) for value in np.linspace(0, max_offset, num=self.eval_num_windows)
                ]
            else:
                raise ValueError(f"Unsupported eval_window_policy={self.eval_window_policy}")
            starts = sorted(set(starts))
            for window_index, start in enumerate(starts):
                samples.append(
                    {
                        **episode,
                        "fixed_window_start": start,
                        "eval_window_index": window_index,
                        "eval_num_windows": len(starts),
                    }
                )
        return samples

    def __len__(self) -> int:
        size = len(self.samples)
        return min(self.limit_size, size) if self.limit_size is not None else size

    def _read_episode(self, sample: dict[str, Any]) -> dict[str, np.ndarray]:
        episode_index = int(sample["episode_index"])
        if self._cached_episode_index == episode_index and self._cached_episode_data is not None:
            return self._cached_episode_data
        start = int(sample["data_start_row"])
        end = int(sample["data_end_row"])
        columns = ["observation.state"]
        if self.use_audio:
            columns.extend(("omg.audio.feature", "omg.condition.has_audio"))
        if self.use_human_motion:
            columns.extend(("omg.humanref.motion", "omg.condition.has_humanref"))
        raw = self.frame_dataset.select_columns(columns)[start:end]
        data = {name: np.asarray(raw[name]) for name in columns}
        qpos = np.asarray(data["observation.state"], dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != 36:
            raise ValueError(f"Expected episode observation.state shape (T, 36), got {tuple(qpos.shape)}")
        self._cached_episode_index = episode_index
        self._cached_episode_data = data
        self._cached_episode_kinematics = None
        self._cached_episode_kinematics_device = None
        return data

    def _prepare_episode_kinematics(
        self,
        sample: dict[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor]:
        episode = self._read_episode(sample)
        target_device = torch.device(device)
        if self._cached_episode_kinematics is None or self._cached_episode_kinematics_device != target_device:
            self.kinematics.to(target_device)
            qpos_all_np = np.asarray(episode["observation.state"], dtype=np.float32)
            qpos_all = torch.from_numpy(np.array(qpos_all_np, dtype=np.float32, copy=True)).to(target_device)
            qpos_all[..., 3:7] = standardize_quaternion(F.normalize(qpos_all[..., 3:7], dim=-1))
            fk = self.kinematics.forward_kinematics(qpos_all)
            body_pos_all = fk["body_pos_w"]
            body_quat_all = standardize_quaternion(F.normalize(fk["body_quat_w"], dim=-1))
            self._cached_episode_kinematics = (qpos_all, body_pos_all, body_quat_all)
            self._cached_episode_kinematics_device = target_device
        qpos_all, body_pos_all, body_quat_all = self._cached_episode_kinematics
        return episode, qpos_all, body_pos_all, body_quat_all

    def _get_window(self, sample: dict[str, Any], total_len: int) -> tuple[int, int]:
        if total_len <= self.window_size:
            return 0, total_len
        fixed_window_start = sample.get("fixed_window_start")
        if fixed_window_start is not None:
            start = max(0, min(int(fixed_window_start), total_len - self.window_size))
            return start, self.window_size
        if self.training:
            start = int(torch.randint(0, total_len - self.window_size + 1, (1,)).item())
            return start, self.window_size
        raise KeyError("fixed_window_start is required for evaluation samples")

    def sample_locator(self, idx: int) -> dict[str, Any]:
        """Return the stable LeRobot identity for one deterministic dataset sample."""
        sample = self.samples[int(idx)]
        total_len = int(sample["data_end_row"]) - int(sample["data_start_row"])
        start, valid_len = self._get_window(sample, total_len)
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "split": self.split,
            "episode_index": int(sample["episode_index"]),
            "window_start": int(start),
            "num_frames": int(self.window_size),
            "valid_frames": int(valid_len),
            "source_dataset": str(sample["source_dataset"]),
            "source_id": str(sample["sequence_name"]),
            "segment_index": int(sample.get("segment_index", 0)),
            "source_start_frame": int(sample.get("source_start_frame", 0)),
            "source_end_frame": int(sample.get("source_end_frame", total_len)),
        }

    def sample_has_condition(self, idx: int, condition: str, *, num_frames: int) -> bool:
        """Check exact full-window condition availability without running FK."""
        required = int(num_frames)
        if required <= 0:
            raise ValueError("num_frames must be positive")
        sample = self.samples[int(idx)]
        locator = self.sample_locator(int(idx))
        if condition == "text":
            return bool(self.use_text and sample.get("has_text", False) and str(sample.get("segment_caption", "")).strip())
        if int(locator["valid_frames"]) < required:
            return False
        columns = {
            "audio": (self.use_audio, "omg.condition.has_audio"),
            "humanref": (self.use_human_motion, "omg.condition.has_humanref"),
        }
        if condition not in columns:
            raise ValueError(f"Unsupported condition={condition!r}")
        enabled, column = columns[condition]
        if not enabled:
            return False
        episode = self._read_episode(sample)
        start = int(locator["window_start"])
        mask = np.asarray(episode[column][start : start + required], dtype=np.bool_).reshape(-1)
        return bool(mask.shape[0] == required and mask.all())

    def _get_prev_indices(self, start: int, total_len: int) -> torch.Tensor:
        return torch.tensor(
            [max(0, min(start - self.num_prev_states + index, total_len - 1)) for index in range(self.num_prev_states)],
            dtype=torch.long,
        )

    def _pad_or_trim(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] >= self.window_size:
            return values[: self.window_size]
        return repeat_to_max_len(values, self.window_size, dim=0)

    @staticmethod
    def _condition_mask(raw: np.ndarray | None, *, start: int, end: int, valid_len: int, window_size: int) -> torch.Tensor:
        mask = torch.zeros(window_size, dtype=torch.bool)
        if raw is not None:
            values = torch.as_tensor(raw[start:end], dtype=torch.bool)
            mask[:valid_len] = values[:valid_len]
        return mask

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
        """Yield exact exhaustive-window features with grouped episode FK."""
        if not self.training or not self.uses_exhaustive_train_windows:
            raise ValueError("Batched stats require an exhaustive train-window dataset")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        episode_batch_frames = int(episode_batch_frames)
        if episode_batch_frames <= 0:
            raise ValueError(f"episode_batch_frames must be positive, got {episode_batch_frames}")
        rank = int(rank)
        world_size = int(world_size)
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid rank/world_size: {rank}/{world_size}")
        window_offsets = [0]
        for episode in self.episodes:
            episode_len = int(episode["data_end_row"]) - int(episode["data_start_row"])
            max_start = episode_len - self.window_size
            count = 1 if max_start <= 0 else (max_start // self.train_window_stride) + 1
            window_offsets.append(window_offsets[-1] + count)
        total_windows = window_offsets[-1]
        effective_total = total_windows if max_samples is None else min(total_windows, int(max_samples))
        rank_window_begin = effective_total * rank // world_size
        rank_window_end = effective_total * (rank + 1) // world_size
        episode_begin = max(0, bisect_right(window_offsets, rank_window_begin) - 1)
        episode_end = bisect_left(window_offsets, rank_window_end)
        episode_begin = min(episode_begin, len(self.episodes))
        episode_end = len(self.episodes) if rank == world_size - 1 else min(episode_end, len(self.episodes))
        selected_episodes = self.episodes[episode_begin:episode_end]
        first_episode_window_skip = rank_window_begin - window_offsets[episode_begin]
        sample_limit = rank_window_end - rank_window_begin
        produced = 0
        target_device = torch.device(device)
        self.kinematics.to(target_device)
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

        episode_cursor = 0
        while episode_cursor < len(selected_episodes) and produced + pending_count < sample_limit:
            group_start = episode_cursor
            grouped_frames = 0
            while episode_cursor < len(selected_episodes):
                episode_frames = int(selected_episodes[episode_cursor]["data_end_row"]) - int(
                    selected_episodes[episode_cursor]["data_start_row"]
                )
                if grouped_frames and grouped_frames + episode_frames > episode_batch_frames:
                    break
                grouped_frames += episode_frames
                episode_cursor += 1
            episode_group = selected_episodes[group_start:episode_cursor]
            data_start = int(episode_group[0]["data_start_row"])
            data_end = int(episode_group[-1]["data_end_row"])
            raw = self.frame_dataset.select_columns(["observation.state"])[data_start:data_end]
            qpos_group = torch.as_tensor(
                np.asarray(raw["observation.state"], dtype=np.float32),
                dtype=torch.float32,
                device=target_device,
            )
            qpos_group[..., 3:7] = standardize_quaternion(F.normalize(qpos_group[..., 3:7], dim=-1))
            fk = self.kinematics.forward_kinematics(qpos_group)
            body_pos_group = fk["body_pos_w"]
            body_quat_group = standardize_quaternion(F.normalize(fk["body_quat_w"], dim=-1))

            for episode_info in episode_group:
                local_start = int(episode_info["data_start_row"]) - data_start
                local_end = int(episode_info["data_end_row"]) - data_start
                qpos_all = qpos_group[local_start:local_end]
                body_pos_all = body_pos_group[local_start:local_end]
                body_quat_all = body_quat_group[local_start:local_end]
                total_len = int(qpos_all.shape[0])
                max_start = total_len - self.window_size
                if max_start <= 0:
                    starts = torch.zeros(1, dtype=torch.long, device=target_device)
                else:
                    starts = torch.arange(
                        0,
                        max_start + 1,
                        self.train_window_stride,
                        dtype=torch.long,
                        device=target_device,
                    )
                if episode_info is selected_episodes[0] and first_episode_window_skip:
                    starts = starts[first_episode_window_skip:]
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
                    pending["qpos_36"].append(qpos_all[frame_indices])
                    pending["body_pos_w"].append(body_pos_all[frame_indices])
                    pending["body_quat_w"].append(body_quat_all[frame_indices])
                    pending["prev_qpos_36"].append(qpos_all[history_indices])
                    pending["prev_body_pos_w"].append(body_pos_all[history_indices])
                    pending["prev_body_quat_w"].append(body_quat_all[history_indices])
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

    def iter_episode_kinematics_groups(
        self,
        *,
        max_frames: int,
        device: torch.device | str = "cpu",
        max_episodes: int | None = None,
        episode_start: int = 0,
    ) -> Iterator[dict[str, Any]]:
        """Yield contiguous episodes after one grouped FK evaluation."""
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        target_device = torch.device(device)
        self.kinematics.to(target_device)
        episode_start = int(episode_start)
        if episode_start < 0 or episode_start > len(self.episodes):
            raise ValueError(f"Invalid episode_start={episode_start}")
        episode_limit = len(self.episodes) if max_episodes is None else min(
            len(self.episodes),
            episode_start + int(max_episodes),
        )
        episode_cursor = episode_start
        while episode_cursor < episode_limit:
            group_start = episode_cursor
            grouped_frames = 0
            while episode_cursor < episode_limit:
                episode = self.episodes[episode_cursor]
                episode_frames = int(episode["data_end_row"]) - int(episode["data_start_row"])
                if grouped_frames and grouped_frames + episode_frames > max_frames:
                    break
                grouped_frames += episode_frames
                episode_cursor += 1
            episode_group = self.episodes[group_start:episode_cursor]
            data_start = int(episode_group[0]["data_start_row"])
            data_end = int(episode_group[-1]["data_end_row"])
            columns = ["observation.state"]
            if self.use_audio:
                columns.extend(("omg.audio.feature", "omg.condition.has_audio"))
            if self.use_human_motion:
                columns.extend(("omg.humanref.motion", "omg.condition.has_humanref"))
            raw = self.frame_dataset.select_columns(columns)[data_start:data_end]
            qpos_36 = torch.as_tensor(
                np.asarray(raw["observation.state"], dtype=np.float32),
                dtype=torch.float32,
                device=target_device,
            )
            qpos_36[..., 3:7] = standardize_quaternion(F.normalize(qpos_36[..., 3:7], dim=-1))
            fk = self.kinematics.forward_kinematics(qpos_36)
            group = {
                "episodes": episode_group,
                "qpos_36": qpos_36,
                "body_pos_w": fk["body_pos_w"],
                "body_quat_w": standardize_quaternion(F.normalize(fk["body_quat_w"], dim=-1)),
            }
            if self.use_audio:
                group["audio_features"] = torch.as_tensor(
                    np.asarray(raw["omg.audio.feature"], dtype=np.float32),
                    device=target_device,
                )
                group["has_audio"] = torch.as_tensor(
                    np.asarray(raw["omg.condition.has_audio"], dtype=np.bool_),
                    device=target_device,
                )
            if self.use_human_motion:
                group["human_motion"] = torch.as_tensor(
                    np.asarray(raw["omg.humanref.motion"], dtype=np.float32),
                    device=target_device,
                )
                group["has_human_motion"] = torch.as_tensor(
                    np.asarray(raw["omg.condition.has_humanref"], dtype=np.bool_),
                    device=target_device,
                )
            yield group

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_info = self.samples[idx]
        episode, qpos_all, body_pos_all, body_quat_all = self._prepare_episode_kinematics(sample_info)
        total_len = int(qpos_all.shape[0])
        start, valid_len = self._get_window(sample_info, total_len)
        end = start + valid_len
        prev_indices = self._get_prev_indices(start, total_len)

        qpos_36 = qpos_all[start:end]
        body_pos_w = body_pos_all[start:end]
        body_quat_w = body_quat_all[start:end]
        prev_qpos_36 = qpos_all[prev_indices]
        prev_body_pos_w = body_pos_all[prev_indices]
        prev_body_quat_w = body_quat_all[prev_indices]
        valid_mask = get_valid_mask(valid_len, valid_len)
        fps_tensor = torch.tensor([self.default_fps], dtype=torch.float32)
        prev_state_features, canon_root_pos, canon_root_quat = self.codec.prev_state_features_from_history(
            prev_qpos_36.unsqueeze(0),
            prev_body_pos_w.unsqueeze(0),
            prev_body_quat_w.unsqueeze(0),
            fps=fps_tensor,
        )
        comps = self.codec.canonicalize(
            qpos_36.unsqueeze(0),
            body_pos_w.unsqueeze(0),
            body_quat_w.unsqueeze(0),
            anchor_root_pos=canon_root_pos,
            anchor_root_quat=canon_root_quat,
            fps=fps_tensor,
            valid_mask=valid_mask.unsqueeze(0),
        )
        motion_features = self.codec.assemble_features(comps)[0]
        qpos_36 = self._pad_or_trim(qpos_36)
        body_pos_w = self._pad_or_trim(body_pos_w)
        body_quat_w = self._pad_or_trim(body_quat_w)
        motion_features = self._pad_or_trim(motion_features)
        root_pos_local = self._pad_or_trim(comps.root_pos_local[0])
        root_rot_local_quat = self._pad_or_trim(comps.root_rot_local_quat[0])
        joint_dof = self._pad_or_trim(comps.joint_dof[0])
        body_link_pos_local = self._pad_or_trim(comps.body_link_pos_local[0])

        audio_features = torch.zeros(self.window_size, self.audio_dim, dtype=torch.float32)
        audio_mask = torch.zeros(self.window_size, dtype=torch.bool)
        if self.use_audio:
            audio = torch.as_tensor(episode["omg.audio.feature"][start:end], dtype=torch.float32)
            audio_features[:valid_len] = audio[:valid_len]
            audio_mask = self._condition_mask(
                episode["omg.condition.has_audio"],
                start=start,
                end=end,
                valid_len=valid_len,
                window_size=self.window_size,
            )
        human_motion = torch.zeros(self.window_size, self.human_motion_dim, dtype=torch.float32)
        humanref_mask = torch.zeros(self.window_size, dtype=torch.bool)
        if self.use_human_motion:
            humanref = torch.as_tensor(episode["omg.humanref.motion"][start:end], dtype=torch.float32)
            human_motion[:valid_len] = humanref[:valid_len]
            humanref_mask = self._condition_mask(
                episode["omg.condition.has_humanref"],
                start=start,
                end=end,
                valid_len=valid_len,
                window_size=self.window_size,
            )
        caption = str(sample_info.get("segment_caption", "")) if self.use_text else ""
        valid_window_mask = get_valid_mask(self.window_size, valid_len)
        return {
            "length": torch.tensor(valid_len, dtype=torch.long),
            "fps": torch.tensor(self.default_fps, dtype=torch.float32),
            "qpos_36": qpos_36,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "audio_features": audio_features,
            "human_motion": human_motion,
            "motion_features": motion_features,
            "root_pos_local": root_pos_local,
            "root_rot_local_quat": root_rot_local_quat,
            "joint_dof": joint_dof,
            "body_link_pos_local": body_link_pos_local,
            "prev_state_features": prev_state_features[0],
            "history_features": prev_state_features[0],
            "canonical_frame_idx": torch.tensor(self.codec.canonical_frame_idx, dtype=torch.long),
            "canon_root_pos": canon_root_pos[0],
            "canon_root_quat": canon_root_quat[0],
            "caption": caption,
            "has_text": torch.tensor(bool(caption.strip()), dtype=torch.bool),
            "mask": {
                "valid": valid_window_mask,
                "has_audio": audio_mask,
                "has_human_motion": humanref_mask,
            },
            "meta": {
                "source_file": f"{self.repo_id}#episode={sample_info['episode_index']}",
                "sequence_name": sample_info["sequence_name"],
                "source_dataset": sample_info["source_dataset"],
                "fps": self.default_fps,
                "split": self.split,
                "window_start": start,
                "window_end": end,
                "segment_index": sample_info.get("segment_index"),
                "segment_frame_start": sample_info.get("source_start_frame"),
                "segment_frame_end": sample_info.get("source_end_frame"),
                "segment_action": sample_info.get("segment_action", ""),
                "segment_style": sample_info.get("segment_style", ""),
                "video_summary": sample_info.get("video_summary", ""),
                "eval_window_index": sample_info.get("eval_window_index", 0),
                "eval_num_windows": sample_info.get("eval_num_windows", 1),
                "lerobot_episode_index": sample_info.get("episode_index"),
            },
        }
