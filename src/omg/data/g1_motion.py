from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from omg.core.tensor import get_valid_mask, repeat_to_max_len
from omg.data.unified import UnifiedG1MotionIndex
from omg.data.windowing import (
    ExhaustiveWindowSampleView,
    is_exhaustive_train_window_policy,
)
from omg.motion.feature_codec import G1MotionFeatureCodec
from omg.robots.g1.kinematics import G1Kinematics
from omg.utils.rotation_conversions import standardize_quaternion


class G1MotionDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        info_path: str | None = None,
        sequence_duration: float = 0.5,
        canonical_frame_idx: int | None = None,
        num_prev_states: int = 2,
        limit_size: int | None = None,
        fps: float = 30.0,
        kinematics_path: str = "assets/robots/g1/g1_kinematics.json",
        labels_root: str | None = None,
        text_root: str | None = None,
        sample_by_segment: bool = True,
        include_style_in_caption: bool = True,
        eval_window_policy: str = "uniform",
        eval_num_windows: int = 3,
        train_window_policy: str | None = None,
        train_window_stride: int | None = None,
        skip_missing_labels: bool = False,
        load_frame_conditions: bool = False,
        use_audio: bool = False,
        audio_dir: str | None = None,
        audio_dim: int = 35,
        use_text: bool = True,
        use_human_motion: bool = False,
        human_motion_dir: str | None = None,
        human_motion_dim: int = 66,
        rotation_representation: str = "quat",
        exhaustive_train_slicing: bool = True,
        train_slice_stride: int | None = None,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.sequence_duration = float(sequence_duration)
        self.num_prev_states = int(num_prev_states)
        self.canonical_frame_idx = canonical_frame_idx
        self.limit_size = limit_size
        self.default_fps = float(fps)
        self.training = split == "train"
        self.sample_by_segment = bool(sample_by_segment)
        self.exhaustive_train_slicing = bool(exhaustive_train_slicing)
        if train_window_policy is None:
            train_window_policy = "exhaustive" if self.exhaustive_train_slicing else "random"
        self.train_window_policy = str(train_window_policy)
        stride_value = train_window_stride if train_window_stride is not None else train_slice_stride
        self.train_window_stride = int(1 if stride_value is None else stride_value)
        if self.train_window_stride <= 0:
            raise ValueError(f"train_window_stride must be positive, got {self.train_window_stride}")
        self.train_slice_stride = self.train_window_stride
        self.load_frame_conditions = bool(load_frame_conditions)
        self.use_audio = bool(use_audio) or self.load_frame_conditions
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None
        self.audio_dim = int(audio_dim)
        self.use_text = bool(use_text)
        self.use_human_motion = bool(use_human_motion) or self.load_frame_conditions
        self.human_motion_dir = Path(human_motion_dir) if human_motion_dir is not None else None
        self.human_motion_dim = int(human_motion_dim)
        self.window_size = int(round(self.sequence_duration * self.default_fps))
        if self.window_size <= 0:
            raise ValueError(f"Invalid sequence_duration={sequence_duration} for fps={fps}")
        if info_path is None:
            info_path = str(self.dataset_root.parent / "info.yaml")

        self.kinematics = G1Kinematics(kinematics_path=kinematics_path)
        index_kwargs = {
            "dataset_root": self.dataset_root,
            "split": split,
            "info_path": info_path,
            "labels_root": labels_root,
            "text_root": text_root,
            "sample_by_segment": sample_by_segment,
            "include_style_in_caption": include_style_in_caption,
            "eval_window_policy": eval_window_policy,
            "eval_num_windows": eval_num_windows,
            "skip_missing_labels": skip_missing_labels,
            "window_size": self.window_size,
            "default_fps": self.default_fps,
            "training": self.training,
        }
        index = UnifiedG1MotionIndex(**index_kwargs)
        self.entries = index.entries
        self.uses_exhaustive_train_windows = (
            self.training and is_exhaustive_train_window_policy(self.train_window_policy)
        )
        self.samples = index.samples
        if self.uses_exhaustive_train_windows:
            if self.sample_by_segment:
                self.samples = self._build_sample_index(self.samples)
            else:
                self.entries = self._build_entry_index(self.entries)
        if self.use_audio and self.audio_dir is None and getattr(index, "audio_dir", None) is not None:
            self.audio_dir = Path(index.audio_dir)

        self.codec = G1MotionFeatureCodec(
            self.kinematics,
            num_prev_states=num_prev_states,
            canonical_frame_idx=canonical_frame_idx,
            rotation_representation=rotation_representation,
        )
        self._sequence_cache: dict[str, dict[str, np.ndarray | float]] = {}
        sample_count = len(self.samples) if self.sample_by_segment else len(self.entries)
        print(
            f"[INFO] G1MotionDataset root={self.dataset_root} split={self.split} "
            f"entries={len(self.entries)} samples={sample_count} "
            f"exhaustive_train_slicing={self.exhaustive_train_slicing} "
            f"train_slice_stride={self.train_slice_stride} "
            f"load_frame_conditions={self.load_frame_conditions} "
            f"train_window_policy={self.train_window_policy} "
            f"train_window_stride={self.train_window_stride} "
            f"use_audio={self.use_audio} audio_dim={self.audio_dim} audio_dir={self.audio_dir} "
            f"use_text={self.use_text} "
            f"use_human_motion={self.use_human_motion} "
            f"rotation_representation={self.codec.rotation_representation} "
            f"human_motion_dim={self.human_motion_dim} human_motion_dir={self.human_motion_dir}"
        )

    @staticmethod
    def _resolve_train_slice_stride(index_name: str, train_slice_stride: int | None) -> int:
        if train_slice_stride is None:
            return 1
        stride = int(train_slice_stride)
        if stride <= 0:
            raise ValueError(f"train_slice_stride must be positive, got {train_slice_stride}")
        return stride
    def _build_sample_index(self, samples):
        return ExhaustiveWindowSampleView(
            samples,
            window_size=self.window_size,
            stride=self.train_window_stride,
        )

    def _build_entry_index(self, entries: list[dict]):
        samples = []
        for entry in entries:
            with np.load(entry["path"], mmap_mode="r") as npz:
                qpos_key = "qpos" if "qpos" in npz else "robot_qpos"
                total_len = int(npz[qpos_key].shape[0])
            samples.append(
                {
                    **entry,
                    "segment_frame_start": 0,
                    "segment_frame_end": total_len,
                    "eval_window_index": 0,
                    "eval_num_windows": 1,
                    "fixed_window_start": None,
                }
            )
        return ExhaustiveWindowSampleView(
            samples,
            window_size=self.window_size,
            stride=self.train_window_stride,
        )

    def __len__(self) -> int:
        size = len(self.samples) if self.sample_by_segment else len(self.entries)
        return min(self.limit_size, size) if self.limit_size is not None else size

    def _pad_or_trim(self, x: torch.Tensor, valid_len: int) -> torch.Tensor:
        if x.shape[0] >= self.window_size:
            return x[: self.window_size]
        return repeat_to_max_len(x, self.window_size, dim=0)

    def _resolve_optional_npy(self, root: Path, sequence_name: str, modality: str) -> Path | None:
        stems = [sequence_name]
        if sequence_name.endswith("_retarget"):
            stems.append(sequence_name[: -len("_retarget")])
        candidates = [root / f"{stem}.npy" for stem in stems]
        existing = [path for path in candidates if path.exists()]
        if len(existing) > 1:
            raise ValueError(
                f"Ambiguous {modality} files for {sequence_name}: "
                + ", ".join(str(path) for path in existing)
            )
        return existing[0] if existing else None

    def _load_audio_features(self, sequence_name: str) -> np.ndarray | None:
        if not self.use_audio or self.audio_dir is None:
            return None
        audio_path = self._resolve_optional_npy(self.audio_dir, sequence_name, "audio")
        if audio_path is None:
            return None
        audio = np.asarray(np.load(audio_path), dtype=np.float32)
        if audio.ndim != 2 or audio.shape[-1] != self.audio_dim:
            raise ValueError(
                f"Expected audio features at {audio_path} to have shape (T, {self.audio_dim}), got {tuple(audio.shape)}"
            )
        return audio

    def _load_human_motion_features(self, sequence_name: str) -> np.ndarray | None:
        if not self.use_human_motion or self.human_motion_dir is None:
            return None
        human_motion_path = self._resolve_optional_npy(self.human_motion_dir, sequence_name, "human motion")
        if human_motion_path is None:
            return None
        return self._coerce_human_motion(np.load(human_motion_path), str(human_motion_path))

    @staticmethod
    def _has_valid_human_motion(human_motion: np.ndarray | torch.Tensor, valid_mask: np.ndarray | torch.Tensor | None = None) -> bool:
        if torch.is_tensor(human_motion):
            human = human_motion.detach()
            if valid_mask is not None:
                mask = valid_mask.detach().to(dtype=torch.bool) if torch.is_tensor(valid_mask) else torch.as_tensor(valid_mask, dtype=torch.bool)
                human = human[mask]
            return bool(human.numel() > 0 and torch.isfinite(human).all().item() and torch.count_nonzero(human).item() > 0)
        human = np.asarray(human_motion)
        if valid_mask is not None:
            human = human[np.asarray(valid_mask, dtype=np.bool_)]
        return bool(human.size > 0 and np.isfinite(human).all() and np.count_nonzero(human) > 0)

    def _coerce_human_motion(self, human_motion: np.ndarray, source: str) -> np.ndarray | None:
        human_motion = np.asarray(human_motion, dtype=np.float32)
        if human_motion.ndim == 3:
            if human_motion.shape[-1] != 3:
                raise ValueError(f"Expected human_joints from {source} to have shape (T, J, 3), got {tuple(human_motion.shape)}")
            human_motion = human_motion.reshape(human_motion.shape[0], -1)
        elif human_motion.ndim != 2:
            raise ValueError(f"Expected human reference condition from {source} to be 2D or 3D, got {tuple(human_motion.shape)}")
        if not self._has_valid_human_motion(human_motion):
            return None
        return human_motion

    def _load_human_motion(
        self,
        sequence_name: str,
        npz_human_joints: np.ndarray | None,
        npz_human_motion: np.ndarray | None,
    ) -> np.ndarray | None:
        if not self.use_human_motion:
            return None
        human_motion_from_dir = self._load_human_motion_features(sequence_name)
        if human_motion_from_dir is not None:
            return human_motion_from_dir
        if npz_human_joints is not None:
            return self._coerce_human_motion(npz_human_joints, f"{sequence_name}.npz:human_joints")
        if npz_human_motion is not None:
            return self._coerce_human_motion(npz_human_motion, f"{sequence_name}.npz:human_motion")
        return None

    def _load_sequence(self, path: Path) -> dict[str, np.ndarray | float]:
        cache_key = str(path)
        cached = self._sequence_cache.get(cache_key)
        if cached is not None:
            return cached
        with np.load(path) as npz:
            qpos_key = "qpos" if "qpos" in npz else "robot_qpos"
            body_pos_key = "body_pos_w" if "body_pos_w" in npz else "robot_body_pos_w"
            body_quat_key = "body_quat_w" if "body_quat_w" in npz else "robot_body_quat_wxyz_w"
            sequence = {
                "qpos": np.asarray(npz[qpos_key], dtype=np.float32),
                "body_pos_w": np.asarray(npz[body_pos_key], dtype=np.float32),
                "body_quat_w": np.asarray(npz[body_quat_key], dtype=np.float32),
                "fps": float(npz["fps"]) if "fps" in npz else self.default_fps,
            }
            npz_human_joints = np.asarray(npz["human_joints"], dtype=np.float32) if "human_joints" in npz else None
            npz_human_motion = np.asarray(npz["human_motion"], dtype=np.float32) if "human_motion" in npz else None
        audio = self._load_audio_features(path.stem)
        if audio is not None:
            if int(audio.shape[0]) != int(sequence["qpos"].shape[0]):
                min_len = min(int(audio.shape[0]), int(sequence["qpos"].shape[0]))
                audio = audio[:min_len]
                sequence["qpos"] = sequence["qpos"][:min_len]
                sequence["body_pos_w"] = sequence["body_pos_w"][:min_len]
                sequence["body_quat_w"] = sequence["body_quat_w"][:min_len]
            sequence["audio_features"] = audio
        human_motion = self._load_human_motion(path.stem, npz_human_joints, npz_human_motion)
        if human_motion is not None:
            if human_motion.shape[-1] != self.human_motion_dim:
                raise ValueError(
                    f"Expected human motion features for {path.stem} to have dim {self.human_motion_dim}, "
                    f"got {human_motion.shape[-1]}"
                )
            if int(human_motion.shape[0]) != int(sequence["qpos"].shape[0]):
                min_len = min(int(human_motion.shape[0]), int(sequence["qpos"].shape[0]))
                human_motion = human_motion[:min_len]
                sequence["qpos"] = sequence["qpos"][:min_len]
                sequence["body_pos_w"] = sequence["body_pos_w"][:min_len]
                sequence["body_quat_w"] = sequence["body_quat_w"][:min_len]
                if "audio_features" in sequence:
                    sequence["audio_features"] = sequence["audio_features"][:min_len]
            sequence["human_motion"] = human_motion
        self._sequence_cache[cache_key] = sequence
        return sequence

    def _get_window(self, sample_info: dict, total_len: int) -> tuple[int, int]:
        if self.sample_by_segment:
            segment_start = max(0, min(int(sample_info["segment_frame_start"]), total_len))
            segment_end = max(segment_start + 1, min(int(sample_info["segment_frame_end"]), total_len))
            segment_len = segment_end - segment_start
            if segment_len <= self.window_size:
                return segment_start, segment_len
            fixed_window_start = sample_info.get("fixed_window_start")
            if fixed_window_start is not None:
                start = max(segment_start, min(int(fixed_window_start), segment_end - self.window_size))
                return start, self.window_size
            if self.training:
                start = int(torch.randint(segment_start, segment_end - self.window_size + 1, (1,)).item())
                return start, self.window_size
            raise KeyError("fixed_window_start is required for evaluation samples")

        if total_len <= self.window_size:
            return 0, total_len
        fixed_window_start = sample_info.get("fixed_window_start")
        if fixed_window_start is not None:
            start = max(0, min(int(fixed_window_start), total_len - self.window_size))
            return start, self.window_size
        if self.training:
            start = int(torch.randint(0, total_len - self.window_size + 1, (1,)).item())
            return start, self.window_size
        return 0, self.window_size

    def _get_prev_indices(self, segment_start: int, start: int) -> torch.Tensor:
        return torch.tensor(
            [max(segment_start, start - self.num_prev_states + i) for i in range(self.num_prev_states)],
            dtype=torch.long,
        )

    def __getitem__(self, idx):
        sample_info = self.samples[idx] if self.sample_by_segment else self.entries[idx]
        path = Path(sample_info["path"])
        sequence = self._load_sequence(path)
        fps = float(sequence["fps"])
        qpos_all = sequence["qpos"]
        body_pos_all = sequence["body_pos_w"]
        body_quat_all = sequence["body_quat_w"]
        audio_all = sequence.get("audio_features")
        human_motion_all = sequence.get("human_motion")
        total_len = int(qpos_all.shape[0])
        start, valid_len = self._get_window(sample_info, total_len)
        end = start + valid_len
        segment_start = int(sample_info.get("segment_frame_start", 0))
        prev_indices = torch.clamp(self._get_prev_indices(segment_start, start), min=0, max=max(total_len - 1, 0))

        def seq_tensor(array, sl):
            return torch.from_numpy(np.array(array[sl], dtype=np.float32, copy=True))

        future_slice = slice(start, end)
        prev_np_indices = prev_indices.numpy()
        qpos_36 = seq_tensor(qpos_all, future_slice)
        body_pos_w = seq_tensor(body_pos_all, future_slice)
        body_quat_w = seq_tensor(body_quat_all, future_slice)
        audio_features = seq_tensor(audio_all, future_slice) if audio_all is not None else None
        human_motion = seq_tensor(human_motion_all, future_slice) if human_motion_all is not None else None
        prev_qpos_36 = seq_tensor(qpos_all, prev_np_indices)
        prev_body_pos_w = seq_tensor(body_pos_all, prev_np_indices)
        prev_body_quat_w = seq_tensor(body_quat_all, prev_np_indices)

        body_quat_w = standardize_quaternion(F.normalize(body_quat_w, dim=-1))
        prev_body_quat_w = standardize_quaternion(F.normalize(prev_body_quat_w, dim=-1))
        qpos_36[..., 3:7] = standardize_quaternion(F.normalize(qpos_36[..., 3:7], dim=-1))
        prev_qpos_36[..., 3:7] = standardize_quaternion(F.normalize(prev_qpos_36[..., 3:7], dim=-1))

        valid_mask = get_valid_mask(valid_len, valid_len)
        fps_tensor = torch.tensor([fps], dtype=torch.float32)
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
        qpos_36 = self._pad_or_trim(qpos_36, valid_len)
        body_pos_w = self._pad_or_trim(body_pos_w, valid_len)
        body_quat_w = self._pad_or_trim(body_quat_w, valid_len)
        has_audio = audio_features is not None
        has_human_motion = human_motion is not None and self._has_valid_human_motion(human_motion, valid_mask)
        if audio_features is not None:
            audio_features = self._pad_or_trim(audio_features, valid_len)
        else:
            audio_features = torch.zeros(self.window_size, self.audio_dim, dtype=torch.float32)
        if has_human_motion:
            human_motion = self._pad_or_trim(human_motion, valid_len)
        else:
            human_motion = torch.zeros(self.window_size, self.human_motion_dim, dtype=torch.float32)
        motion_features = self._pad_or_trim(motion_features, valid_len)
        root_pos_local = self._pad_or_trim(comps.root_pos_local[0], valid_len)
        root_rot_local_quat = self._pad_or_trim(comps.root_rot_local_quat[0], valid_len)
        joint_dof = self._pad_or_trim(comps.joint_dof[0], valid_len)
        body_link_pos_local = self._pad_or_trim(comps.body_link_pos_local[0], valid_len)

        caption = sample_info.get("segment_caption", "") if self.use_text else ""
        has_text = caption != ""
        valid_window_mask = get_valid_mask(self.window_size, valid_len)
        return {
            "length": torch.tensor(valid_len, dtype=torch.long),
            "fps": torch.tensor(fps, dtype=torch.float32),
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
            "has_text": torch.tensor(has_text, dtype=torch.bool),
            "mask": {
                "valid": valid_window_mask,
                "has_audio": valid_window_mask if has_audio else torch.zeros(self.window_size, dtype=torch.bool),
                "has_human_motion": valid_window_mask if has_human_motion else torch.zeros(self.window_size, dtype=torch.bool),
            },
            "meta": {
                "source_file": str(path),
                "sequence_name": sample_info["sequence_name"],
                "fps": fps,
                "split": self.split,
                "window_start": start,
                "window_end": end,
                "segment_index": sample_info.get("segment_index"),
                "segment_frame_start": sample_info.get("segment_frame_start"),
                "segment_frame_end": sample_info.get("segment_frame_end"),
                "segment_action": sample_info.get("segment_action", ""),
                "segment_style": sample_info.get("segment_style", ""),
                "video_summary": sample_info.get("video_summary", ""),
                "label_path": sample_info.get("label_path"),
                "text_path": sample_info.get("text_path"),
                "eval_window_index": sample_info.get("eval_window_index", 0),
                "eval_num_windows": sample_info.get("eval_num_windows", 1),
                **(
                    {"wav_path": str(Path(sample_info["music_wav_path"]).resolve())}
                    if sample_info.get("music_wav_path") is not None
                    else {}
                ),
            },
        }
