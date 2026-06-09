from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_human_motion_array(path: str | Path, expected_dim: int) -> np.ndarray:
    ref_path = Path(path).expanduser()
    if not ref_path.exists():
        raise FileNotFoundError(f"Human reference motion file does not exist: {ref_path}")
    if ref_path.suffix == ".npy":
        human = np.load(ref_path)
    elif ref_path.suffix == ".npz":
        with np.load(ref_path) as npz:
            for key in ("human_motion", "human_joints", "joints", "poses"):
                if key in npz:
                    human = np.asarray(npz[key])
                    break
            else:
                raise KeyError(f"No human_motion/human_joints/joints/poses key found in {ref_path}")
    else:
        raise ValueError(f"Unsupported human reference motion extension: {ref_path.suffix}")
    human = np.asarray(human, dtype=np.float32)
    if human.ndim == 3:
        if human.shape[-1] != 3:
            raise ValueError(f"Expected human joints shape (T,J,3), got {human.shape}")
        human = human.reshape(human.shape[0], -1)
    if human.ndim != 2 or human.shape[-1] != int(expected_dim):
        raise ValueError(f"Expected human reference motion shape (T,{int(expected_dim)}), got {human.shape}")
    if not np.isfinite(human).all():
        raise ValueError("Human reference motion contains non-finite values")
    return human.astype(np.float32, copy=False)


def _pad_or_trim_features(features: np.ndarray, num_frames: int) -> tuple[np.ndarray, np.ndarray, int | None]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected human reference motion with shape (T,D), got {features.shape}")
    frames = int(num_frames)
    if frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if features.shape[0] >= frames:
        return features[:frames].astype(np.float32, copy=False), np.ones(frames, dtype=bool), None
    if features.shape[0] <= 0:
        raise ValueError("Human reference motion must contain at least one frame")
    valid_frames = int(features.shape[0])
    pad = np.zeros((frames - valid_frames, features.shape[1]), dtype=np.float32)
    valid = np.zeros(frames, dtype=bool)
    valid[:valid_frames] = True
    return np.concatenate([features, pad], axis=0).astype(np.float32, copy=False), valid, valid_frames


@dataclass(frozen=True)
class PipelineHumanMotion:
    features: np.ndarray
    mask: np.ndarray
    source_path: str
    fps: float
    padded_from_frames: int | None = None

    def features_for_plan(
        self,
        plan_index: int,
        *,
        num_frames: int,
        sequence_length: int,
        allow_multi_chunk: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(self.features, dtype=np.float32)
        mask = np.asarray(self.mask, dtype=bool)
        if allow_multi_chunk:
            return features[: int(num_frames)], mask[: int(num_frames)]
        index = int(plan_index)
        start = index * int(sequence_length)
        end = start + int(num_frames)
        if end > features.shape[0]:
            raise ValueError(
                f"Human reference condition has {features.shape[0]} frames, but plan {index} requests [{start}, {end})"
            )
        return features[start:end], mask[start:end]

    def features_for_frame(self, start_frame: int, *, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(self.features, dtype=np.float32)
        mask = np.asarray(self.mask, dtype=bool)
        start = int(start_frame)
        frames = int(num_frames)
        if start < 0:
            raise ValueError(f"start_frame must be non-negative, got {start_frame}")
        if frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        end = start + frames
        if end > features.shape[0]:
            raise ValueError(
                f"Human reference condition has {features.shape[0]} frames, but requests [{start}, {end})"
            )
        return features[start:end], mask[start:end]

    def describe(self) -> dict[str, Any]:
        return {
            "path": self.source_path,
            "fps": float(self.fps),
            "shape": list(np.asarray(self.features).shape),
            "valid_frames": int(np.asarray(self.mask, dtype=bool).sum()),
            "padded_from_frames": self.padded_from_frames,
        }


def load_pipeline_human_motion(
    path: str | Path,
    *,
    fps: float,
    human_motion_dim: int,
    num_frames: int,
) -> PipelineHumanMotion:
    human_path = Path(path).expanduser()
    features = _load_human_motion_array(human_path, int(human_motion_dim))
    features, mask, padded_from_frames = _pad_or_trim_features(features, int(num_frames))
    return PipelineHumanMotion(
        features=features,
        mask=mask,
        source_path=str(human_path),
        fps=float(fps),
        padded_from_frames=padded_from_frames,
    )


def human_motion_for_plan(
    human_motion: PipelineHumanMotion | None,
    plan_index: int,
    *,
    num_frames: int,
    sequence_length: int,
    allow_multi_chunk: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    if human_motion is None:
        return None
    return human_motion.features_for_plan(
        int(plan_index),
        num_frames=int(num_frames),
        sequence_length=int(sequence_length),
        allow_multi_chunk=bool(allow_multi_chunk),
    )


def human_motion_for_timeline_frame(
    human_motion: PipelineHumanMotion | None,
    start_frame: int,
    *,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if human_motion is None:
        return None
    return human_motion.features_for_frame(int(start_frame), num_frames=int(num_frames))


def describe_pipeline_human_motion(human_motion: PipelineHumanMotion | None) -> dict[str, Any] | None:
    if human_motion is None:
        return None
    return human_motion.describe()
