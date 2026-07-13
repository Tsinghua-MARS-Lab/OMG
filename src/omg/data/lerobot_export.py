from __future__ import annotations

import json
import math
import os
import shutil
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from omg.data.lerobot_schema import (
    LEROBOT_CHUNKS_SIZE,
    LEROBOT_DATA_PATH,
    LEROBOT_DATASET_VERSION,
    LEROBOT_EPISODES_PATH,
    LEROBOT_FPS,
    LEROBOT_TASKS_PATH,
    frame_features,
)
from omg.data.unified import UnifiedG1MotionIndex

np = None
pa = None
pd = None
pq = None
yaml = None
OmegaConf = None


def _ensure_runtime_imports() -> None:
    global np, pa, pd, pq, yaml, OmegaConf
    if np is not None:
        return
    import numpy as _np
    import pandas as _pd
    import pyarrow as _pa
    import pyarrow.parquet as _pq
    import yaml as _yaml
    from omegaconf import OmegaConf as _OmegaConf

    np = _np
    pa = _pa
    pd = _pd
    pq = _pq
    yaml = _yaml
    OmegaConf = _OmegaConf


def _load_yaml(path: Path) -> dict[str, Any]:
    _ensure_runtime_imports()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return data


def _atomic_write_parquet(table: Any, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incomplete_path = path.with_name(f".{path.name}.incomplete")
    try:
        pq.write_table(table, incomplete_path, **kwargs)
        with incomplete_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(incomplete_path, path)
    finally:
        incomplete_path.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incomplete_path = path.with_name(f".{path.name}.incomplete")
    try:
        with incomplete_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(incomplete_path, path)
    finally:
        incomplete_path.unlink(missing_ok=True)


def _resolve_config(raw: dict[str, Any], *, paths: dict[str, Any], representation: dict[str, Any]) -> dict[str, Any]:
    _ensure_runtime_imports()
    root = OmegaConf.create({"paths": paths, "representation": representation, "dataset": raw})
    OmegaConf.resolve(root)
    resolved = OmegaConf.to_container(root.dataset, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("Resolved dataset config must be a mapping")
    return resolved


def _resolve_paths(paths_config: dict[str, Any]) -> dict[str, Any]:
    _ensure_runtime_imports()
    root = OmegaConf.create({"paths": paths_config})
    OmegaConf.resolve(root)
    paths = OmegaConf.to_container(root.paths, resolve=True)
    if not isinstance(paths, dict):
        raise ValueError("Resolved paths config must be a mapping")
    return paths


def _fixed_float_array(values: Any, size: int):
    matrix = np.asarray(values, dtype=np.float32).reshape(-1, size)
    flat = pa.array(matrix.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, size)


def _frame_table(columns: dict[str, list[Any]], *, include_audio: bool, audio_dim: int, include_humanref: bool, humanref_dim: int):
    data: dict[str, Any] = {
        "observation.state": _fixed_float_array(np.concatenate(columns["observation.state"], axis=0), 36),
        "action": _fixed_float_array(np.concatenate(columns["action"], axis=0), 36),
        "omg.condition.has_text": pa.array(np.concatenate(columns["omg.condition.has_text"]), type=pa.bool_()),
        "timestamp": pa.array(np.concatenate(columns["timestamp"]), type=pa.float32()),
        "frame_index": pa.array(np.concatenate(columns["frame_index"]), type=pa.int64()),
        "episode_index": pa.array(np.concatenate(columns["episode_index"]), type=pa.int64()),
        "index": pa.array(np.concatenate(columns["index"]), type=pa.int64()),
        "task_index": pa.array(np.concatenate(columns["task_index"]), type=pa.int64()),
    }
    if include_audio:
        data["omg.audio.feature"] = _fixed_float_array(
            np.concatenate(columns["omg.audio.feature"], axis=0), audio_dim
        )
        data["omg.condition.has_audio"] = pa.array(
            np.concatenate(columns["omg.condition.has_audio"]), type=pa.bool_()
        )
    if include_humanref:
        data["omg.humanref.motion"] = _fixed_float_array(
            np.concatenate(columns["omg.humanref.motion"], axis=0), humanref_dim
        )
        data["omg.condition.has_humanref"] = pa.array(
            np.concatenate(columns["omg.condition.has_humanref"]), type=pa.bool_()
        )
    return pa.table(data)


def _episode_table(rows: list[dict[str, Any]]):
    return pa.table(
        {
            "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
            "tasks": pa.array([row["tasks"] for row in rows], type=pa.list_(pa.string())),
            "length": pa.array([row["length"] for row in rows], type=pa.int64()),
            "dataset_from_index": pa.array([row["dataset_from_index"] for row in rows], type=pa.int64()),
            "dataset_to_index": pa.array([row["dataset_to_index"] for row in rows], type=pa.int64()),
            "data/chunk_index": pa.array([row["data/chunk_index"] for row in rows], type=pa.int64()),
            "data/file_index": pa.array([row["data/file_index"] for row in rows], type=pa.int64()),
            "meta/episodes/chunk_index": pa.array(
                [row["meta/episodes/chunk_index"] for row in rows], type=pa.int64()
            ),
            "meta/episodes/file_index": pa.array(
                [row["meta/episodes/file_index"] for row in rows], type=pa.int64()
            ),
            "omg/dataset": pa.array([row["omg/dataset"] for row in rows], type=pa.string()),
            "omg/source_id": pa.array([row["omg/source_id"] for row in rows], type=pa.string()),
            "omg/split": pa.array([row["omg/split"] for row in rows], type=pa.string()),
            "omg/segment_index": pa.array([row["omg/segment_index"] for row in rows], type=pa.int64()),
            "omg/source_start_frame": pa.array(
                [row["omg/source_start_frame"] for row in rows], type=pa.int64()
            ),
            "omg/source_end_frame": pa.array([row["omg/source_end_frame"] for row in rows], type=pa.int64()),
            "omg/has_text": pa.array([row["omg/has_text"] for row in rows], type=pa.bool_()),
            "omg/has_audio": pa.array([row["omg/has_audio"] for row in rows], type=pa.bool_()),
            "omg/has_humanref": pa.array([row["omg/has_humanref"] for row in rows], type=pa.bool_()),
        }
    )


@dataclass
class _Sequence:
    qpos: Any
    fps: float
    human_motion: Any = None


def _load_sequence(path: Path, *, default_fps: float) -> _Sequence:
    with np.load(path, mmap_mode="r") as data:
        key = "qpos" if "qpos" in data else "robot_qpos"
        qpos = np.asarray(data[key], dtype=np.float32)
        fps = float(data["fps"]) if "fps" in data else float(default_fps)
        human = None
        if "human_joints" in data:
            human = np.asarray(data["human_joints"], dtype=np.float32)
        elif "human_motion" in data:
            human = np.asarray(data["human_motion"], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos_36 at {path}, got shape {tuple(qpos.shape)}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"Non-finite qpos values at {path}")
    if human is not None and human.ndim == 3:
        if human.shape[-1] != 3:
            raise ValueError(f"Expected human joints with shape (T, J, 3) at {path}, got {tuple(human.shape)}")
        human = human.reshape(human.shape[0], -1)
    return _Sequence(qpos=qpos, fps=fps, human_motion=human)


@dataclass
class _PreparedEntry:
    entry_index: int
    sequence_name: str
    sequence: _Sequence
    samples: list[dict[str, Any]]
    audio: Any
    humanref: Any


def _resolve_optional_npy(root: str | Path | None, sequence_name: str) -> Path | None:
    if root is None:
        return None
    root_path = Path(root)
    stems = [sequence_name]
    if sequence_name.endswith("_retarget"):
        stems.append(sequence_name[: -len("_retarget")])
    matches = [root_path / f"{stem}.npy" for stem in stems if (root_path / f"{stem}.npy").exists()]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous side-condition files for {sequence_name}: {matches}")
    return matches[0] if matches else None


def _load_optional_feature(path: Path | None, *, expected_dim: int, name: str) -> Any:
    if path is None:
        return None
    values = np.asarray(np.load(path), dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] == 3:
        values = values.reshape(values.shape[0], -1)
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(f"Expected {name} shape (T, {expected_dim}) at {path}, got {tuple(values.shape)}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite {name} values at {path}")
    return values


def _slice_condition(values: Any, *, start: int, end: int, expected_dim: int, name: str) -> Any:
    if values is None:
        return None
    if values.shape[0] < end:
        raise ValueError(f"{name} has {values.shape[0]} frames but motion slice requires frame {end}")
    sliced = np.asarray(values[start:end], dtype=np.float32)
    if sliced.shape != (end - start, expected_dim):
        raise ValueError(f"Expected sliced {name} shape {(end - start, expected_dim)}, got {tuple(sliced.shape)}")
    return sliced


@dataclass
class _Stats:
    count: int = 0
    total: Any = field(default=None)
    total_sq: Any = field(default=None)
    minimum: Any = field(default=None)
    maximum: Any = field(default=None)

    def add(self, values: Any) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        summed = values.sum(axis=0)
        sumsq = np.square(values).sum(axis=0)
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        self.total = summed if self.total is None else self.total + summed
        self.total_sq = sumsq if self.total_sq is None else self.total_sq + sumsq
        self.minimum = minimum if self.minimum is None else np.minimum(self.minimum, minimum)
        self.maximum = maximum if self.maximum is None else np.maximum(self.maximum, maximum)
        self.count += int(values.shape[0])

    def to_json(self) -> dict[str, Any]:
        if self.count == 0:
            return {}
        mean = self.total / float(self.count)
        variance = self.total_sq / float(self.count) - np.square(mean)
        return {
            "count": [self.count],
            "mean": mean.astype(float).tolist(),
            "std": np.sqrt(np.maximum(variance, 0.0)).astype(float).tolist(),
            "min": self.minimum.astype(float).tolist(),
            "max": self.maximum.astype(float).tolist(),
        }


class LeRobotV3ExportWriter:
    """Bulk writer for the official LeRobotDataset v3 storage contract."""

    def __init__(
        self,
        output_root: Path,
        *,
        repo_id: str,
        frames_per_file: int,
        episodes_per_file: int,
        fps: int,
        include_audio: bool,
        audio_dim: int,
        include_humanref: bool,
        humanref_dim: int,
        overwrite: bool,
    ) -> None:
        _ensure_runtime_imports()
        self.output_root = output_root
        self.repo_id = str(repo_id)
        self.frames_per_file = int(frames_per_file)
        self.episodes_per_file = int(episodes_per_file)
        self.fps = int(fps)
        self.include_audio = bool(include_audio)
        self.audio_dim = int(audio_dim)
        self.include_humanref = bool(include_humanref)
        self.humanref_dim = int(humanref_dim)
        if self.frames_per_file <= 0 or self.episodes_per_file <= 0:
            raise ValueError("frames_per_file and episodes_per_file must be positive")
        if output_root.exists():
            if not overwrite:
                raise FileExistsError(f"Output root already exists: {output_root}")
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True)
        self.columns: dict[str, list[Any]] = {
            key: []
            for key in frame_features(
                include_audio=self.include_audio,
                audio_dim=self.audio_dim,
                include_humanref=self.include_humanref,
                humanref_dim=self.humanref_dim,
            )
        }
        self.episodes: list[dict[str, Any]] = []
        self.tasks: dict[str, int] = {}
        self.split_ranges: dict[str, list[int]] = {}
        self.buffered_frames = 0
        self.data_file_count = 0
        self.global_frame_index = 0
        self.episode_index = 0
        self.observation_stats = _Stats()
        self.action_stats = _Stats()
        self.max_data_file_size = 0

    def _task_index(self, caption: str) -> int:
        task = caption.strip()
        if task not in self.tasks:
            self.tasks[task] = len(self.tasks)
        return self.tasks[task]

    def _current_data_location(self) -> tuple[int, int]:
        return divmod(self.data_file_count, LEROBOT_CHUNKS_SIZE)

    @property
    def visible_data_files(self) -> int:
        return self.data_file_count + int(self.buffered_frames > 0)

    def _flush_frames(self) -> None:
        if self.buffered_frames == 0:
            return
        chunk_index, file_index = self._current_data_location()
        path = self.output_root / LEROBOT_DATA_PATH.format(chunk_index=chunk_index, file_index=file_index)
        table = _frame_table(
            self.columns,
            include_audio=self.include_audio,
            audio_dim=self.audio_dim,
            include_humanref=self.include_humanref,
            humanref_dim=self.humanref_dim,
        )
        _atomic_write_parquet(table, path, compression="snappy", use_dictionary=True)
        self.max_data_file_size = max(self.max_data_file_size, path.stat().st_size)
        self.data_file_count += 1
        self.buffered_frames = 0
        for values in self.columns.values():
            values.clear()

    def add_episode(
        self,
        *,
        dataset_name: str,
        split: str,
        source_id: str,
        segment_index: int,
        source_start_frame: int,
        source_end_frame: int,
        fps: float,
        qpos: Any,
        caption: str,
        audio: Any = None,
        humanref: Any = None,
    ) -> None:
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != 36 or qpos.shape[0] == 0:
            raise ValueError(f"Expected non-empty episode qpos shape (T, 36), got {tuple(qpos.shape)}")
        if not math.isclose(float(fps), float(self.fps), rel_tol=0.0, abs_tol=1e-4):
            raise ValueError(f"LeRobotDataset requires one dataset FPS; expected {self.fps}, got {fps} for {source_id}")
        num_frames = int(qpos.shape[0])
        if self.buffered_frames and self.buffered_frames + num_frames > self.frames_per_file:
            self._flush_frames()
        chunk_index, file_index = self._current_data_location()
        task_index = self._task_index(caption)
        start_index = self.global_frame_index
        end_index = start_index + num_frames
        has_text = bool(caption.strip())
        has_audio = audio is not None
        has_humanref = humanref is not None
        action = np.concatenate((qpos[1:], qpos[-1:]), axis=0)

        self.columns["observation.state"].append(qpos)
        self.columns["action"].append(action)
        self.columns["omg.condition.has_text"].append(np.full(num_frames, has_text, dtype=np.bool_))
        self.columns["timestamp"].append(np.arange(num_frames, dtype=np.float32) / float(self.fps))
        self.columns["frame_index"].append(np.arange(num_frames, dtype=np.int64))
        self.columns["episode_index"].append(np.full(num_frames, self.episode_index, dtype=np.int64))
        self.columns["index"].append(np.arange(start_index, end_index, dtype=np.int64))
        self.columns["task_index"].append(np.full(num_frames, task_index, dtype=np.int64))
        if self.include_audio:
            audio_values = (
                np.asarray(audio, dtype=np.float32)
                if has_audio
                else np.zeros((num_frames, self.audio_dim), dtype=np.float32)
            )
            if audio_values.shape != (num_frames, self.audio_dim):
                raise ValueError(f"Expected audio shape {(num_frames, self.audio_dim)}, got {tuple(audio_values.shape)}")
            self.columns["omg.audio.feature"].append(audio_values)
            self.columns["omg.condition.has_audio"].append(np.full(num_frames, has_audio, dtype=np.bool_))
        if self.include_humanref:
            humanref_values = (
                np.asarray(humanref, dtype=np.float32)
                if has_humanref
                else np.zeros((num_frames, self.humanref_dim), dtype=np.float32)
            )
            if humanref_values.shape != (num_frames, self.humanref_dim):
                raise ValueError(
                    f"Expected humanref shape {(num_frames, self.humanref_dim)}, got {tuple(humanref_values.shape)}"
                )
            self.columns["omg.humanref.motion"].append(humanref_values)
            self.columns["omg.condition.has_humanref"].append(
                np.full(num_frames, has_humanref, dtype=np.bool_)
            )

        split_range = self.split_ranges.setdefault(split, [self.episode_index, self.episode_index])
        if split_range[1] != self.episode_index:
            raise ValueError(f"Episodes for split {split!r} must be contiguous")
        split_range[1] = self.episode_index + 1
        self.episodes.append(
            {
                "episode_index": self.episode_index,
                "tasks": [caption.strip()],
                "length": num_frames,
                "dataset_from_index": start_index,
                "dataset_to_index": end_index,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "omg/dataset": dataset_name,
                "omg/source_id": source_id,
                "omg/split": split,
                "omg/segment_index": int(segment_index),
                "omg/source_start_frame": int(source_start_frame),
                "omg/source_end_frame": int(source_end_frame),
                "omg/has_text": has_text,
                "omg/has_audio": has_audio,
                "omg/has_humanref": has_humanref,
            }
        )
        self.observation_stats.add(qpos)
        self.action_stats.add(action)
        self.buffered_frames += num_frames
        self.global_frame_index = end_index
        self.episode_index += 1

    def _write_episodes(self) -> int:
        files = 0
        for start in range(0, len(self.episodes), self.episodes_per_file):
            rows = self.episodes[start : start + self.episodes_per_file]
            chunk_index, file_index = divmod(files, LEROBOT_CHUNKS_SIZE)
            for row in rows:
                row["meta/episodes/chunk_index"] = chunk_index
                row["meta/episodes/file_index"] = file_index
            path = self.output_root / LEROBOT_EPISODES_PATH.format(
                chunk_index=chunk_index, file_index=file_index
            )
            _atomic_write_parquet(
                _episode_table(rows), path, compression="snappy", use_dictionary=True
            )
            files += 1
        return files

    def close(self) -> dict[str, Any]:
        self._flush_frames()
        episode_files = self._write_episodes()
        task_items = sorted((index, task) for task, index in self.tasks.items())
        task_frame = pd.DataFrame(
            {"task_index": [index for index, _ in task_items]},
            index=pd.Index([task for _, task in task_items], name="task"),
        )
        tasks_path = self.output_root / LEROBOT_TASKS_PATH
        _atomic_write_parquet(pa.Table.from_pandas(task_frame, preserve_index=True), tasks_path)
        features = frame_features(
            include_audio=self.include_audio,
            audio_dim=self.audio_dim,
            include_humanref=self.include_humanref,
            humanref_dim=self.humanref_dim,
        )
        info = {
            "codebase_version": LEROBOT_DATASET_VERSION,
            "robot_type": "unitree_g1",
            "total_episodes": self.episode_index,
            "total_frames": self.global_frame_index,
            "total_tasks": len(self.tasks),
            "chunks_size": LEROBOT_CHUNKS_SIZE,
            "data_files_size_in_mb": max(1, int(math.ceil(self.max_data_file_size / (1024**2)))),
            "video_files_size_in_mb": 200,
            "fps": self.fps,
            "splits": {name: f"{bounds[0]}:{bounds[1]}" for name, bounds in self.split_ranges.items()},
            "data_path": LEROBOT_DATA_PATH,
            "video_path": None,
            "features": features,
        }
        stats = {
            "observation.state": self.observation_stats.to_json(),
            "action": self.action_stats.to_json(),
        }
        _atomic_write_text(
            self.output_root / "meta" / "info.json",
            json.dumps(info, indent=2, sort_keys=True) + "\n",
        )
        _atomic_write_text(
            self.output_root / "meta" / "stats.json",
            json.dumps(stats, indent=2, sort_keys=True) + "\n",
        )
        manifest = {
            "format": "LeRobotDataset-v3.0",
            "repo_id": self.repo_id,
            "episodes": self.episode_index,
            "frames": self.global_frame_index,
            "tasks": len(self.tasks),
            "data_files": self.data_file_count,
            "episode_files": episode_files,
            "splits": info["splits"],
        }
        modalities = ["text"] if any(task for task in self.tasks) else []
        if self.include_audio:
            modalities.append("audio")
        if self.include_humanref:
            modalities.append("human reference")
        split_lines = "\n".join(f"- `{name}`: episodes {bounds}" for name, bounds in info["splits"].items())
        modality_text = ", ".join(modalities) if modalities else "unconditioned"
        card = f"""---
pretty_name: OMG-Data
task_categories:
- robotics
tags:
- LeRobot
- humanoid
- motion-generation
---

# OMG-Data

Official LeRobotDataset v3.0 release for [OMG](https://github.com/Tsinghua-MARS-Lab/OMG).
It contains Unitree G1 reference motion with {modality_text} conditioning.

## Dataset

- Episodes: {self.episode_index}
- Frames: {self.global_frame_index}
- FPS: {self.fps}
- Robot: Unitree G1, 29 actuated joints

{split_lines}

`observation.state` is the 36D G1 `qpos` at frame `t`. `action` is the target
36D G1 `qpos` at frame `t+1`; it is a reference-motion target, not a low-level
motor command. Text conditions use the standard LeRobot task table.

## Loading

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("{self.repo_id}")
sample = dataset[0]
```

OMG training converts these source episodes into fixed windows and the 125D
motion representation. See the repository data documentation for direct and
materialized training commands.

## Terms

OMG-Data aggregates motions derived from multiple research datasets. Original
source licenses and usage terms continue to apply to their corresponding
motions; users must review those terms before redistribution or commercial use.
"""
        _atomic_write_text(self.output_root / "README.md", card)
        # The manifest is the completion marker consumed by release tooling.
        _atomic_write_text(
            self.output_root / "meta" / "omg_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        return manifest


def _iter_dataset_cfgs(
    data_cfg: dict[str, Any],
    *,
    paths: dict[str, Any],
    representation: dict[str, Any],
    split: str,
    requested_datasets: set[str],
) -> Iterable[tuple[str, dict[str, Any]]]:
    dataset_opts = data_cfg.get("dataset_opts")
    if not isinstance(dataset_opts, dict):
        raise ValueError("data config must contain dataset_opts mapping")
    split_cfgs = dataset_opts.get(split)
    if not isinstance(split_cfgs, dict):
        raise KeyError(f"Split {split!r} not found in data config")
    for name, raw in split_cfgs.items():
        if requested_datasets and name not in requested_datasets:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"Dataset config for {split}:{name} must be a mapping")
        cfg = _resolve_config(raw, paths=paths, representation=representation)
        if cfg.get("_target_") != "omg.data.g1_motion.G1MotionDataset":
            raise ValueError(f"LeRobot export requires G1MotionDataset source configs, got {cfg.get('_target_')}")
        cfg["split"] = split
        yield name, cfg


def _samples_for_entry(
    index: UnifiedG1MotionIndex,
    entry_info: dict[str, Any],
    *,
    total_len: int,
    fps: float,
    use_text: bool,
) -> list[dict[str, Any]]:
    if not use_text:
        return index._make_sample(entry_info, fps, total_len, 0, total_len, "", label_path=None)
    try:
        label_path = index._resolve_label_or_text_path(entry_info)
    except FileNotFoundError:
        if index.labels_root is None and index.text_root is None:
            label_path = None
        elif index.skip_missing_labels:
            return []
        else:
            raise
    if label_path is None:
        return index._make_sample(entry_info, fps, total_len, 0, total_len, "", label_path=None)
    return index._samples_from_label(entry_info, label_path, total_len, fps)


def _modality_dimensions(jobs: list[tuple[str, str, dict[str, Any]]], key: str, dim_key: str, default: int) -> tuple[bool, int]:
    dimensions = {int(cfg.get(dim_key, default)) for _, _, cfg in jobs if bool(cfg.get(key, False))}
    if len(dimensions) > 1:
        raise ValueError(f"All exported {key} datasets must use one feature dimension, got {sorted(dimensions)}")
    return bool(dimensions), next(iter(dimensions), default)


def _prepare_entry(
    *,
    entry_index: int,
    entry_info: dict[str, Any],
    index: UnifiedG1MotionIndex,
    cfg: dict[str, Any],
    audio_dim: int,
    humanref_dim: int,
) -> _PreparedEntry:
    sequence = _load_sequence(Path(entry_info["path"]), default_fps=float(cfg.get("fps", LEROBOT_FPS)))
    samples = _samples_for_entry(
        index,
        entry_info,
        total_len=int(sequence.qpos.shape[0]),
        fps=sequence.fps,
        use_text=bool(cfg.get("use_text", True)),
    )
    sequence_name = str(entry_info.get("sequence_name", Path(entry_info["path"]).stem))
    audio = None
    if bool(cfg.get("use_audio", False)):
        audio_path = _resolve_optional_npy(cfg.get("audio_dir"), sequence_name)
        audio = _load_optional_feature(audio_path, expected_dim=audio_dim, name="audio")
    humanref = None
    if bool(cfg.get("use_human_motion", False)):
        humanref_path = _resolve_optional_npy(cfg.get("human_motion_dir"), sequence_name)
        humanref = _load_optional_feature(humanref_path, expected_dim=humanref_dim, name="humanref")
        if humanref is None and sequence.human_motion is not None:
            humanref = _load_optional_feature_array(
                sequence.human_motion,
                expected_dim=humanref_dim,
                name=f"humanref in {entry_info['path']}",
            )
    return _PreparedEntry(
        entry_index=entry_index,
        sequence_name=sequence_name,
        sequence=sequence,
        samples=samples,
        audio=audio,
        humanref=humanref,
    )


def _iter_prepared_entries(
    *,
    entries: list[dict[str, Any]],
    index: UnifiedG1MotionIndex,
    cfg: dict[str, Any],
    audio_dim: int,
    humanref_dim: int,
    read_workers: int,
) -> Iterable[_PreparedEntry]:
    if read_workers <= 1:
        for entry_index, entry_info in enumerate(entries, start=1):
            yield _prepare_entry(
                entry_index=entry_index,
                entry_info=entry_info,
                index=index,
                cfg=cfg,
                audio_dim=audio_dim,
                humanref_dim=humanref_dim,
            )
        return
    max_pending = read_workers * 2
    source = iter(enumerate(entries, start=1))
    with ThreadPoolExecutor(max_workers=read_workers, thread_name_prefix="omg-data-read") as executor:
        pending: deque[Future[_PreparedEntry]] = deque()

        def submit_next() -> bool:
            try:
                entry_index, entry_info = next(source)
            except StopIteration:
                return False
            pending.append(
                executor.submit(
                    _prepare_entry,
                    entry_index=entry_index,
                    entry_info=entry_info,
                    index=index,
                    cfg=cfg,
                    audio_dim=audio_dim,
                    humanref_dim=humanref_dim,
                )
            )
            return True

        while len(pending) < max_pending and submit_next():
            pass
        while pending:
            yield pending.popleft().result()
            submit_next()


def export_lerobot(args: Any) -> dict[str, Any]:
    _ensure_runtime_imports()
    data_cfg = _load_yaml(Path(args.data_config))
    representation = _load_yaml(Path(args.representation_config))
    paths = _resolve_paths(_load_yaml(Path(args.paths_config)))
    requested_datasets = set(args.datasets or [])
    jobs = [
        (split, name, cfg)
        for split in args.splits
        for name, cfg in _iter_dataset_cfgs(
            data_cfg,
            paths=paths,
            representation=representation,
            split=split,
            requested_datasets=requested_datasets,
        )
    ]
    include_audio, audio_dim = _modality_dimensions(jobs, "use_audio", "audio_dim", 35)
    include_humanref, humanref_dim = _modality_dimensions(
        jobs, "use_human_motion", "human_motion_dim", 66
    )
    writer = LeRobotV3ExportWriter(
        Path(args.output_root),
        repo_id=str(args.repo_id),
        frames_per_file=int(args.frames_per_file),
        episodes_per_file=int(args.episodes_per_file),
        fps=int(args.fps),
        include_audio=include_audio,
        audio_dim=audio_dim,
        include_humanref=include_humanref,
        humanref_dim=humanref_dim,
        overwrite=bool(args.overwrite),
    )
    progress_every = int(getattr(args, "progress_every", 1000) or 0)
    read_workers = int(getattr(args, "read_workers", 1))
    if read_workers <= 0:
        raise ValueError(f"read_workers must be positive, got {read_workers}")
    for split, dataset_name, cfg in jobs:
        index = UnifiedG1MotionIndex(
            dataset_root=cfg["dataset_root"],
            split=split,
            info_path=cfg.get("info_path"),
            labels_root=cfg.get("labels_root"),
            text_root=cfg.get("text_root"),
            sample_by_segment=False,
            include_style_in_caption=cfg.get("include_style_in_caption", True),
            skip_missing_labels=cfg.get("skip_missing_labels", False),
            window_size=int(round(float(cfg.get("sequence_duration", 2.0)) * float(cfg.get("fps", 30.0)))),
            default_fps=float(cfg.get("fps", 30.0)),
            training=True,
            max_entries=args.max_entries_per_dataset,
        )
        if bool(cfg.get("use_text", True)):
            index._get_label_relative_files()
            index._get_text_relative_files()
        print(f"[export_lerobot] split={split} dataset={dataset_name} entries={len(index.entries)}", flush=True)
        exported = 0
        prepared_entries = _iter_prepared_entries(
            entries=index.entries,
            index=index,
            cfg=cfg,
            audio_dim=audio_dim,
            humanref_dim=humanref_dim,
            read_workers=read_workers,
        )
        for prepared in prepared_entries:
            for sample in prepared.samples:
                if args.max_episodes is not None and writer.episode_index >= int(args.max_episodes):
                    return writer.close()
                if args.max_episodes_per_dataset is not None and exported >= int(args.max_episodes_per_dataset):
                    break
                start = int(sample["segment_frame_start"])
                end = int(sample["segment_frame_end"])
                caption = str(sample.get("segment_caption", "")) if bool(cfg.get("use_text", True)) else ""
                audio = _slice_condition(
                    prepared.audio,
                    start=start,
                    end=end,
                    expected_dim=audio_dim,
                    name=f"audio for {prepared.sequence_name}",
                )
                humanref = _slice_condition(
                    prepared.humanref,
                    start=start,
                    end=end,
                    expected_dim=humanref_dim,
                    name=f"humanref for {prepared.sequence_name}",
                )
                writer.add_episode(
                    dataset_name=dataset_name,
                    split=split,
                    source_id=prepared.sequence_name,
                    segment_index=int(sample.get("segment_index", 0)),
                    source_start_frame=start,
                    source_end_frame=end,
                    fps=float(sample.get("fps", prepared.sequence.fps)),
                    qpos=prepared.sequence.qpos[start:end],
                    caption=caption,
                    audio=audio,
                    humanref=humanref,
                )
                exported += 1
                if progress_every > 0 and exported % progress_every == 0:
                    print(
                        "[export_lerobot] "
                        f"split={split} dataset={dataset_name} exported={exported} "
                        f"entries={prepared.entry_index}/{len(index.entries)} total_episodes={writer.episode_index} "
                        f"total_frames={writer.global_frame_index} data_files={writer.visible_data_files}",
                        flush=True,
                    )
            if args.max_episodes_per_dataset is not None and exported >= int(args.max_episodes_per_dataset):
                break
        print(
            "[export_lerobot] "
            f"split={split} dataset={dataset_name} done exported={exported} "
            f"total_episodes={writer.episode_index} total_frames={writer.global_frame_index} "
            f"data_files={writer.visible_data_files}",
            flush=True,
        )
    return writer.close()


def _load_optional_feature_array(values: Any, *, expected_dim: int, name: str) -> Any:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] == 3:
        values = values.reshape(values.shape[0], -1)
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(f"Expected {name} shape (T, {expected_dim}), got {tuple(values.shape)}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite {name} values")
    if np.count_nonzero(values) == 0:
        return None
    return values
