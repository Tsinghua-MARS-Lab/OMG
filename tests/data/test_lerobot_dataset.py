from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from omg.cli.data.materialize_episode_cache import write_episode_cache
from omg.benchmarks.lerobot import BENCHMARK_SAMPLE_SCHEMA, build_lerobot_benchmark_views
from omg.data.episode_cache import EpisodeCachedG1MotionDataset
from omg.data.lerobot_dataset import LeRobotG1MotionDataset


def _write_lerobot_fixture(root: Path, *, split: str = "train") -> None:
    frame_root = root / "data" / "chunk-000"
    episode_root = root / "meta" / "episodes" / "chunk-000"
    frame_root.mkdir(parents=True)
    episode_root.mkdir(parents=True)

    frames = 5
    qpos = np.zeros((frames, 36), dtype=np.float32)
    qpos[:, 0] = np.arange(2, 2 + frames, dtype=np.float32)
    qpos[:, 3] = 1.0
    action = np.concatenate((qpos[1:], qpos[-1:]), axis=0)
    audio = np.ones((frames, 35), dtype=np.float32)
    human = np.ones((frames, 66), dtype=np.float32)
    pq.write_table(
        pa.table(
            {
                "observation.state": qpos.tolist(),
                "action": action.tolist(),
                "omg.audio.feature": audio.tolist(),
                "omg.humanref.motion": human.tolist(),
                "omg.condition.has_audio": [True] * frames,
                "omg.condition.has_humanref": [True] * frames,
            }
        ),
        frame_root / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "length": [frames],
                "dataset_from_index": [0],
                "dataset_to_index": [frames],
                "tasks": [["walk forward; style: steady"]],
                "omg/source_id": ["toy"],
                "omg/dataset": [f"toy_{split}"],
                "omg/segment_index": [0],
                "omg/source_start_frame": [2],
                "omg/source_end_frame": [7],
                "omg/has_text": [True],
                "omg/has_audio": [True],
                "omg/has_humanref": [True],
                "omg/split": [split],
            }
        ),
        episode_root / "file-000.parquet",
    )
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "splits": {split: "0:1"},
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [36]},
                    "action": {"dtype": "float32", "shape": [36]},
                    "omg.audio.feature": {"dtype": "float32", "shape": [35]},
                    "omg.humanref.motion": {"dtype": "float32", "shape": [66]},
                },
            }
        ),
        encoding="utf-8",
    )


def test_lerobot_reader_and_episode_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_root = tmp_path / "lerobot"
    _write_lerobot_fixture(dataset_root)
    dataset = LeRobotG1MotionDataset(
        dataset_root=dataset_root,
        repo_id="THU-MARS/OMG-Data",
        revision="test-revision",
        split="train",
        sequence_duration=2.0,
        fps=30.0,
        num_prev_states=2,
        train_window_policy="exhaustive",
        rotation_representation="rot6d",
        use_audio=True,
        use_human_motion=True,
    )
    fk_calls = 0
    original_forward_kinematics = dataset.kinematics.forward_kinematics

    def counted_forward_kinematics(*args, **kwargs):
        nonlocal fk_calls
        fk_calls += 1
        return original_forward_kinematics(*args, **kwargs)

    monkeypatch.setattr(dataset.kinematics, "forward_kinematics", counted_forward_kinematics)
    sample = dataset[0]
    dataset[0]
    assert fk_calls == 1
    sample_fk_calls = fk_calls
    stats_batch = next(dataset.iter_stats_batches(batch_size=4))
    valid = stats_batch["valid_mask"][0]
    torch.testing.assert_close(stats_batch["motion_features"][0, valid], sample["motion_features"][sample["mask"]["valid"]])
    torch.testing.assert_close(stats_batch["qpos_36"][0, valid], sample["qpos_36"][sample["mask"]["valid"]])
    assert fk_calls == sample_fk_calls + 1
    assert sample["caption"] == "walk forward; style: steady"
    assert sample["qpos_36"].shape == (60, 36)
    assert sample["motion_features"].shape[-1] == 125
    assert sample["mask"]["valid"].sum().item() == 5
    assert sample["qpos_36"][0, 0].item() == 2.0
    assert sample["mask"]["has_audio"].sum().item() == 5
    assert sample["mask"]["has_human_motion"].sum().item() == 5

    cache_root = tmp_path / "episode-cache"
    cache_summary = write_episode_cache(
        dataset,
        output_root=cache_root,
        split="train",
        max_frames_per_shard=100,
        device="cpu",
        overwrite=False,
    )
    assert cache_summary["windows"] == 1
    cached_sample = EpisodeCachedG1MotionDataset(
        root=cache_root,
        split="train",
        source_repo_id="THU-MARS/OMG-Data",
        source_revision="test-revision",
    )[0]
    for key in (
        "qpos_36",
        "body_pos_w",
        "body_quat_w",
        "audio_features",
        "human_motion",
        "motion_features",
        "history_features",
    ):
        torch.testing.assert_close(cached_sample[key], sample[key])
    assert cached_sample["mask"]["valid"].equal(sample["mask"]["valid"])
    assert cached_sample["mask"]["has_audio"].equal(sample["mask"]["has_audio"])
    assert cached_sample["mask"]["has_human_motion"].equal(sample["mask"]["has_human_motion"])
    assert cached_sample["caption"] == sample["caption"]


def test_lerobot_benchmark_view_resolves_complete_identity(tmp_path: Path) -> None:
    dataset_root = tmp_path / "lerobot"
    _write_lerobot_fixture(dataset_root, split="test")
    dataset = LeRobotG1MotionDataset(
        dataset_root=dataset_root,
        repo_id="THU-MARS/OMG-Data",
        revision="test-revision",
        split="test",
        sequence_duration=0.1,
        fps=30.0,
        num_prev_states=2,
        rotation_representation="rot6d",
        use_audio=True,
        use_human_motion=True,
        eval_num_windows=1,
    )
    views = build_lerobot_benchmark_views(dataset)
    view = views["toy_test"]
    identity = view.sample_identity(0)
    assert identity == {
        "schema": BENCHMARK_SAMPLE_SCHEMA,
        "repo_id": "THU-MARS/OMG-Data",
        "revision": "test-revision",
        "split": "test",
        "episode_index": 0,
        "window_start": 0,
        "num_frames": 3,
        "source_dataset": "toy_test",
        "source_id": "toy",
        "segment_index": 0,
        "source_start_frame": 2,
        "source_end_frame": 7,
    }
    assert view.resolve_identity(identity) == 0
    assert view.sample_has_condition(0, "text", num_frames=3)
    assert view.sample_has_condition(0, "audio", num_frames=3)
    assert view.sample_has_condition(0, "humanref", num_frames=3)
    with pytest.raises(ValueError, match="revision"):
        view.resolve_identity({**identity, "revision": "other"})
    with pytest.raises(KeyError, match="Unknown LeRobot source datasets"):
        build_lerobot_benchmark_views(dataset, include=["toy"])
