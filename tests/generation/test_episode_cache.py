from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omg.data.episode_cache import EpisodeCachedG1MotionDataset
from omg.data.episode_cache_inspect import inspect_episode_cache
from omg.cli.data.materialize_episode_cache import write_episode_cache

SOURCE_REPO_ID = "THU-MARS/OMG-Data"
SOURCE_REVISION = "test-revision"


def _open_cache(root, split="train"):
    return EpisodeCachedG1MotionDataset(
        root=root,
        split=split,
        source_repo_id=SOURCE_REPO_ID,
        source_revision=SOURCE_REVISION,
    )


def test_episode_cache_maps_exact_windows_and_spans(tmp_path) -> None:
    split_root = tmp_path / "cache" / "train"
    shard_root = split_root / "shards" / "shard_00000"
    shard_root.mkdir(parents=True)
    qpos = np.zeros((8, 36), dtype=np.float32)
    qpos[:, 0] = np.arange(8, dtype=np.float32)
    qpos[:, 3] = 1.0
    body_pos = np.zeros((8, 30, 3), dtype=np.float32)
    body_pos[:, 0, 0] = qpos[:, 0]
    body_quat = np.zeros((8, 30, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.save(shard_root / "qpos_36.npy", qpos)
    np.save(shard_root / "body_pos_w.npy", body_pos)
    np.save(shard_root / "body_quat_w.npy", body_quat)
    np.savez(
        split_root / "episodes.npz",
        shard=np.asarray([0, 0], dtype=np.int32),
        frame_offset=np.asarray([0, 5], dtype=np.int64),
        length=np.asarray([5, 3], dtype=np.int32),
        window_offset=np.asarray([0, 3, 4], dtype=np.int64),
    )
    (split_root / "captions.json").write_text(json.dumps(["walk", "stand"]), encoding="utf-8")
    (split_root / "summary.json").write_text(
        json.dumps(
            {
                "format": EpisodeCachedG1MotionDataset.FORMAT,
                "source_repo_id": SOURCE_REPO_ID,
                "source_revision": SOURCE_REVISION,
                "fps": 30.0,
                "window_size": 3,
                "num_prev_states": 2,
                "canonical_frame_idx": 1,
                "rotation_representation": "rot6d",
                "train_window_stride": 1,
                "episodes": 2,
                "frames": 8,
                "windows": 4,
                "shards": 1,
            }
        ),
        encoding="utf-8",
    )
    (split_root / "source_identity.json").write_text(
        json.dumps({"repo_id": SOURCE_REPO_ID, "revision": SOURCE_REVISION}),
        encoding="utf-8",
    )

    dataset = _open_cache(tmp_path / "cache")
    with pytest.raises(ValueError, match="identity mismatch"):
        EpisodeCachedG1MotionDataset(
            root=tmp_path / "cache",
            split="train",
            source_repo_id=SOURCE_REPO_ID,
            source_revision="different-revision",
        )
    assert len(dataset) == 4
    assert dataset.materialized_shard_spans() == [(0, 3), (3, 1)]
    assert dataset[0]["qpos_36"][:, 0].tolist() == [0.0, 1.0, 2.0]
    assert len(dataset._episode_cache) == 1
    dataset[1]
    assert len(dataset._episode_cache) == 1
    assert dataset[2]["qpos_36"][:, 0].tolist() == [2.0, 3.0, 4.0]
    assert dataset[3]["qpos_36"][:, 0].tolist() == [5.0, 6.0, 7.0]
    assert dataset[3]["caption"] == "stand"
    assert dataset[3]["motion_features"].shape == (3, 125)
    assert dataset[3]["mask"]["valid"].dtype == torch.bool


def test_episode_cache_writer_publishes_atomically(tmp_path) -> None:
    qpos = torch.zeros(8, 36)
    qpos[:, 0] = torch.arange(8)
    qpos[:, 3] = 1.0
    body_pos = torch.zeros(8, 30, 3)
    body_quat = torch.zeros(8, 30, 4)
    body_quat[..., 0] = 1.0

    class FakeDataset:
        repo_id = SOURCE_REPO_ID
        revision = SOURCE_REVISION
        default_fps = 30.0
        window_size = 3
        num_prev_states = 2
        train_window_stride = 1
        codec = SimpleNamespace(canonical_frame_idx=1, rotation_representation="rot6d")

        def iter_episode_kinematics_groups(self, *, max_frames, device, max_episodes, episode_start):
            assert max_frames == 16
            assert max_episodes is None
            assert episode_start == 0
            yield {
                "episodes": [
                    {"data_start_row": 0, "data_end_row": 5, "segment_caption": "walk"},
                    {"data_start_row": 5, "data_end_row": 8, "segment_caption": "stand"},
                ],
                "qpos_36": qpos.to(device),
                "body_pos_w": body_pos.to(device),
                "body_quat_w": body_quat.to(device),
            }

    output_root = tmp_path / "cache"
    summary = write_episode_cache(
        FakeDataset(),
        output_root=output_root,
        split="train",
        max_frames_per_shard=16,
        device="cpu",
        overwrite=False,
    )
    assert summary["episodes"] == 2
    assert summary["frames"] == 8
    assert summary["windows"] == 4
    assert not list(output_root.glob(".*.incomplete.*"))
    dataset = _open_cache(output_root)
    assert dataset[2]["qpos_36"][:, 0].tolist() == [2.0, 3.0, 4.0]
    report = inspect_episode_cache(output_root, "train")
    assert report["valid"] is True
    assert report["errors"] == []

    expected = torch.cat(
        [dataset[index]["motion_features"][dataset[index]["mask"]["valid"]] for index in range(len(dataset))]
    )
    single_rank = torch.cat(
        [
            batch["motion_features"][batch["valid_mask"]]
            for batch in dataset.iter_stats_batches(batch_size=2)
        ]
    )
    distributed = torch.cat(
        [
            batch["motion_features"][batch["valid_mask"]]
            for rank in range(2)
            for batch in dataset.iter_stats_batches(batch_size=2, rank=rank, world_size=2)
        ]
    )
    torch.testing.assert_close(single_rank, expected)
    torch.testing.assert_close(distributed, expected)


def test_episode_cache_writer_resumes_completed_shards(tmp_path) -> None:
    class ResumableDataset:
        repo_id = SOURCE_REPO_ID
        revision = SOURCE_REVISION
        default_fps = 30.0
        window_size = 3
        num_prev_states = 2
        train_window_stride = 1
        codec = SimpleNamespace(canonical_frame_idx=1, rotation_representation="rot6d")

        def __init__(self, fail_after_first: bool) -> None:
            self.fail_after_first = fail_after_first

        def iter_episode_kinematics_groups(self, *, max_frames, device, max_episodes, episode_start):
            for episode_index in range(episode_start, 2):
                qpos = torch.zeros(3, 36, device=device)
                qpos[:, 0] = torch.arange(3, device=device) + episode_index * 3
                qpos[:, 3] = 1.0
                body_pos = torch.zeros(3, 30, 3, device=device)
                body_quat = torch.zeros(3, 30, 4, device=device)
                body_quat[..., 0] = 1.0
                yield {
                    "episodes": [
                        {
                            "data_start_row": episode_index * 3,
                            "data_end_row": episode_index * 3 + 3,
                            "segment_caption": f"episode {episode_index}",
                        }
                    ],
                    "qpos_36": qpos,
                    "body_pos_w": body_pos,
                    "body_quat_w": body_quat,
                }
                if self.fail_after_first:
                    raise RuntimeError("simulated interruption")

    output_root = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        write_episode_cache(
            ResumableDataset(fail_after_first=True),
            output_root=output_root,
            split="train",
            max_frames_per_shard=3,
            device="cpu",
            overwrite=False,
        )
    assert (output_root / ".train.incomplete/shards/shard_00000/manifest.json").is_file()

    summary = write_episode_cache(
        ResumableDataset(fail_after_first=False),
        output_root=output_root,
        split="train",
        max_frames_per_shard=3,
        device="cpu",
        overwrite=False,
    )
    assert summary["episodes"] == 2
    assert summary["shards"] == 2
    dataset = _open_cache(output_root)
    assert dataset[1]["qpos_36"][:, 0].tolist() == [3.0, 4.0, 5.0]
