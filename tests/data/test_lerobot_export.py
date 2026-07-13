from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from omg.data import lerobot_export as lerobot_export_module
from omg.data.lerobot_export import export_lerobot
from omg.data.lerobot_dataset import LeRobotG1MotionDataset
from omg.data.lerobot_inspect import inspect_lerobot
from omg.data.episode_cache import EpisodeCachedG1MotionDataset
from omg.cli.data.materialize_episode_cache import write_episode_cache


def _minimal_npz(path: Path, frames: int = 8, *, with_humanref: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    qpos = np.zeros((frames, 36), dtype=np.float32)
    qpos[:, 0] = np.arange(frames, dtype=np.float32)
    qpos[:, 3] = 1.0
    payload = {"qpos": qpos, "fps": np.float32(30.0)}
    if with_humanref:
        payload["human_motion"] = np.ones((frames, 66), dtype=np.float32)
    np.savez(path, **payload)


def test_export_lerobot_writes_segment_episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "OMG-Data" / "omg_data" / "original" / "toy"
    _minimal_npz(root / "g1" / "clip01.npz", frames=10, with_humanref=True)
    (root / "music_npy").mkdir(parents=True)
    np.save(root / "music_npy" / "clip01.npy", np.ones((10, 35), dtype=np.float32))
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "clip01.json").write_text(
        json.dumps(
            {
                "video_summary": "toy motion",
                "segments": [{"start_frame": 2, "end_frame": 7, "action": "walk forward", "style": "steady"}],
            }
        ),
        encoding="utf-8",
    )
    (root / "info.yaml").write_text(yaml.safe_dump({"train": {"clip01": 1}}), encoding="utf-8")
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "dataset_opts": {
                    "train": {
                        "toy_train": {
                            "_target_": "omg.data.g1_motion.G1MotionDataset",
                            "dataset_root": "${paths.data_root}/omg_data/original/toy/g1",
                            "info_path": "${paths.data_root}/omg_data/original/toy/info.yaml",
                            "labels_root": "${paths.data_root}/omg_data/original/toy/labels",
                            "split": "train",
                            "sequence_duration": 2.0,
                            "fps": 30.0,
                            "sample_by_segment": True,
                            "include_style_in_caption": True,
                            "skip_missing_labels": False,
                            "use_text": True,
                            "use_audio": True,
                            "audio_dir": str(root / "music_npy"),
                            "audio_dim": 35,
                            "use_human_motion": True,
                            "human_motion_dim": 66,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text(yaml.safe_dump({"data_root": str(tmp_path / "OMG-Data")}), encoding="utf-8")
    repr_config = tmp_path / "repr.yaml"
    repr_config.write_text(yaml.safe_dump({"rotation_representation": "rot6d"}), encoding="utf-8")

    class Args:
        pass

    args = Args()
    args.data_config = data_config
    args.representation_config = repr_config
    args.paths_config = paths_config
    args.output_root = tmp_path / "lerobot"
    args.repo_id = "THU-MARS/OMG-Data"
    args.splits = ["train"]
    args.datasets = None
    args.fps = 30
    args.frames_per_file = 100
    args.episodes_per_file = 100
    args.max_entries_per_dataset = None
    args.max_episodes_per_dataset = None
    args.max_episodes = None
    args.overwrite = True

    lerobot_export_module._ensure_runtime_imports()
    parquet_paths: list[Path] = []
    text_paths: list[Path] = []
    original_write_table = lerobot_export_module.pq.write_table
    original_write_text = lerobot_export_module._atomic_write_text

    def tracked_write_table(table, path, *positional, **kwargs):
        incomplete_path = Path(path)
        parquet_paths.append(incomplete_path)
        assert incomplete_path.name.startswith(".")
        assert incomplete_path.name.endswith(".incomplete")
        final_path = incomplete_path.with_name(
            incomplete_path.name[1 : -len(".incomplete")]
        )
        assert not final_path.exists()
        return original_write_table(table, incomplete_path, *positional, **kwargs)

    def tracked_write_text(path: Path, content: str) -> None:
        original_write_text(path, content)
        text_paths.append(path.relative_to(args.output_root))

    monkeypatch.setattr(lerobot_export_module.pq, "write_table", tracked_write_table)
    monkeypatch.setattr(lerobot_export_module, "_atomic_write_text", tracked_write_text)

    summary = export_lerobot(args)

    assert summary["episodes"] == 1
    frames = pq.read_table(args.output_root / "data/chunk-000/file-000.parquet")
    episodes = pq.read_table(args.output_root / "meta/episodes/chunk-000/file-000.parquet")
    assert frames.num_rows == 5
    assert episodes.num_rows == 1
    assert frames.column("observation.state")[0].as_py()[0] == 2.0
    assert frames.column("action")[0].as_py()[0] == 3.0
    assert episodes.column("tasks")[0].as_py() == ["walk forward; style: steady"]
    assert (args.output_root / "meta/tasks.parquet").is_file()
    assert parquet_paths
    assert not list(args.output_root.rglob("*.incomplete"))
    assert text_paths[-1] == Path("meta/omg_manifest.json")

    dataset = LeRobotG1MotionDataset(
        dataset_root=args.output_root,
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
    assert sample["audio_features"][:5].count_nonzero().item() == 5 * 35
    assert sample["human_motion"][:5].count_nonzero().item() == 5 * 66

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
    cached_sample = EpisodeCachedG1MotionDataset(root=cache_root, split="train")[0]
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
    torch.testing.assert_close(cached_sample["canon_root_pos"], sample["canon_root_pos"])
    torch.testing.assert_close(cached_sample["canon_root_quat"], sample["canon_root_quat"])
    assert cached_sample["mask"]["valid"].equal(sample["mask"]["valid"])
    assert cached_sample["mask"]["has_audio"].equal(sample["mask"]["has_audio"])
    assert cached_sample["mask"]["has_human_motion"].equal(sample["mask"]["has_human_motion"])
    assert cached_sample["caption"] == sample["caption"]

    inspection = inspect_lerobot(args.output_root)
    assert inspection["episodes"] == 1
    assert inspection["frames"] == 5
    assert inspection["official_lerobot_v3"] is True
    assert inspection["empty_captions"] == 0
    assert inspection["datasets"][0]["dataset"] == "toy_train"
    assert inspection["datasets"][0]["length_frames"]["mean"] == 5.0
