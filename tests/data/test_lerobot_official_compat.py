from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

lerobot = pytest.importorskip("lerobot")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from omg.data.lerobot_export import LeRobotV3ExportWriter


def test_official_lerobot_loader_accepts_omg_export(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    writer = LeRobotV3ExportWriter(
        root,
        repo_id="THU-MARS/OMG-Data",
        frames_per_file=100,
        episodes_per_file=100,
        fps=30,
        include_audio=False,
        audio_dim=35,
        include_humanref=False,
        humanref_dim=66,
        overwrite=False,
    )
    qpos = np.zeros((5, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    writer.add_episode(
        dataset_name="toy_train",
        split="train",
        source_id="toy",
        segment_index=0,
        source_start_frame=0,
        source_end_frame=5,
        fps=30.0,
        qpos=qpos,
        caption="walk forward",
    )
    writer.close()

    dataset = LeRobotDataset(repo_id="THU-MARS/OMG-Data", root=root, download_videos=False)
    assert len(dataset) == 5
    assert dataset[0]["task"] == "walk forward"
    assert tuple(dataset[0]["observation.state"].shape) == (36,)
