from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from omg.data.unified import UnifiedG1MotionIndex


def _minimal_npz(path: Path, frames: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    qpos = np.zeros((frames, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    np.savez(
        path,
        qpos=qpos,
        body_pos_w=np.zeros((frames, 30, 3), dtype=np.float32),
        body_quat_w=np.zeros((frames, 30, 4), dtype=np.float32),
        fps=np.float32(30.0),
    )


def _write_info(path: Path, *entries: str) -> None:
    path.write_text(yaml.safe_dump({"train": {entry: 1 for entry in entries}}), encoding="utf-8")


def test_unified_index_reads_json_segments_with_nested_entries(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    entry = "folderA/clip01_retarget__seg0000__part0000"
    _minimal_npz(root / "g1" / f"{entry}.npz", frames=90)
    label_path = root / "labels" / f"{entry}.json"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        json.dumps(
            {
                "video_summary": "summary caption",
                "segments": [
                    {
                        "start_time": 0.0,
                        "end_time": 2.0,
                        "action": "walk forward",
                        "style": "steady",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_info(root / "info.yaml", entry)

    idx = UnifiedG1MotionIndex(
        dataset_root=root / "g1",
        split="train",
        info_path=root / "info.yaml",
        labels_root=root / "labels",
        window_size=4,
        training=True,
    )

    assert len(idx.entries) == 1
    assert len(idx.samples) == 1
    sample = idx.samples[0]
    assert sample["segment_caption"] == "walk forward; style: steady"
    assert sample["segment_frame_start"] == 0
    assert sample["segment_frame_end"] == 60
    assert sample["label_path"] == str(label_path)


def test_unified_index_reads_txt_caption(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    entry = "clip02_retarget"
    _minimal_npz(root / "g1" / f"{entry}.npz")
    texts = root / "texts"
    texts.mkdir(parents=True)
    (texts / "clip02_retarget.txt").write_text("first caption\nsecond caption\n", encoding="utf-8")
    _write_info(root / "info.yaml", entry)

    idx = UnifiedG1MotionIndex(
        dataset_root=root / "g1",
        split="train",
        info_path=root / "info.yaml",
        text_root=texts,
        window_size=4,
        training=True,
    )

    assert len(idx.samples) == 1
    assert idx.samples[0]["segment_caption"] == "first caption second caption"
    assert idx.samples[0]["text_path"] == str(texts / "clip02_retarget.txt")


def test_unified_index_skips_missing_labels_when_requested(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _minimal_npz(root / "g1" / "missing_retarget.npz")
    _minimal_npz(root / "g1" / "ok_retarget.npz")
    labels = root / "labels"
    labels.mkdir(parents=True)
    (labels / "ok_retarget.json").write_text(json.dumps({"caption": "kept caption"}), encoding="utf-8")
    _write_info(root / "info.yaml", "missing_retarget", "ok_retarget")

    idx = UnifiedG1MotionIndex(
        dataset_root=root / "g1",
        split="train",
        info_path=root / "info.yaml",
        labels_root=labels,
        window_size=4,
        training=True,
        skip_missing_labels=True,
    )

    assert len(idx.entries) == 2
    assert len(idx.samples) == 1
    assert idx.samples[0]["segment_caption"] == "kept caption"
