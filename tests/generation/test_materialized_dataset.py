from __future__ import annotations

import json

import numpy as np
import torch

from omg.data.datamodule import motion_collate_fn
from omg.data.materialized import MASK_KEYS, MaterializedG1MotionDataset, TENSOR_KEYS


def test_materialized_dataset_returns_generation_sample(tmp_path):
    root = tmp_path / "materialized"
    split_root = root / "train"
    split_root.mkdir(parents=True)
    arrays = {}
    for key in TENSOR_KEYS:
        if key in {"length", "canonical_frame_idx"}:
            arrays[key] = np.asarray([2], dtype=np.int64)
        elif key == "fps":
            arrays[key] = np.asarray([30.0], dtype=np.float32)
        elif key == "has_text":
            arrays[key] = np.asarray([True], dtype=np.bool_)
        elif key in {"qpos_36"}:
            arrays[key] = np.zeros((1, 3, 36), dtype=np.float32)
        elif key == "body_pos_w":
            arrays[key] = np.zeros((1, 3, 30, 3), dtype=np.float32)
        elif key == "body_quat_w":
            arrays[key] = np.zeros((1, 3, 30, 4), dtype=np.float32)
        elif key == "audio_features":
            arrays[key] = np.zeros((1, 3, 35), dtype=np.float32)
        elif key == "human_motion":
            arrays[key] = np.zeros((1, 3, 66), dtype=np.float32)
        elif key in {"motion_features", "prev_state_features", "history_features"}:
            arrays[key] = np.zeros((1, 3, 125), dtype=np.float32)
        elif key == "root_pos_local":
            arrays[key] = np.zeros((1, 3, 3), dtype=np.float32)
        elif key == "root_rot_local_quat":
            arrays[key] = np.zeros((1, 3, 4), dtype=np.float32)
        elif key == "joint_dof":
            arrays[key] = np.zeros((1, 3, 29), dtype=np.float32)
        elif key == "body_link_pos_local":
            arrays[key] = np.zeros((1, 3, 29, 3), dtype=np.float32)
        elif key == "canon_root_pos":
            arrays[key] = np.zeros((1, 1, 3), dtype=np.float32)
        elif key == "canon_root_quat":
            arrays[key] = np.zeros((1, 1, 4), dtype=np.float32)
        else:
            raise AssertionError(key)
    for key in MASK_KEYS:
        arrays[f"mask__{key}"] = np.asarray([[True, True, False]], dtype=np.bool_)
    np.savez(split_root / "shard_00000.npz", **arrays)
    (split_root / "shard_00000.json").write_text(
        json.dumps([{"caption": "walk forward", "meta": {"sequence_name": "sample"}}]),
        encoding="utf-8",
    )
    (split_root / "index.jsonl").write_text(json.dumps({"shard": "shard_00000.npz", "offset": 0}) + "\n", encoding="utf-8")

    dataset = MaterializedG1MotionDataset(root=root, split="train")
    sample = dataset[0]
    assert sample["caption"] == "walk forward"
    assert sample["motion_features"].shape == (3, 125)
    assert sample["human_motion"].shape == (3, 66)
    assert sample["mask"]["valid"].dtype == torch.bool
    batch = motion_collate_fn([sample, sample])
    assert batch["motion_features"].shape == (2, 3, 125)
    assert batch["mask"]["valid"].tolist() == [[True, True, False], [True, True, False]]
