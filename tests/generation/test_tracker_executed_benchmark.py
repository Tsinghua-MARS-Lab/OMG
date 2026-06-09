from __future__ import annotations

import argparse

import numpy as np
import pytest

from omg.benchmarks.runners.tracker_executed import (
    _pad_sequence_list,
    _select_tracker_indices,
    _tracker_metric_values,
    _write_tracker_artifacts,
    validate_tracker_executed_args,
)


def test_select_tracker_indices_is_deterministic_subset():
    first = _select_tracker_indices(10, 4, seed=7)
    second = _select_tracker_indices(10, 4, seed=7)
    assert first.tolist() == second.tolist()
    assert first.tolist() == sorted(first.tolist())
    assert len(set(first.tolist())) == 4
    assert _select_tracker_indices(3, None, seed=0).tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="only 3 generated"):
        _select_tracker_indices(3, 4, seed=0)


def test_pad_sequence_list_writes_valid_mask():
    padded, valid = _pad_sequence_list(
        [np.ones((2, 3), dtype=np.float32), np.full((4, 3), 2.0, dtype=np.float32)]
    )
    assert padded.shape == (2, 4, 3)
    assert valid.tolist() == [[True, True, False, False], [True, True, True, True]]
    assert np.allclose(padded[0, 2:], 0.0)
    with pytest.raises(ValueError, match="trailing shape"):
        _pad_sequence_list([np.zeros((2, 3)), np.zeros((2, 4))])


def test_tracker_metric_values_zero_for_identical_positions():
    positions = np.zeros((5, 4, 3), dtype=np.float32)
    positions[:, :, 0] = np.linspace(0.0, 1.0, 5, dtype=np.float32)[:, None]
    metrics = _tracker_metric_values(positions, positions, root_index=0)
    assert metrics == {"g_mpjpe": 0.0, "mpjpe": 0.0, "e_vel": 0.0, "e_acc": 0.0}


def test_validate_tracker_args_requires_existing_onnx(tmp_path):
    args = argparse.Namespace(
        tracker_executed=False,
        holomotion_onnx=None,
        tracker_num_samples=None,
        tracker_target_fps=50.0,
        tracker_steps=None,
        tracker_control_substeps=10,
        tracker_action_clip=10.0,
        tracker_root_index=0,
    )
    validate_tracker_executed_args(args)

    args.tracker_executed = True
    with pytest.raises(ValueError, match="requires --holomotion_onnx"):
        validate_tracker_executed_args(args)

    onnx_path = tmp_path / "tracker.onnx"
    onnx_path.write_bytes(b"onnx")
    args.holomotion_onnx = str(onnx_path)
    validate_tracker_executed_args(args)


def test_write_tracker_artifacts_schema(tmp_path):
    selected = np.asarray([0, 2], dtype=np.int64)
    generated = np.zeros((3, 4, 36), dtype=np.float32)
    reference = np.ones((3, 2, 36), dtype=np.float32)
    executed = [np.zeros((2, 36), dtype=np.float32), np.ones((3, 36), dtype=np.float32)]
    tracker_reference = [np.zeros((2, 36), dtype=np.float32), np.ones((3, 36), dtype=np.float32)]
    actions = [np.zeros((2, 12), dtype=np.float32), np.ones((3, 12), dtype=np.float32)]

    artifact_path, valid = _write_tracker_artifacts(
        stage_dir=tmp_path,
        selected_indices=selected,
        executed_qpos=executed,
        tracker_reference_qpos=tracker_reference,
        actions=actions,
        generated_qpos=generated,
        reference_qpos=reference,
        fps=np.asarray([30.0, 30.0, 30.0], dtype=np.float32),
        captions=["a", "b", "c"],
        dataset_names=["d0", "d1", "d2"],
        dataset_indices=[10, 11, 12],
        holomotion_onnx="tracker.onnx",
        tracker_fps=50.0,
    )
    assert artifact_path.exists()
    assert valid.tolist() == [[True, True, False], [True, True, True]]
    payload = np.load(artifact_path)
    assert payload["executed_qpos_36"].shape == (2, 3, 36)
    assert payload["tracker_reference_qpos_36"].shape == (2, 3, 36)
    assert payload["generated_qpos_36"].shape == (2, 4, 36)
    assert payload["reference_qpos_36"].shape == (2, 2, 36)
    assert payload["actions"].shape == (2, 3, 12)
    assert payload["sample_index"].tolist() == [0, 2]
    assert payload["dataset"].tolist() == ["d0", "d2"]
    assert payload["dataset_index"].tolist() == [10, 12]
