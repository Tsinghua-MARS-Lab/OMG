from __future__ import annotations

import numpy as np
import pytest

from omg.benchmarks.metrics import (
    aistpp_edge_features,
    aistpp_edge_metric_summary,
    beat_align_from_beats,
    beat_align,
    e_acc,
    e_vel,
    g_mpjpe,
    geometric_features,
    kinetic_features,
    motion_beats_from_positions,
    mpjpe,
    physical_foot_contact_scores,
)


def _audio_features(length: int, beats: list[int]) -> np.ndarray:
    features = np.zeros((length, 4), dtype=np.float32)
    features[beats, -1] = 1.0
    return features


def _motion_positions_with_velocity_minimum(length: int, beat_frame: int, joints: int = 2) -> np.ndarray:
    positions = np.zeros((length, joints, 3), dtype=np.float32)
    x = np.arange(length, dtype=np.float32)
    x[beat_frame + 1 :] -= 1.0
    positions[:, :, 0] = x[:, None]
    positions[:, 1, 1] = x
    return positions


def test_beat_align_defaults_to_bailando_edge_music_to_motion_direction():
    # Three music beats, but the dance only hits one of them. Music->motion must
    # penalize the missed music beats more than motion->music.
    music_to_motion = beat_align_from_beats(
        audio_beats=np.array([10, 30, 50]),
        motion_beats=np.array([10]),
        fps=60.0,
        sigma_frames=3.0,
        direction="music_to_motion",
    )
    motion_to_music = beat_align_from_beats(
        audio_beats=np.array([10, 30, 50]),
        motion_beats=np.array([10]),
        fps=60.0,
        sigma_frames=3.0,
        direction="motion_to_music",
    )
    assert music_to_motion < motion_to_music
    assert motion_to_music == pytest.approx(1.0)


def test_beat_align_uses_gaussian_frame_sigma():
    # 3-frame offset with sigma=3 gives exp(-0.5).
    score = beat_align_from_beats(
        audio_beats=np.array([10]),
        motion_beats=np.array([13]),
        fps=60.0,
        sigma_frames=3.0,
        direction="music_to_motion",
    )
    assert score == pytest.approx(np.exp(-0.5))


def test_beat_align_batch_returns_per_sample_scores_without_dropping_items():
    scores = beat_align_from_beats(
        audio_beats=[np.array([10]), np.array([10, 30])],
        motion_beats=[np.array([10]), np.array([10])],
        fps=60.0,
        sigma_frames=3.0,
        direction="music_to_motion",
    )
    assert scores.shape == (2,)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] < scores[0]


def test_motion_beats_are_extracted_from_kinetic_velocity_minima():
    motion_positions = _motion_positions_with_velocity_minimum(64, beat_frame=10)
    beats = motion_beats_from_positions(motion_positions, fps=60.0, min_distance_seconds=0.1)
    assert beats.tolist() == [10]


def test_beat_align_wrapper_requires_body_or_joint_positions():
    audio = _audio_features(64, [10, 30, 50])
    root = np.zeros((64, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="root-only positions are not accepted"):
        beat_align(audio_features=audio, motion_positions=root, fps=60.0)


def test_beat_align_wrapper_uses_kinetic_velocity_minima_and_default_direction():
    audio = _audio_features(64, [10, 30, 50])
    motion_positions = _motion_positions_with_velocity_minimum(64, beat_frame=10)
    score = beat_align(audio_features=audio, motion_positions=motion_positions, fps=60.0)
    assert 0.0 <= score <= 1.0


def test_g_mpjpe_measures_global_body_position_error_in_mm():
    ref = np.zeros((2, 5, 3, 3), dtype=np.float32)
    pred = ref.copy()
    pred[..., 0] += 0.001
    assert g_mpjpe(pred, ref) == pytest.approx(1.0)


def test_mpjpe_is_root_relative_body_position_error_in_mm():
    ref = np.zeros((5, 2, 3), dtype=np.float32)
    pred = ref.copy()
    pred[:, :, 0] += 0.001
    assert g_mpjpe(pred, ref) == pytest.approx(1.0)
    assert mpjpe(pred, ref) == pytest.approx(0.0)

    pred[:, 1, 0] += 0.002
    assert mpjpe(pred, ref) == pytest.approx(1.0)


def test_tracking_velocity_and_acceleration_errors_use_frame_differences_in_mm():
    ref = np.zeros((4, 1, 3), dtype=np.float32)
    pred = ref.copy()
    pred[:, 0, 0] = np.array([0.0, 0.001, 0.003, 0.006], dtype=np.float32)

    assert e_vel(pred, ref) == pytest.approx(2.0)
    assert e_acc(pred, ref) == pytest.approx(1.0)


def test_tracking_metrics_reject_flat_motion_vectors():
    pred = np.zeros((2, 5, 90), dtype=np.float32)
    ref = np.zeros((2, 5, 90), dtype=np.float32)
    with pytest.raises(ValueError, match="must have shape"):
        mpjpe(pred, ref)



def test_aistpp_kinetic_features_are_zero_for_static_motion():
    positions = np.zeros((1, 5, 2, 3), dtype=np.float32)
    features = kinetic_features(positions, fps=np.array([30.0]), up_axis="z")
    assert features.shape == (1, 6)
    assert np.allclose(features, 0.0)


def test_aistpp_kinetic_features_measure_root_relative_horizontal_energy():
    positions = np.zeros((1, 6, 2, 3), dtype=np.float32)
    positions[0, :, 1, 0] = np.arange(6, dtype=np.float32) / 30.0
    features = kinetic_features(positions, fps=np.array([30.0]), up_axis="z", sliding_window=1)
    assert features[0, 3] == pytest.approx(1.0)
    assert features[0, 4] == pytest.approx(0.0)
    assert features[0, 5] == pytest.approx(0.0, abs=1e-5)


def test_g1_geometric_features_use_supplied_topology_edges():
    positions = np.zeros((2, 4, 3, 3), dtype=np.float32)
    positions[:, :, 1, 0] = 1.0
    positions[:, :, 2, 1] = 1.0
    no_edges = geometric_features(positions, root_index=0)
    with_edges = geometric_features(positions, root_index=0, parent_edges=[(0, 1), (1, 2)])
    assert no_edges.shape[0] == 2
    assert with_edges.shape[1] > no_edges.shape[1]


def test_edge_pfc_is_zero_for_static_contact():
    positions = np.zeros((1, 5, 3, 3), dtype=np.float32)
    scores = physical_foot_contact_scores(
        positions,
        fps=np.array([30.0]),
        root_index=0,
        left_foot_indices=[1],
        right_foot_indices=[2],
    )
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(0.0)



def test_edge_pfc_accepts_multiple_foot_points_per_side():
    positions = np.zeros((1, 6, 5, 3), dtype=np.float32)
    positions[0, :, 0, 2] = np.linspace(0.0, 0.1, 6)
    positions[0, :, 1, 0] = np.linspace(0.0, 0.05, 6)
    positions[0, :, 2, 0] = np.linspace(0.0, 0.02, 6)
    positions[0, :, 3, 1] = np.linspace(0.0, 0.03, 6)
    positions[0, :, 4, 1] = np.linspace(0.0, 0.01, 6)
    scores = physical_foot_contact_scores(
        positions,
        fps=np.array([30.0]),
        root_index=0,
        left_foot_indices=[1, 2],
        right_foot_indices=[3, 4],
    )
    assert scores.shape == (1,)
    assert np.isfinite(scores[0])


def test_aistpp_edge_metric_summary_contains_fid_diversity_and_pfc():
    rng = np.random.default_rng(123)
    reference = rng.normal(size=(4, 8, 3, 3)).astype(np.float32)
    generated = reference.copy()
    generated[:, :, 1, 0] += 0.05
    features = aistpp_edge_features(
        reference,
        generated,
        fps=np.full(4, 30.0),
        root_index=0,
        parent_edges=[(0, 1), (1, 2)],
        left_foot_indices=[1],
        right_foot_indices=[2],
    )
    metrics = aistpp_edge_metric_summary(features)
    assert np.isfinite(metrics["fid_k"])
    assert np.isfinite(metrics["fid_g"])
    assert np.isfinite(metrics["div_k_generated"])
    assert np.isfinite(metrics["div_g_generated"])
    assert np.isfinite(metrics["pfc_generated"]["mean"])
