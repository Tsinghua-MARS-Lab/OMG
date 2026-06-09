"""Audio-conditioned motion benchmark metrics.

Implements:
- BeatAlign: music-beat to motion-beat alignment using kinetic-velocity minima.
- AIST++/EDGE-style FIDk/FIDg and Divk/Divg over kinetic/geometric dance features.
- EDGE Physical Foot Contact (PFC) score.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from scipy.signal import find_peaks

from omg.benchmarks.metrics.distribution import diversity, motion_fid

BeatAlignDirection = Literal["music_to_motion", "motion_to_music"]


def _as_batched_features(x: np.ndarray, *, name: str, last_dim: int | None = None) -> tuple[np.ndarray, bool]:
    array = np.asarray(x, dtype=np.float32)
    single = array.ndim == 2
    if single:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (T, D) or (B, T, D), got {tuple(array.shape)}")
    if last_dim is not None and array.shape[-1] != last_dim:
        raise ValueError(f"{name} must have last dimension {last_dim}, got {array.shape[-1]}")
    return array, single


def _as_batched_positions(motion_positions: np.ndarray) -> tuple[np.ndarray, bool]:
    positions = np.asarray(motion_positions, dtype=np.float32)
    single = positions.ndim == 3
    if single:
        positions = positions[None]
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError(
            "motion_positions must have shape (T, J, 3) or (B, T, J, 3); "
            "root-only positions are not accepted for BeatAlign"
        )
    if positions.shape[-2] <= 1:
        raise ValueError("motion_positions must contain more than one body/joint for kinetic velocity")
    return positions, single


def _as_batched_beats(beats: np.ndarray | list[np.ndarray], *, batch_size: int, name: str) -> list[np.ndarray]:
    if isinstance(beats, list):
        if len(beats) != batch_size:
            raise ValueError(f"{name} list length must match batch size {batch_size}, got {len(beats)}")
        return [np.asarray(item, dtype=np.float64).reshape(-1) for item in beats]

    array = np.asarray(beats)
    if array.ndim == 1:
        if batch_size != 1:
            raise ValueError(f"{name} with shape (N,) is only valid for unbatched inputs")
        return [array.astype(np.float64, copy=False).reshape(-1)]
    if array.ndim == 2:
        if array.shape[0] != batch_size:
            raise ValueError(f"{name} first dimension must match batch size {batch_size}, got {array.shape[0]}")
        return [array[idx].astype(np.float64, copy=False).reshape(-1) for idx in range(batch_size)]
    raise ValueError(f"{name} must be an array with shape (N,) or (B, N), or a list of arrays")


def audio_beats_from_features(audio_features: np.ndarray, *, threshold: float = 0.5) -> list[np.ndarray] | np.ndarray:
    """Extract audio beat frame indices from the last feature channel.

    The AIST++/FineDance-style feature convention used in this repo stores a
    beat one-hot/probability in the last channel. For batched input this returns
    a list of arrays; for unbatched input it returns a single array.
    """
    features, single = _as_batched_features(audio_features, name="audio_features")
    beats = [np.where(features[idx, :, -1] > float(threshold))[0].astype(np.float64) for idx in range(features.shape[0])]
    return beats[0] if single else beats


def motion_beats_from_positions(
    motion_positions: np.ndarray,
    *,
    fps: float,
    min_distance_seconds: float = 0.25,
) -> list[np.ndarray] | np.ndarray:
    """Estimate dance/kinematic beat frame indices from kinetic velocity minima.

    ``motion_positions`` must be body/joint positions ``(T, J, 3)`` or
    ``(B, T, J, 3)``. The kinetic signal is the mean body/joint speed per frame;
    local minima are treated as dance beats, following the common AIST++ /
    Bailando / EDGE BeatAlign setup.
    """
    positions, single = _as_batched_positions(motion_positions)
    velocity = np.linalg.norm(np.diff(positions, axis=1), axis=-1).mean(axis=-1)
    distance = max(int(round(float(min_distance_seconds) * float(fps))), 1)
    beats = [find_peaks(-velocity[idx], distance=distance)[0].astype(np.float64) for idx in range(velocity.shape[0])]
    return beats[0] if single else beats


def _score_source_to_target(
    source_frames: np.ndarray,
    target_frames: np.ndarray,
    *,
    source_fps: float,
    target_fps: float,
    sigma_seconds: float,
) -> float:
    source = np.asarray(source_frames, dtype=np.float64).reshape(-1)
    target = np.asarray(target_frames, dtype=np.float64).reshape(-1)
    if source.size == 0 or target.size == 0:
        return 0.0
    source_times = source / float(source_fps)
    target_times = target / float(target_fps)
    distances = np.min(np.abs(source_times[:, None] - target_times[None, :]), axis=1)
    scores = np.exp(-(distances ** 2) / (2.0 * float(sigma_seconds) ** 2))
    return float(scores.mean())


def beat_align_from_beats(
    *,
    audio_beats: np.ndarray | list[np.ndarray],
    motion_beats: np.ndarray | list[np.ndarray],
    fps: float,
    audio_fps: float | None = None,
    sigma_frames: float = 3.0,
    direction: BeatAlignDirection = "music_to_motion",
) -> float | np.ndarray:
    """Compute BeatAlign from precomputed beat frame indices.

    The default ``music_to_motion`` direction matches Bailando/EDGE: each music
    beat is matched to the closest dance beat. ``motion_to_music`` is the
    AIST++/FACT-style diagnostic direction.
    """
    audio_fps = float(audio_fps or fps)
    motion_fps = float(fps)
    sigma_seconds = float(sigma_frames) / motion_fps
    batch_size = len(audio_beats) if isinstance(audio_beats, list) else (1 if np.asarray(audio_beats).ndim == 1 else np.asarray(audio_beats).shape[0])
    audio_list = _as_batched_beats(audio_beats, batch_size=batch_size, name="audio_beats")
    motion_list = _as_batched_beats(motion_beats, batch_size=batch_size, name="motion_beats")

    scores = []
    for audio, motion in zip(audio_list, motion_list):
        if direction == "music_to_motion":
            scores.append(_score_source_to_target(audio, motion, source_fps=audio_fps, target_fps=motion_fps, sigma_seconds=sigma_seconds))
        elif direction == "motion_to_music":
            scores.append(_score_source_to_target(motion, audio, source_fps=motion_fps, target_fps=audio_fps, sigma_seconds=sigma_seconds))
        else:
            raise ValueError(f"Unsupported BeatAlign direction: {direction}")
    result = np.asarray(scores, dtype=np.float64)
    return float(result[0]) if result.shape[0] == 1 else result


def beat_align(
    *,
    audio_features: np.ndarray,
    motion_positions: np.ndarray,
    fps: float,
    audio_fps: float | None = None,
    sigma_frames: float = 3.0,
    direction: BeatAlignDirection = "music_to_motion",
    beat_threshold: float = 0.5,
    min_motion_beat_distance_seconds: float = 0.25,
) -> float | np.ndarray:
    """Compute paper-style BeatAlign between audio beats and dance beats.

    Motion beats are extracted from kinetic-velocity local minima over
    body/joint positions. Root-only positions are intentionally unsupported so
    this API cannot be mistaken for a paper-faithful BeatAlign implementation.
    Defaults to the Bailando/EDGE convention: music beat -> closest dance beat.
    Set ``direction=\"motion_to_music\"`` for AIST++/FACT-style diagnostics.
    """
    audio_beats = audio_beats_from_features(audio_features, threshold=beat_threshold)
    motion_beats = motion_beats_from_positions(
        motion_positions,
        fps=fps,
        min_distance_seconds=min_motion_beat_distance_seconds,
    )
    return beat_align_from_beats(
        audio_beats=audio_beats,
        motion_beats=motion_beats,
        fps=fps,
        audio_fps=audio_fps,
        sigma_frames=sigma_frames,
        direction=direction,
    )


_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def axis_index(axis: str | int) -> int:
    if isinstance(axis, str):
        key = axis.lower()
        if key not in _AXIS_TO_INDEX:
            raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
        return _AXIS_TO_INDEX[key]
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    return axis


def _as_positions(x: np.ndarray | Sequence[Any]) -> np.ndarray:
    positions = np.asarray(x, dtype=np.float64)
    if positions.ndim == 3:
        positions = positions[None]
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError(f"positions must have shape (B, T, J, 3) or (T, J, 3), got {positions.shape}")
    if positions.shape[1] < 2:
        raise ValueError("positions must contain at least two frames")
    if positions.shape[2] < 2:
        raise ValueError("positions must contain at least two bodies/joints")
    return positions


def _fps_array(fps: float | Sequence[float] | np.ndarray, batch_size: int) -> np.ndarray:
    arr = np.asarray(fps, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(batch_size, float(arr), dtype=np.float64)
    if arr.shape != (batch_size,):
        raise ValueError(f"fps must be scalar or shape ({batch_size},), got {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("fps must contain positive finite values")
    return arr


def _root_relative(positions: np.ndarray, root_index: int) -> np.ndarray:
    root_index = int(root_index)
    if root_index < 0 or root_index >= positions.shape[2]:
        raise ValueError(f"root_index {root_index} out of range for {positions.shape[2]} joints")
    return positions - positions[:, :, root_index : root_index + 1, :]


def _average_velocity(
    positions: np.ndarray,
    frame: int,
    joint: int,
    sliding_window: int,
    dt: float,
    axes: Sequence[int],
) -> float:
    current_window = 0
    average_velocity = np.zeros(3, dtype=np.float64)
    for offset in range(-sliding_window, sliding_window + 1):
        curr = frame + offset
        prev = curr - 1
        if prev < 0 or curr >= positions.shape[0]:
            continue
        average_velocity += positions[curr, joint] - positions[prev, joint]
        current_window += 1
    if current_window == 0:
        return 0.0
    velocity = average_velocity[list(axes)] / (current_window * dt)
    return float(np.linalg.norm(velocity))


def _average_acceleration(
    positions: np.ndarray,
    frame: int,
    joint: int,
    sliding_window: int,
    dt: float,
) -> float:
    current_window = 0
    average_acceleration = np.zeros(3, dtype=np.float64)
    for offset in range(-sliding_window, sliding_window + 1):
        prev = frame + offset - 1
        curr = frame + offset
        nxt = frame + offset + 1
        if prev < 0 or nxt >= positions.shape[0]:
            continue
        v2 = (positions[nxt, joint] - positions[curr, joint]) / dt
        v1 = (positions[curr, joint] - positions[prev, joint]) / dt
        average_acceleration += (v2 - v1) / dt
        current_window += 1
    if current_window == 0:
        return 0.0
    return float(np.linalg.norm(average_acceleration / current_window))


def kinetic_features(
    positions: np.ndarray | Sequence[Any],
    *,
    fps: float | Sequence[float] | np.ndarray,
    root_index: int = 0,
    up_axis: str | int = "z",
    sliding_window: int = 2,
) -> np.ndarray:
    """AIST++/fairmotion-style per-joint kinetic feature vectors.

    The feature vector contains, for every body/joint, average horizontal kinetic
    energy, average vertical kinetic energy, and average acceleration magnitude.
    Positions are root-relative, matching fairmotion's ``position_wrt_root``.
    """
    pos = _root_relative(_as_positions(positions), int(root_index))
    up = axis_index(up_axis)
    flat_axes = tuple(axis for axis in range(3) if axis != up)
    fps_values = _fps_array(fps, pos.shape[0])
    sliding_window = int(sliding_window)
    if sliding_window < 0:
        raise ValueError("sliding_window must be non-negative")

    out = np.zeros((pos.shape[0], pos.shape[2] * 3), dtype=np.float64)
    for batch in range(pos.shape[0]):
        dt = 1.0 / float(fps_values[batch])
        values: list[float] = []
        for joint in range(pos.shape[2]):
            horizontal = 0.0
            vertical = 0.0
            acceleration = 0.0
            for frame in range(1, pos.shape[1]):
                h_vel = _average_velocity(pos[batch], frame, joint, sliding_window, dt, flat_axes)
                v_vel = _average_velocity(pos[batch], frame, joint, sliding_window, dt, (up,))
                horizontal += h_vel * h_vel
                vertical += v_vel * v_vel
                acceleration += _average_acceleration(pos[batch], frame, joint, sliding_window, dt)
            denom = float(pos.shape[1] - 1)
            values.extend([horizontal / denom, vertical / denom, acceleration / denom])
        out[batch] = np.asarray(values, dtype=np.float64)
    return out.astype(np.float32)


def _validate_edges(parent_edges: Sequence[tuple[int, int]] | None, joints: int) -> list[tuple[int, int]]:
    if parent_edges is None:
        return []
    edges = [(int(parent), int(child)) for parent, child in parent_edges]
    for parent, child in edges:
        if parent < 0 or child < 0 or parent >= joints or child >= joints:
            raise ValueError(f"Invalid parent edge ({parent}, {child}) for {joints} joints")
        if parent == child:
            raise ValueError(f"Invalid self edge ({parent}, {child})")
    return edges


def geometric_features(
    positions: np.ndarray | Sequence[Any],
    *,
    root_index: int = 0,
    parent_edges: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """G1 topology geometry features for AIST++/EDGE-style FIDg/Divg.

    AIST++ uses hand-picked fairmotion manual features for SMPL. G1 has a
    different topology, so this extractor uses only faithful G1 geometry:
    temporal mean/std of root-relative body positions plus mean/std of kinematic
    edge lengths and directions when parent edges are supplied.
    """
    pos = _root_relative(_as_positions(positions), int(root_index))
    edges = _validate_edges(parent_edges, pos.shape[2])
    features = []
    for sample in pos:
        pieces = [sample.mean(axis=0).reshape(-1), sample.std(axis=0).reshape(-1)]
        if edges:
            parent = np.asarray([edge[0] for edge in edges], dtype=np.int64)
            child = np.asarray([edge[1] for edge in edges], dtype=np.int64)
            bone = sample[:, child] - sample[:, parent]
            lengths = np.linalg.norm(bone, axis=-1)
            directions = bone / np.maximum(lengths[..., None], 1e-8)
            pieces.extend(
                [
                    lengths.mean(axis=0),
                    lengths.std(axis=0),
                    directions.mean(axis=0).reshape(-1),
                    directions.std(axis=0).reshape(-1),
                ]
            )
        features.append(np.concatenate(pieces, axis=0))
    return np.asarray(features, dtype=np.float32)


def physical_foot_contact_scores(
    positions: np.ndarray | Sequence[Any],
    *,
    fps: float | Sequence[float] | np.ndarray,
    root_index: int = 0,
    left_foot_indices: Sequence[int] = (5, 6),
    right_foot_indices: Sequence[int] = (11, 12),
    up_axis: str | int = "z",
) -> np.ndarray:
    """EDGE Physical Foot Contact score per sequence.

    This follows EDGE's PFC equation: normalized positive-up root acceleration
    multiplied by the minimum horizontal foot speed on each side, averaged over
    frames and scaled by 10000. Lower is better.
    """
    pos = _as_positions(positions)
    up = axis_index(up_axis)
    flat_axes = [axis for axis in range(3) if axis != up]
    fps_values = _fps_array(fps, pos.shape[0])
    root_index = int(root_index)
    left = np.asarray([int(index) for index in left_foot_indices], dtype=np.int64)
    right = np.asarray([int(index) for index in right_foot_indices], dtype=np.int64)
    if left.size == 0 or right.size == 0:
        raise ValueError("left_foot_indices and right_foot_indices must be non-empty")
    if root_index < 0 or root_index >= pos.shape[2]:
        raise ValueError(f"root_index {root_index} out of range for {pos.shape[2]} joints")
    all_feet = np.concatenate([left, right])
    if np.any(all_feet < 0) or np.any(all_feet >= pos.shape[2]):
        raise ValueError(f"foot index out of range for {pos.shape[2]} joints")
    if pos.shape[1] < 3:
        raise ValueError("PFC requires at least three frames")

    scores = []
    for batch in range(pos.shape[0]):
        dt = 1.0 / float(fps_values[batch])
        root_v = (pos[batch, 1:, root_index] - pos[batch, :-1, root_index]) / dt
        root_a = (root_v[1:] - root_v[:-1]) / dt
        root_a[:, up] = np.maximum(root_a[:, up], 0.0)
        root_a_norm = np.linalg.norm(root_a, axis=-1)
        scale = float(root_a_norm.max())
        if scale > 0.0:
            root_a_norm = root_a_norm / scale
        else:
            root_a_norm = np.zeros_like(root_a_norm)

        left_pos = np.take(pos[batch], left, axis=1)
        right_pos = np.take(pos[batch], right, axis=1)
        left_v = np.linalg.norm(left_pos[2:, :, :][:, :, flat_axes] - left_pos[1:-1, :, :][:, :, flat_axes], axis=-1)
        right_v = np.linalg.norm(right_pos[2:, :, :][:, :, flat_axes] - right_pos[1:-1, :, :][:, :, flat_axes], axis=-1)
        left_min = left_v.min(axis=1)
        right_min = right_v.min(axis=1)
        scores.append(float((left_min * right_min * root_a_norm).mean() * 10000.0))
    return np.asarray(scores, dtype=np.float64)


def summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] == 0:
        raise ValueError("summary_stats requires a non-empty 1D array")
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def aistpp_edge_features(
    reference_positions: np.ndarray | Sequence[Any],
    generated_positions: np.ndarray | Sequence[Any],
    *,
    fps: float | Sequence[float] | np.ndarray,
    root_index: int = 0,
    up_axis: str | int = "z",
    parent_edges: Sequence[tuple[int, int]] | None = None,
    left_foot_indices: Sequence[int] = (5, 6),
    right_foot_indices: Sequence[int] = (11, 12),
) -> dict[str, np.ndarray]:
    reference = _as_positions(reference_positions)
    generated = _as_positions(generated_positions)
    if reference.shape != generated.shape:
        raise ValueError(f"reference and generated positions must have the same shape, got {reference.shape} and {generated.shape}")
    fps_values = _fps_array(fps, reference.shape[0])
    return {
        "reference_kinetic": kinetic_features(reference, fps=fps_values, root_index=root_index, up_axis=up_axis),
        "generated_kinetic": kinetic_features(generated, fps=fps_values, root_index=root_index, up_axis=up_axis),
        "reference_geometric": geometric_features(reference, root_index=root_index, parent_edges=parent_edges),
        "generated_geometric": geometric_features(generated, root_index=root_index, parent_edges=parent_edges),
        "reference_pfc": physical_foot_contact_scores(
            reference,
            fps=fps_values,
            root_index=root_index,
            left_foot_indices=left_foot_indices,
            right_foot_indices=right_foot_indices,
            up_axis=up_axis,
        ),
        "generated_pfc": physical_foot_contact_scores(
            generated,
            fps=fps_values,
            root_index=root_index,
            left_foot_indices=left_foot_indices,
            right_foot_indices=right_foot_indices,
            up_axis=up_axis,
        ),
    }


def aistpp_edge_metric_summary(
    features: dict[str, np.ndarray],
    *,
    indices: np.ndarray | Sequence[int] | None = None,
    diversity_pairs: int = 300,
) -> dict[str, Any]:
    if indices is None:
        selected = None
    else:
        selected = np.asarray(indices, dtype=np.int64)

    def pick(key: str) -> np.ndarray:
        value = np.asarray(features[key])
        return value if selected is None else value[selected]

    ref_k = pick("reference_kinetic")
    gen_k = pick("generated_kinetic")
    ref_g = pick("reference_geometric")
    gen_g = pick("generated_geometric")
    ref_pfc = pick("reference_pfc")
    gen_pfc = pick("generated_pfc")

    result: dict[str, Any] = {
        "num_samples": int(gen_k.shape[0]),
        "kinetic_feature_dim": int(gen_k.shape[1]),
        "geometric_feature_dim": int(gen_g.shape[1]),
        "pfc_generated": summary_stats(gen_pfc),
        "pfc_reference": summary_stats(ref_pfc),
        "pfc_gap": summary_stats(gen_pfc - ref_pfc),
    }
    if gen_k.shape[0] < 2 or ref_k.shape[0] < 2:
        result.update(
            {
                "fid_k": None,
                "fid_g": None,
                "div_k_generated": None,
                "div_g_generated": None,
                "div_k_reference": None,
                "div_g_reference": None,
                "div_k_gap": None,
                "div_g_gap": None,
            }
        )
        return result
    div_k_generated = diversity(gen_k, num_pairs=diversity_pairs)
    div_g_generated = diversity(gen_g, num_pairs=diversity_pairs)
    div_k_reference = diversity(ref_k, num_pairs=diversity_pairs)
    div_g_reference = diversity(ref_g, num_pairs=diversity_pairs)
    result.update(
        {
            "fid_k": motion_fid(ref_k, gen_k),
            "fid_g": motion_fid(ref_g, gen_g),
            "div_k_generated": div_k_generated,
            "div_g_generated": div_g_generated,
            "div_k_reference": div_k_reference,
            "div_g_reference": div_g_reference,
            "div_k_gap": div_k_generated - div_k_reference,
            "div_g_gap": div_g_generated - div_g_reference,
        }
    )
    return result
