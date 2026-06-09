"""Autoregressive transition smoothness metrics.

Implements TextOp/temporal-composition style transition metrics:
- PJ: peak L1 jerk over all links and transition frames.
- AUJ: area under jerk deviation from the reference dataset jerk level.

For autoregressive generation, transition windows are centered on each chunk
boundary. With a 60-frame chunk and 120-frame rollout, the default transition
window is frames [30, 90), centered on frame 60.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] == 0:
        raise ValueError("transition summary requires a non-empty 1D array")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _as_qpos(qpos: np.ndarray, *, name: str = "qpos") -> np.ndarray:
    array = np.asarray(qpos, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (B, T, D), got {tuple(array.shape)}")
    if array.shape[1] < 4:
        raise ValueError(f"{name} transition metrics require at least 4 frames")
    return array


def _as_body_pos(
    body_pos: np.ndarray,
    *,
    batch_size: int | None = None,
    num_frames: int | None = None,
    name: str = "body_pos",
) -> np.ndarray:
    array = np.asarray(body_pos, dtype=np.float64)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (B, T, J, 3), got {tuple(array.shape)}")
    if array.shape[1] < 4:
        raise ValueError(f"{name} transition metrics require at least 4 frames")
    if batch_size is not None and array.shape[0] != batch_size:
        raise ValueError(f"{name} batch size must be {batch_size}, got {array.shape[0]}")
    if num_frames is not None and array.shape[1] != num_frames:
        raise ValueError(f"{name} num frames must be {num_frames}, got {array.shape[1]}")
    return array


def _boundary_frames(num_frames: int, chunk_length: int) -> np.ndarray:
    chunk_length = int(chunk_length)
    if chunk_length <= 3:
        raise ValueError(f"chunk_length must be greater than 3 for jerk metrics, got {chunk_length}")
    return np.arange(chunk_length, int(num_frames), chunk_length, dtype=np.int64)


def _transition_windows(num_frames: int, chunk_length: int, window_length: int | None) -> list[tuple[int, int]]:
    boundaries = _boundary_frames(num_frames, chunk_length)
    if boundaries.shape[0] == 0:
        raise ValueError(
            f"No transition boundary in {num_frames} frames for chunk_length={chunk_length}; "
            "generate more than one chunk to run transition metrics."
        )
    length = int(window_length or chunk_length)
    if length < 4:
        raise ValueError(f"transition_window_length must be at least 4 for jerk metrics, got {length}")
    windows: list[tuple[int, int]] = []
    left = length // 2
    for boundary in boundaries:
        start = int(boundary) - left
        end = start + length
        if start < 0 or end > int(num_frames):
            raise ValueError(
                f"Transition window [{start}, {end}) around boundary {int(boundary)} does not fit "
                f"inside {num_frames} frames; reduce transition_window_length."
            )
        windows.append((start, end))
    return windows


def _third_difference(signal: np.ndarray) -> np.ndarray:
    return signal[:, 3:] - 3.0 * signal[:, 2:-1] + 3.0 * signal[:, 1:-2] - signal[:, :-3]


def _body_jerk_l1(body_pos: np.ndarray) -> np.ndarray:
    jerk = _third_difference(body_pos)
    return np.abs(jerk).sum(axis=-1)


def _qpos_jerk_abs(qpos: np.ndarray) -> np.ndarray:
    return np.abs(_third_difference(qpos))


def _windowed_jerk(jerk: np.ndarray, windows: list[tuple[int, int]]) -> np.ndarray:
    chunks = []
    for start, end in windows:
        chunks.append(jerk[:, start : end - 3])
    return np.concatenate(chunks, axis=1)


def _reference_level(jerk: np.ndarray) -> float:
    if jerk.size == 0:
        raise ValueError("reference jerk level requires a non-empty jerk array")
    return float(jerk.max(axis=1).mean())


def _body_pj_auj(
    *,
    body_pos: np.ndarray,
    reference_body_pos: np.ndarray,
    windows: list[tuple[int, int]],
) -> dict[str, np.ndarray | float]:
    generated_jerk = _windowed_jerk(_body_jerk_l1(body_pos), windows)
    reference_jerk_level = _reference_level(_body_jerk_l1(reference_body_pos))
    pj = generated_jerk.max(axis=(1, 2))
    auj = np.abs(generated_jerk - reference_jerk_level).max(axis=2).sum(axis=1)
    return {
        "pj": pj,
        "auj": auj,
        "body_pj": pj,
        "body_auj": auj,
        "reference_body_jerk_level": reference_jerk_level,
    }


def _qpos_pj_auj(
    *,
    qpos: np.ndarray,
    reference_qpos: np.ndarray,
    windows: list[tuple[int, int]],
) -> dict[str, np.ndarray | float]:
    generated_jerk = _windowed_jerk(_qpos_jerk_abs(qpos), windows)
    reference_jerk_level = _reference_level(_qpos_jerk_abs(reference_qpos))
    qpos_pj = generated_jerk.max(axis=(1, 2))
    qpos_auj = np.abs(generated_jerk - reference_jerk_level).max(axis=2).sum(axis=1)
    return {
        "qpos_pj": qpos_pj,
        "qpos_auj": qpos_auj,
        "reference_qpos_jerk_level": reference_jerk_level,
    }


def transition_metric_values(
    *,
    qpos: np.ndarray,
    chunk_length: int,
    reference_qpos: np.ndarray | None = None,
    body_pos: np.ndarray | None = None,
    reference_body_pos: np.ndarray | None = None,
    transition_window_length: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Return per-sample PJ/AUJ transition values.

    PJ/AUJ are computed on body positions when body_pos/reference_body_pos are
    provided; this is the metric reported in benchmark summaries. qpos PJ/AUJ
    are additionally emitted when reference_qpos is available for diagnostics.
    """
    qpos_array = _as_qpos(qpos)
    batch_size, num_frames = int(qpos_array.shape[0]), int(qpos_array.shape[1])
    windows = _transition_windows(num_frames, int(chunk_length), transition_window_length)

    values: dict[str, np.ndarray | float] = {}
    if reference_qpos is not None:
        reference_qpos_array = _as_qpos(reference_qpos, name="reference_qpos")
        values.update(_qpos_pj_auj(qpos=qpos_array, reference_qpos=reference_qpos_array, windows=windows))

    if body_pos is not None or reference_body_pos is not None:
        if body_pos is None or reference_body_pos is None:
            raise ValueError("body PJ/AUJ require both body_pos and reference_body_pos")
        body_array = _as_body_pos(body_pos, batch_size=batch_size, num_frames=num_frames)
        reference_body_array = _as_body_pos(reference_body_pos, name="reference_body_pos")
        values.update(_body_pj_auj(body_pos=body_array, reference_body_pos=reference_body_array, windows=windows))

    if not values and reference_qpos is None:
        raise ValueError("transition PJ/AUJ require reference_qpos or reference_body_pos to compute AUJ")
    return values


def transition_metric_summary(
    values: dict[str, np.ndarray | float],
    *,
    chunk_length: int,
    num_frames: int,
    transition_window_length: int | None = None,
    indices: np.ndarray | None = None,
) -> dict[str, Any]:
    windows = _transition_windows(int(num_frames), int(chunk_length), transition_window_length)
    summary: dict[str, Any] = {}
    num_samples: int | None = None
    for key, value in values.items():
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            summary[key] = float(array)
            continue
        selected = array if indices is None else array[indices]
        summary[key] = _summary_stats(selected)
        num_samples = int(selected.shape[0])
    if num_samples is None:
        raise ValueError("transition summary requires at least one per-sample metric")
    summary.update(
        {
            "num_samples": num_samples,
            "num_frames": int(num_frames),
            "chunk_length": int(chunk_length),
            "transition_window_length": int(transition_window_length or chunk_length),
            "boundary_frames": _boundary_frames(int(num_frames), int(chunk_length)).astype(int).tolist(),
            "transition_windows": [[int(start), int(end)] for start, end in windows],
            "definition": "PJ=max L1 jerk over links/time; AUJ=sum_t max_link |jerk(t)-reference_jerk_level| over transition windows.",
        }
    )
    return summary
