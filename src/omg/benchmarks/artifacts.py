"""Schema constants and verification helpers for generated motion benchmark artifacts.

The `.npz` schema mirrors what
`omg.benchmarks.runners.{text,audio}._run_single_benchmark`
writes for `generated_qpos.npz` / `reference_qpos.npz`, so the same downstream
evaluation code can ingest artifact-based benchmark paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


GENERATED_KEYS: frozenset[str] = frozenset(
    {"qpos_36", "fps", "captions", "dataset", "dataset_index"}
)
REFERENCE_KEYS: frozenset[str] = frozenset(
    {"qpos_36", "fps", "valid", "captions", "dataset", "dataset_index"}
)
QPOS_36_DIM: int = 36


class ArtifactError(ValueError):
    """Raised when an `.npz` artifact does not match the expected schema."""


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise ArtifactError(f"artifact does not exist: {path}")
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _batch_size(arrays: dict[str, np.ndarray], label: str) -> int:
    qpos = arrays["qpos_36"]
    if qpos.ndim != 3:
        raise ArtifactError(
            f"[{label}] qpos_36 must have shape (N, T, {QPOS_36_DIM}); got {qpos.shape}"
        )
    if qpos.shape[-1] != QPOS_36_DIM:
        raise ArtifactError(
            f"[{label}] qpos_36 last dim must be {QPOS_36_DIM}; got {qpos.shape[-1]}"
        )
    if qpos.dtype != np.float32:
        raise ArtifactError(
            f"[{label}] qpos_36 dtype must be float32; got {qpos.dtype}"
        )
    return int(qpos.shape[0])


def verify_schema(
    arrays: dict[str, Any],
    required_keys: set[str] | frozenset[str],
    *,
    label: str,
) -> None:
    """Validate `arrays` against the expected key set.

    Raises:
        ArtifactError: if a required key is missing, dtypes / shapes are
            inconsistent, or fps contains non-positive entries.
    """

    missing = set(required_keys) - set(arrays.keys())
    if missing:
        raise ArtifactError(
            f"[{label}] missing required keys: {sorted(missing)}"
        )

    batch_size = _batch_size(arrays, label)

    fps = np.asarray(arrays["fps"])
    if fps.dtype.kind not in {"f", "i", "u"}:
        raise ArtifactError(
            f"[{label}] fps dtype must be numeric; got {fps.dtype}"
        )
    if fps.shape != (batch_size,):
        raise ArtifactError(
            f"[{label}] fps must have shape ({batch_size},); got {fps.shape}"
        )
    if not np.all(np.asarray(fps, dtype=np.float64) > 0.0):
        raise ArtifactError(
            f"[{label}] fps must be strictly positive; got min={float(fps.min())}"
        )

    captions = np.asarray(arrays["captions"])
    if captions.shape != (batch_size,):
        raise ArtifactError(
            f"[{label}] captions shape must be ({batch_size},); got {captions.shape}"
        )
    if captions.dtype.kind not in {"U", "S", "O"}:
        raise ArtifactError(
            f"[{label}] captions dtype must be string-like; got {captions.dtype}"
        )

    dataset = np.asarray(arrays["dataset"])
    if dataset.shape != (batch_size,):
        raise ArtifactError(
            f"[{label}] dataset shape must be ({batch_size},); got {dataset.shape}"
        )
    if dataset.dtype.kind not in {"U", "S", "O"}:
        raise ArtifactError(
            f"[{label}] dataset dtype must be string-like; got {dataset.dtype}"
        )

    dataset_index = np.asarray(arrays["dataset_index"])
    if dataset_index.shape != (batch_size,):
        raise ArtifactError(
            f"[{label}] dataset_index shape must be ({batch_size},); got {dataset_index.shape}"
        )
    if dataset_index.dtype.kind not in {"i", "u"}:
        raise ArtifactError(
            f"[{label}] dataset_index must be integer dtype; got {dataset_index.dtype}"
        )

    if "valid" in required_keys:
        valid = np.asarray(arrays["valid"])
        qpos_shape = arrays["qpos_36"].shape
        if valid.shape != qpos_shape[:2]:
            raise ArtifactError(
                f"[{label}] valid shape must be {qpos_shape[:2]}; got {valid.shape}"
            )
        if valid.dtype != np.bool_:
            raise ArtifactError(
                f"[{label}] valid dtype must be bool; got {valid.dtype}"
            )


def verify_generated_qpos(path: str | Path) -> dict[str, np.ndarray]:
    """Load + validate a `generated_qpos.npz` artifact."""

    arrays = _load_npz(Path(path))
    verify_schema(arrays, GENERATED_KEYS, label="generated_qpos")
    return arrays


def verify_reference_qpos(path: str | Path) -> dict[str, np.ndarray]:
    """Load + validate a `reference_qpos.npz` artifact."""

    arrays = _load_npz(Path(path))
    verify_schema(arrays, REFERENCE_KEYS, label="reference_qpos")
    return arrays
