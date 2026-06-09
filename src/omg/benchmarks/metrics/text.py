"""Text-to-motion retrieval metrics.

Implements:
- r_precision: retrieval hit rate at top-k for matched text/motion embeddings.
- matching_score: paired text/motion embedding distance.
"""

from __future__ import annotations

import numpy as np


def matching_score(motion_embeddings: np.ndarray, text_embeddings: np.ndarray, reduction: str = "mean"):
    distances = np.linalg.norm(np.asarray(motion_embeddings) - np.asarray(text_embeddings), axis=1)
    if reduction == "none":
        return distances
    if reduction == "sum":
        return float(distances.sum())
    if reduction == "mean":
        return float(distances.mean())
    raise ValueError(f"Unsupported reduction: {reduction}")


def r_precision(motion_embeddings: np.ndarray, text_embeddings: np.ndarray, top_k: int = 3) -> np.ndarray:
    motion = np.asarray(motion_embeddings, dtype=np.float64)
    text = np.asarray(text_embeddings, dtype=np.float64)
    distances = -2.0 * motion.dot(text.T) + np.square(motion).sum(axis=1, keepdims=True) + np.square(text).sum(axis=1)[None, :]
    nearest = np.argsort(distances, axis=1)[:, :top_k]
    hit = np.maximum.accumulate(nearest == np.arange(motion.shape[0])[:, None], axis=1)
    return hit.mean(axis=0)
