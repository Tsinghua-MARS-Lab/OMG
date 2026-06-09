"""Distribution metrics shared by generation benchmarks.

Implements:
- motion_fid: Frechet distance between real and generated motion embeddings/features.
- motion_fvd: alias of motion_fid for video-style feature sequences.
- motion_kid: polynomial-kernel MMD/KID between real and generated embeddings/features.
- diversity: average pairwise distance within generated embeddings/features.
- multimodality: average same-condition pairwise distance across repeated generations.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def diversity(motion_embeddings: np.ndarray, num_pairs: int = 300, seed: int = 1234) -> float:
    emb = np.asarray(motion_embeddings, dtype=np.float64)
    if emb.shape[0] < 2:
        raise ValueError("At least two embeddings are required")
    rng = np.random.default_rng(seed)
    first = rng.integers(0, emb.shape[0], size=num_pairs)
    second = rng.integers(0, emb.shape[0] - 1, size=num_pairs)
    second = second + (second >= first)
    return float(np.linalg.norm(emb[first] - emb[second], axis=1).mean())


def multimodality(repeated_embeddings: np.ndarray, num_pairs: int = 10, seed: int = 1234) -> dict[str, float | int]:
    embeddings = np.asarray(repeated_embeddings, dtype=np.float64)
    if embeddings.ndim != 3:
        raise ValueError("multimodality embeddings must have shape [num_conditions, repeats, dim]")
    num_conditions, repeats, _ = embeddings.shape
    if repeats < 2:
        raise ValueError("multimodality requires at least two generated samples per condition")
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    rng = np.random.default_rng(seed)
    first = rng.integers(0, repeats, size=num_pairs)
    second = rng.integers(0, repeats - 1, size=num_pairs)
    second = second + (second >= first)
    distances = np.linalg.norm(embeddings[:, first] - embeddings[:, second], axis=-1)
    per_condition = distances.mean(axis=1)
    return {
        "mean": float(per_condition.mean()),
        "std": float(per_condition.std(ddof=0)),
        "min": float(per_condition.min()),
        "max": float(per_condition.max()),
        "num_texts": int(num_conditions),
        "num_conditions": int(num_conditions),
        "repeats": int(repeats),
        "num_pairs": int(num_pairs),
        "seed": int(seed),
    }


def _stats(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=0), np.cov(x, rowvar=False)


def motion_fid(real_embeddings: np.ndarray, generated_embeddings: np.ndarray, eps: float = 1e-6) -> float:
    mu1, sigma1 = _stats(real_embeddings)
    mu2, sigma2 = _stats(generated_embeddings)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(max(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean), 0.0))


def _kernel(x: np.ndarray, y: np.ndarray, degree: int = 3) -> np.ndarray:
    gamma = 1.0 / x.shape[1]
    return (gamma * x.dot(y.T) + 1.0) ** degree


def motion_kid(real_embeddings: np.ndarray, generated_embeddings: np.ndarray) -> dict[str, float]:
    real = np.asarray(real_embeddings, dtype=np.float64)
    gen = np.asarray(generated_embeddings, dtype=np.float64)
    k_rr = _kernel(real, real)
    k_gg = _kernel(gen, gen)
    k_rg = _kernel(real, gen)
    m, n = real.shape[0], gen.shape[0]
    value = ((k_rr.sum() - np.trace(k_rr)) / (m * (m - 1))) + ((k_gg.sum() - np.trace(k_gg)) / (n * (n - 1))) - 2.0 * k_rg.mean()
    return {"mean": float(value), "std": 0.0}


motion_fvd = motion_fid
