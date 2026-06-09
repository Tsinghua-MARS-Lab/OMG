from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from omg.benchmarks.evaluator.text_encoder import TextEncoder


def _safe_cache_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def text_cache_root(text_encoder: TextEncoder, cache_root: str | Path) -> Path:
    return Path(cache_root) / _safe_cache_name(text_encoder.model_name) / f"maxlen_{text_encoder.max_length}"


def text_cache_path(text_encoder: TextEncoder, cache_root: str | Path, text: str) -> Path:
    key = f"{text_encoder.model_name}\n{text_encoder.max_length}\n{text}".encode("utf-8")
    return text_cache_root(text_encoder, cache_root) / f"{hashlib.sha256(key).hexdigest()}.pt"


def load_text_cache(path: Path, expected_dim: int) -> torch.Tensor | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"[INFO] Text cache is unreadable, rebuilding: {path} ({exc})")
            return None
    embedding = payload["embedding"] if isinstance(payload, dict) else payload
    if not isinstance(embedding, torch.Tensor):
        print(f"[INFO] Text cache has no tensor embedding, rebuilding: {path}")
        return None
    if embedding.ndim != 1 or embedding.shape[0] != expected_dim:
        print(
            f"[INFO] Text cache shape mismatch, rebuilding: {path} "
            f"(got {tuple(embedding.shape)}, expected ({expected_dim},))"
        )
        return None
    return embedding.float()


def save_text_cache(path: Path, embedding: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(embedding.detach().cpu().float(), tmp_path)
    tmp_path.replace(path)


def encode_texts_with_cache(
    text_encoder: TextEncoder,
    captions: list[str],
    *,
    device: torch.device,
    cache_root: str | Path,
) -> torch.Tensor:
    cache_dir = text_cache_root(text_encoder, cache_root)
    if not hasattr(text_encoder, "_logged_external_cache_roots"):
        text_encoder._logged_external_cache_roots = set()
    if str(cache_dir) not in text_encoder._logged_external_cache_roots:
        print(f"[INFO] Text feature cache root: {cache_dir}")
        text_encoder._logged_external_cache_roots.add(str(cache_dir))

    raw_embeddings: list[torch.Tensor | None] = []
    missing_texts: list[str] = []
    missing_indices: list[int] = []
    for idx, caption in enumerate(captions):
        text = str(caption)
        path = text_cache_path(text_encoder, cache_root, text)
        embedding = load_text_cache(path, text_encoder.t5_dim) if path.exists() else None
        if embedding is None:
            raw_embeddings.append(None)
            missing_texts.append(text)
            missing_indices.append(idx)
        else:
            raw_embeddings.append(embedding)

    if missing_texts:
        print(f"[INFO] Text cache miss: encoding {len(missing_texts)} texts with {text_encoder.model_name}")
        encoded = text_encoder.raw_encode(missing_texts, device=device).detach().cpu().float()
        for text, idx, embedding in zip(missing_texts, missing_indices, encoded):
            save_text_cache(text_cache_path(text_encoder, cache_root, text), embedding)
            raw_embeddings[idx] = embedding

    raw = torch.stack([embedding for embedding in raw_embeddings if embedding is not None], dim=0).to(device)
    return text_encoder.encode_raw(raw)
