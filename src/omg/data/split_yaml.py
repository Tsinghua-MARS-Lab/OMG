"""Helpers for reading dataset ``info.yaml`` split sections.

Split sections are typically a mapping from **relative clip id** (often a POSIX path
without ``.npz``, e.g. ``folder4/279303_retarget``) to a numeric weight. Legacy configs
may nest folders as nested dicts; :func:`flatten_dataset_split_paths` normalizes both
layouts to a sorted list of string keys suitable for joining against ``g1/``.
"""

from __future__ import annotations

from pathlib import Path


def flatten_dataset_split_paths(split_data: object) -> list[str]:
    """Return sorted clip ids for a split value from ``info.yaml``.

    Supports:

    * ``dict`` — keys are clip ids; values may be weights (``int`` / ``float``) **or**
      nested ``dict`` objects (legacy folder nesting).
    * ``list`` — each element is a clip id string.
    """
    if split_data is None:
        return []
    if isinstance(split_data, list):
        return sorted(str(x) for x in split_data)
    if not isinstance(split_data, dict):
        raise TypeError(f"split data must be dict or list, got {type(split_data)}")

    def walk(prefix: tuple[str, ...], node: dict) -> list[str]:
        out: list[str] = []
        for raw_key, val in node.items():
            key = str(raw_key)
            if isinstance(val, dict):
                out.extend(walk(prefix + (key,), val))
            else:
                rel = Path(*prefix, key) if prefix else Path(key)
                out.append(rel.as_posix())
        return out

    return sorted(walk((), split_data))


def dataset_motion_npz_path(dataset_root: str | Path, entry: str) -> Path:
    """Resolve ``dataset_root / <entry>.npz`` using the same relative-path rules as loaders."""
    root = Path(dataset_root)
    rel = Path(str(entry))
    if rel.suffix != ".npz":
        rel = rel.with_suffix(".npz")
    return root / rel


def _strip_retarget_suffix(stem: str) -> str:
    return stem[: -len("_retarget")] if stem.endswith("_retarget") else stem


def resolve_dataset_caption_txt_path(entry: str, texts_root: str | Path) -> Path | None:
    """Locate a parallel ``.txt`` caption next to motion layout under ``texts_root``.

    Entry keys follow ``info.yaml`` (no ``.npz`` suffix): e.g. ``folder4/279303_retarget``
    maps to ``texts_root/folder4/279303.txt``.
    """
    texts_root = Path(texts_root)
    entry_path = Path(str(entry))
    rel_npz = entry_path if entry_path.suffix == ".npz" else entry_path.with_suffix(".npz")
    seq_stem = rel_npz.stem
    text_stem = _strip_retarget_suffix(seq_stem)
    rel_parent = rel_npz.parent

    candidates: list[Path] = [
        texts_root / rel_parent / f"{text_stem}.txt",
        texts_root / rel_npz.with_suffix(".txt"),
        texts_root / f"{text_stem}.txt",
    ]
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand
    return None
