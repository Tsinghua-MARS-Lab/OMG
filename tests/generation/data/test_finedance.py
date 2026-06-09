from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest


DEFAULT_OMG_DATA_ROOT = Path(os.environ.get("OMG_DATA_ROOT", "data/OMG-Data"))
DEFAULT_FINEDANCE_MOTION_ROOT = DEFAULT_OMG_DATA_ROOT / "datasets" / "finedance" / "g1"
DEFAULT_FINEDANCE_MUSIC_ROOT = DEFAULT_OMG_DATA_ROOT / "datasets" / "finedance" / "music_npy"


def _natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.as_posix())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _data_root(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default)))


def _sorted_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        pytest.skip(f"{root} does not exist; set the matching FINEDANCE_*_ROOT env var")
    return sorted(root.rglob(pattern), key=_natural_key)


def _motion_num_frames(path: Path) -> int:
    with np.load(path, mmap_mode="r") as data:
        if "qpos" not in data:
            raise AssertionError(f"{path} does not contain qpos")
        return int(data["qpos"].shape[0])


def _music_num_frames(path: Path) -> int:
    music = np.load(path, mmap_mode="r")
    return int(music.shape[0])


def test_finedance_motion_and_music_counts_match_by_order():
    motion_root = _data_root("FINEDANCE_MOTION_ROOT", DEFAULT_FINEDANCE_MOTION_ROOT)
    music_root = _data_root("FINEDANCE_MUSIC_ROOT", DEFAULT_FINEDANCE_MUSIC_ROOT)

    motion_files = _sorted_files(motion_root, "*.npz")
    music_files = _sorted_files(music_root, "*.npy")

    assert motion_files, f"No motion .npz files found under {motion_root}"
    assert music_files, f"No music .npy files found under {music_root}"
    assert len(motion_files) == len(music_files), (
        "FineDance motion/music file counts differ when compared by sorted order: "
        f"motion={len(motion_files)} root={motion_root}, "
        f"music={len(music_files)} root={music_root}"
    )


def test_finedance_motion_and_music_frame_counts_match_by_order():
    motion_root = _data_root("FINEDANCE_MOTION_ROOT", DEFAULT_FINEDANCE_MOTION_ROOT)
    music_root = _data_root("FINEDANCE_MUSIC_ROOT", DEFAULT_FINEDANCE_MUSIC_ROOT)

    motion_files = _sorted_files(motion_root, "*.npz")
    music_files = _sorted_files(music_root, "*.npy")

    assert len(motion_files) == len(music_files), (
        "Cannot compare frame counts by order because file counts differ: "
        f"motion={len(motion_files)}, music={len(music_files)}"
    )

    mismatches = []
    for index, (motion_path, music_path) in enumerate(zip(motion_files, music_files), start=1):
        motion_frames = _motion_num_frames(motion_path)
        music_frames = _music_num_frames(music_path)
        if motion_frames != music_frames:
            mismatches.append(
                (
                    index,
                    motion_path.relative_to(motion_root).as_posix(),
                    motion_frames,
                    music_path.relative_to(music_root).as_posix(),
                    music_frames,
                )
            )

    assert not mismatches, _format_mismatches(mismatches)


def _format_mismatches(mismatches: list[tuple[int, str, int, str, int]]) -> str:
    preview = "\n".join(
        f"#{index}: motion={motion_name} frames={motion_frames}, "
        f"music={music_name} frames={music_frames}"
        for index, motion_name, motion_frames, music_name, music_frames in mismatches[:20]
    )
    suffix = "" if len(mismatches) <= 20 else f"\n... and {len(mismatches) - 20} more mismatches"
    return (
        "FineDance motion/music frame counts differ when paired by sorted order "
        f"({len(mismatches)} mismatches):\n{preview}{suffix}"
    )
