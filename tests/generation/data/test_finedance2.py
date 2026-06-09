from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest


DEFAULT_OMG_DATA_ROOT = Path(os.environ.get("OMG_DATA_ROOT", "data/OMG-Data"))
DEFAULT_FINEDANCE_MOTION_ROOT = DEFAULT_OMG_DATA_ROOT / "datasets" / "finedance" / "g1"
DEFAULT_FINEDANCE_MUSIC_ROOT = DEFAULT_OMG_DATA_ROOT / "datasets" / "finedance" / "music_npy"


def _data_root(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default)))


def _sorted_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        pytest.skip(f"{root} does not exist; set the matching FINEDANCE_*_ROOT env var")
    return sorted(root.rglob(pattern), key=lambda path: path.as_posix())


def _numeric_id(path: Path) -> str:
    match = re.search(r"\d+", path.stem)
    if match is None:
        return path.stem
    return match.group(0).lstrip("0") or "0"


def _index_by_numeric_id(paths: list[Path]) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        indexed[_numeric_id(path)].append(path)
    return dict(indexed)


def _motion_num_frames(path: Path) -> int:
    with np.load(path, mmap_mode="r") as data:
        if "qpos" not in data:
            raise AssertionError(f"{path} does not contain qpos")
        return int(data["qpos"].shape[0])


def _audio_num_frames(path: Path) -> int:
    audio = np.load(path, mmap_mode="r")
    return int(audio.shape[0])


def test_finedance_frame_counts_for_shared_numeric_ids():
    motion_root = _data_root("FINEDANCE_MOTION_ROOT", DEFAULT_FINEDANCE_MOTION_ROOT)
    music_root = _data_root("FINEDANCE_MUSIC_ROOT", DEFAULT_FINEDANCE_MUSIC_ROOT)

    motion_files = _sorted_files(motion_root, "*.npz")
    music_files = _sorted_files(music_root, "*.npy")

    motion_by_id = _index_by_numeric_id(motion_files)
    music_by_id = _index_by_numeric_id(music_files)

    motion_ids = set(motion_by_id)
    music_ids = set(music_by_id)
    shared_ids = sorted(motion_ids & music_ids, key=lambda value: int(value) if value.isdigit() else value)
    motion_only_ids = sorted(motion_ids - music_ids, key=lambda value: int(value) if value.isdigit() else value)
    music_only_ids = sorted(music_ids - motion_ids, key=lambda value: int(value) if value.isdigit() else value)

    duplicate_motion_ids = {key: value for key, value in motion_by_id.items() if len(value) > 1}
    duplicate_music_ids = {key: value for key, value in music_by_id.items() if len(value) > 1}

    pair_rows = []
    mismatches = []
    checked_pairs = 0
    for numeric_id in shared_ids:
        motion_candidates = motion_by_id[numeric_id]
        music_candidates = music_by_id[numeric_id]
        if len(motion_candidates) != 1 or len(music_candidates) != 1:
            pair_rows.append(
                (
                    numeric_id,
                    "DUPLICATE",
                    [path.relative_to(motion_root).as_posix() for path in motion_candidates],
                    None,
                    [path.relative_to(music_root).as_posix() for path in music_candidates],
                    None,
                )
            )
            continue
        motion_path = motion_candidates[0]
        music_path = music_candidates[0]
        motion_frames = _motion_num_frames(motion_path)
        music_frames = _audio_num_frames(music_path)
        checked_pairs += 1
        status = "MATCH" if motion_frames == music_frames else "MISMATCH"
        pair_rows.append(
            (
                numeric_id,
                status,
                motion_path.relative_to(motion_root).as_posix(),
                motion_frames,
                music_path.relative_to(music_root).as_posix(),
                music_frames,
            )
        )
        if motion_frames != music_frames:
            mismatches.append(
                (
                    numeric_id,
                    motion_path.relative_to(motion_root).as_posix(),
                    motion_frames,
                    music_path.relative_to(music_root).as_posix(),
                    music_frames,
                )
            )

    print(f"motion_root={motion_root}")
    print(f"music_root={music_root}")
    print(f"motion_files={len(motion_files)} music_files={len(music_files)}")
    print(f"motion_numeric_ids={len(motion_ids)} music_numeric_ids={len(music_ids)} shared_ids={len(shared_ids)}")
    print(f"checked_unique_pairs={checked_pairs}")
    print(f"motion_only_ids={motion_only_ids}")
    print(f"music_only_ids={music_only_ids}")
    print(f"duplicate_motion_ids={_format_duplicate_ids(duplicate_motion_ids, motion_root)}")
    print(f"duplicate_music_ids={_format_duplicate_ids(duplicate_music_ids, music_root)}")
    print("shared_pair_details:")
    for numeric_id, status, motion_name, motion_frames, music_name, music_frames in pair_rows:
        print(
            f"id={numeric_id} status={status} "
            f"motion={motion_name} motion_frames={motion_frames} "
            f"music={music_name} music_frames={music_frames}"
        )

    if mismatches:
        print(f"frame_mismatches={len(mismatches)}")
        for numeric_id, motion_name, motion_frames, music_name, music_frames in mismatches[:50]:
            print(
                f"id={numeric_id}: motion={motion_name} frames={motion_frames}, "
                f"music={music_name} frames={music_frames}"
            )
        if len(mismatches) > 50:
            print(f"... and {len(mismatches) - 50} more frame mismatches")
    else:
        print("frame_mismatches=0")


def _format_duplicate_ids(indexed: dict[str, list[Path]], root: Path) -> dict[str, list[str]]:
    return {
        numeric_id: [path.relative_to(root).as_posix() for path in paths]
        for numeric_id, paths in sorted(indexed.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0])
    }
