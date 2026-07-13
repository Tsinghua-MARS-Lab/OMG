from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omg.data.lerobot_schema import LEROBOT_DATASET_VERSION, LEROBOT_TASKS_PATH

np = None
pq = None


def _ensure_runtime_imports() -> None:
    global np, pq
    if np is not None:
        return
    import numpy as _np
    import pyarrow.parquet as _pq

    np = _np
    pq = _pq


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"min": int(array.min()), "mean": float(array.mean()), "max": int(array.max())}


def _data_file_stats(root: Path) -> dict[str, Any]:
    files = sorted((root / "data").rglob("*.parquet"))
    rows = 0
    schemas = set()
    for path in files:
        metadata = pq.read_metadata(path)
        rows += int(metadata.num_rows)
        schemas.add(tuple(metadata.schema.names))
    return {
        "files": len(files),
        "rows": rows,
        "bytes": int(sum(path.stat().st_size for path in files)),
        "schema_count": len(schemas),
    }


def inspect_lerobot(root: str | Path) -> dict[str, Any]:
    _ensure_runtime_imports()
    root = Path(root)
    info_path = root / "meta" / "info.json"
    tasks_path = root / LEROBOT_TASKS_PATH
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot info: {info_path}")
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot tasks: {tasks_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'meta' / 'episodes'}")
    rows: list[dict[str, Any]] = []
    for path in episode_files:
        rows.extend(pq.read_table(path).to_pylist())
    data = _data_file_stats(root)

    errors: list[str] = []
    required_episode_columns = {
        "episode_index",
        "tasks",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
    }
    if rows:
        missing = required_episode_columns - set(rows[0])
        if missing:
            errors.append(f"missing episode columns: {sorted(missing)}")
    for expected_index, row in enumerate(rows):
        if int(row["episode_index"]) != expected_index:
            errors.append(f"episode index discontinuity at row {expected_index}")
            break
        expected_start = 0 if expected_index == 0 else int(rows[expected_index - 1]["dataset_to_index"])
        if int(row["dataset_from_index"]) != expected_start:
            errors.append(f"frame range discontinuity at episode {expected_index}")
            break
        if int(row["dataset_to_index"]) - int(row["dataset_from_index"]) != int(row["length"]):
            errors.append(f"episode length mismatch at episode {expected_index}")
            break
    total_frames = int(sum(int(row.get("length", 0)) for row in rows))
    if info.get("codebase_version") != LEROBOT_DATASET_VERSION:
        errors.append(f"codebase_version={info.get('codebase_version')!r}")
    if int(info.get("total_episodes", -1)) != len(rows):
        errors.append("info total_episodes mismatch")
    if int(info.get("total_frames", -1)) != total_frames or data["rows"] != total_frames:
        errors.append("frame count mismatch between info, episodes, and data files")
    if data["schema_count"] != 1:
        errors.append(f"data files contain {data['schema_count']} schemas")

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        dataset = str(row.get("omg/dataset", ""))
        split = str(row.get("omg/split", ""))
        key = (split, dataset)
        entry = by_key.setdefault(
            key,
            {
                "split": split,
                "dataset": dataset,
                "episodes": 0,
                "frames": 0,
                "empty_captions": 0,
                "has_audio_episodes": 0,
                "has_human_motion_episodes": 0,
                "_lengths": [],
            },
        )
        frames = int(row.get("length", 0))
        tasks = row.get("tasks") or []
        caption = str(tasks[0]).strip() if tasks else ""
        entry["episodes"] += 1
        entry["frames"] += frames
        entry["_lengths"].append(frames)
        entry["empty_captions"] += int(caption == "")
        entry["has_audio_episodes"] += int(bool(row.get("omg/has_audio", False)))
        entry["has_human_motion_episodes"] += int(bool(row.get("omg/has_humanref", False)))
    dataset_summaries = []
    for _, entry in sorted(by_key.items()):
        entry["length_frames"] = _summary(entry.pop("_lengths"))
        dataset_summaries.append(entry)
    total_empty = int(sum(item["empty_captions"] for item in dataset_summaries))
    manifest_path = root / "meta" / "omg_manifest.json"
    return {
        "root": str(root),
        "format": info.get("codebase_version"),
        "official_lerobot_v3": not errors,
        "errors": errors,
        "episodes": len(rows),
        "frames": total_frames,
        "empty_captions": total_empty,
        "episode_files": len(episode_files),
        "data": data,
        "info": info,
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None,
        "datasets": dataset_summaries,
    }
