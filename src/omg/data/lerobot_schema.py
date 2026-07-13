from __future__ import annotations

from typing import Any

LEROBOT_DATASET_VERSION = "v3.0"
LEROBOT_REPO_ID = "THU-MARS/OMG-Data"
LEROBOT_FPS = 30
LEROBOT_CHUNKS_SIZE = 1000
LEROBOT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
LEROBOT_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
LEROBOT_TASKS_PATH = "meta/tasks.parquet"

G1_QPOS_NAMES = (
    "root_x",
    "root_y",
    "root_z",
    "root_qw",
    "root_qx",
    "root_qy",
    "root_qz",
    *(f"joint_{index:02d}" for index in range(29)),
)

DEFAULT_FRAME_FEATURES: dict[str, dict[str, Any]] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def frame_features(*, include_audio: bool, audio_dim: int, include_humanref: bool, humanref_dim: int) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": [36], "names": list(G1_QPOS_NAMES)},
        "action": {"dtype": "float32", "shape": [36], "names": list(G1_QPOS_NAMES)},
        "omg.condition.has_text": {"dtype": "bool", "shape": [1], "names": None},
    }
    if include_audio:
        features.update(
            {
                "omg.audio.feature": {
                    "dtype": "float32",
                    "shape": [int(audio_dim)],
                    "names": [f"audio_{index:02d}" for index in range(int(audio_dim))],
                },
                "omg.condition.has_audio": {"dtype": "bool", "shape": [1], "names": None},
            }
        )
    if include_humanref:
        features.update(
            {
                "omg.humanref.motion": {
                    "dtype": "float32",
                    "shape": [int(humanref_dim)],
                    "names": [f"humanref_{index:02d}" for index in range(int(humanref_dim))],
                },
                "omg.condition.has_humanref": {"dtype": "bool", "shape": [1], "names": None},
            }
        )
    return {**features, **DEFAULT_FRAME_FEATURES}
