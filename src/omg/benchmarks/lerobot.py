from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from torch.utils.data import Dataset

from omg.data.lerobot_dataset import LeRobotG1MotionDataset


BENCHMARK_SAMPLE_SCHEMA = "omg.benchmark.sample.v2"


class LeRobotBenchmarkView(Dataset):
    """A source-dataset view over one shared, pinned LeRobot dataset instance."""

    def __init__(self, dataset: LeRobotG1MotionDataset, name: str, indices: Iterable[int]) -> None:
        if dataset.revision is None:
            raise ValueError("LeRobot benchmarks require a pinned dataset revision")
        if dataset.training:
            raise ValueError("LeRobot benchmark identities require a deterministic val/test split")
        self.dataset = dataset
        self.name = str(name)
        self.indices = tuple(int(index) for index in indices)
        if not self.indices:
            raise ValueError(f"LeRobot benchmark view {self.name!r} is empty")
        self.samples = [dataset.samples[index] for index in self.indices]
        self.use_text = dataset.use_text
        self.use_audio = dataset.use_audio
        self.use_human_motion = dataset.use_human_motion
        self.window_size = dataset.window_size
        self.default_fps = dataset.default_fps
        self.repo_id = dataset.repo_id
        self.revision = dataset.revision
        self.split = dataset.split
        self._identity_to_index: dict[tuple[int, int, int], int] = {}
        for local_index, base_index in enumerate(self.indices):
            locator = dataset.sample_locator(base_index)
            key = (
                int(locator["episode_index"]),
                int(locator["window_start"]),
                int(locator["num_frames"]),
            )
            if key in self._identity_to_index:
                raise ValueError(f"Duplicate LeRobot benchmark identity in {self.name}: {key}")
            self._identity_to_index[key] = local_index

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[int(index)]]

    def sample_identity(self, index: int) -> dict[str, Any]:
        locator = self.dataset.sample_locator(self.indices[int(index)])
        if locator["source_dataset"] != self.name:
            raise ValueError(
                f"View/source mismatch: view={self.name!r} sample={locator['source_dataset']!r}"
            )
        return {
            "schema": BENCHMARK_SAMPLE_SCHEMA,
            "repo_id": locator["repo_id"],
            "revision": locator["revision"],
            "split": locator["split"],
            "episode_index": locator["episode_index"],
            "window_start": locator["window_start"],
            "num_frames": locator["num_frames"],
            "source_dataset": locator["source_dataset"],
            "source_id": locator["source_id"],
            "segment_index": locator["segment_index"],
            "source_start_frame": locator["source_start_frame"],
            "source_end_frame": locator["source_end_frame"],
        }

    def resolve_identity(self, identity: dict[str, Any]) -> int:
        expected = {
            "schema": BENCHMARK_SAMPLE_SCHEMA,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "split": self.split,
            "source_dataset": self.name,
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ValueError(
                    f"LeRobot benchmark identity mismatch for {key}: "
                    f"manifest={identity.get(key)!r} dataset={value!r}"
                )
        lookup = (
            int(identity["episode_index"]),
            int(identity["window_start"]),
            int(identity["num_frames"]),
        )
        if lookup not in self._identity_to_index:
            raise KeyError(f"LeRobot benchmark sample not found in {self.name}: {lookup}")
        index = self._identity_to_index[lookup]
        resolved = self.sample_identity(index)
        for key in ("source_id", "segment_index", "source_start_frame", "source_end_frame"):
            if key in identity and identity[key] != resolved[key]:
                raise ValueError(
                    f"LeRobot benchmark identity mismatch for {key}: "
                    f"manifest={identity[key]!r} dataset={resolved[key]!r}"
                )
        return index

    def sample_has_condition(self, index: int, condition: str, *, num_frames: int) -> bool:
        return self.dataset.sample_has_condition(
            self.indices[int(index)],
            condition,
            num_frames=int(num_frames),
        )

    def global_index(self, index: int) -> int:
        return self.indices[int(index)]


def build_lerobot_benchmark_views(
    dataset: LeRobotG1MotionDataset,
    *,
    include: list[str] | None = None,
) -> dict[str, LeRobotBenchmarkView]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        source_dataset = str(sample.get("source_dataset", "")).strip()
        if not source_dataset:
            raise ValueError(f"LeRobot episode {sample.get('episode_index')} has no omg/dataset identity")
        grouped[source_dataset].append(index)
    views: dict[str, LeRobotBenchmarkView] = {}
    requested = None if include is None else {str(name) for name in include}
    if requested is not None:
        missing = sorted(requested.difference(grouped))
        if missing:
            raise KeyError(f"Unknown LeRobot source datasets: {missing}")
    for name in sorted(grouped):
        if requested is not None and name not in requested:
            continue
        views[name] = LeRobotBenchmarkView(dataset, name, grouped[name])
    if not views:
        raise ValueError(f"No LeRobot benchmark source views selected; include={include}")
    return views
