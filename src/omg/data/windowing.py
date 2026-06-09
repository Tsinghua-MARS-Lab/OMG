from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from typing import Any


TRAIN_WINDOW_RANDOM = "random"
TRAIN_WINDOW_EXHAUSTIVE = "exhaustive"


def normalize_train_window_policy(policy: str) -> str:
    normalized = str(policy).strip().lower().replace("-", "_")
    if normalized in {"random", "rand", "sample", "random_sample"}:
        return TRAIN_WINDOW_RANDOM
    if normalized in {"exhaustive", "sliding", "sliding_window", "stride", "all"}:
        return TRAIN_WINDOW_EXHAUSTIVE
    raise ValueError(
        f"Unsupported train_window_policy={policy!r}; expected 'random' or 'exhaustive'"
    )


def is_exhaustive_train_window_policy(policy: str) -> bool:
    return normalize_train_window_policy(policy) == TRAIN_WINDOW_EXHAUSTIVE


class ExhaustiveWindowSampleView(Sequence[dict[str, Any]]):
    """Compact sequence of concrete fixed windows over segment-level sample metadata."""

    def __init__(self, samples: Sequence[dict[str, Any]], *, window_size: int, stride: int) -> None:
        self.samples = samples
        self.window_size = int(window_size)
        self.stride = int(stride)
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self._counts: list[int] = []
        self._offsets: list[int] = [0]
        for sample in self.samples:
            count = self._count_windows(sample)
            self._counts.append(count)
            self._offsets.append(self._offsets[-1] + count)

    def _segment_bounds(self, sample: dict[str, Any]) -> tuple[int, int]:
        start = int(sample.get("segment_frame_start", 0))
        end = int(sample.get("segment_frame_end", start + 1))
        return start, max(start + 1, end)

    def _count_windows(self, sample: dict[str, Any]) -> int:
        start, end = self._segment_bounds(sample)
        max_start = end - self.window_size
        if max_start <= start:
            return 1
        return ((max_start - start) // self.stride) + 1

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sample_index = bisect_right(self._offsets, index) - 1
        local_index = index - self._offsets[sample_index]
        sample = dict(self.samples[sample_index])
        start, end = self._segment_bounds(sample)
        max_start = end - self.window_size
        if max_start <= start:
            window_start = start
        else:
            window_start = start + local_index * self.stride
            if window_start > max_start:
                raise IndexError(index)
        sample["fixed_window_start"] = int(window_start)
        sample["eval_window_index"] = int(local_index)
        sample["eval_num_windows"] = int(self._counts[sample_index])
        return sample
