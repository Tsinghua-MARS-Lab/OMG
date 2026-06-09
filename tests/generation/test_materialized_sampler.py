from __future__ import annotations

from torch.utils.data import Dataset

from omg.data.datamodule import DistributedMaterializedShardSampler


class FakeMaterializedDataset(Dataset):
    def __init__(self, span_count: int = 8, span_size: int = 10) -> None:
        self.span_count = int(span_count)
        self.span_size = int(span_size)

    def __len__(self) -> int:
        return self.span_count * self.span_size

    def __getitem__(self, idx: int) -> int:
        return int(idx)

    def materialized_shard_spans(self, base_index: int = 0) -> list[tuple[int, int]]:
        return [
            (int(base_index) + span_idx * self.span_size, self.span_size)
            for span_idx in range(self.span_count)
        ]


def _span_id(index: int, span_size: int) -> int:
    return int(index) // int(span_size)


def test_materialized_sampler_interleaves_active_shards(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    dataset = FakeMaterializedDataset(span_count=8, span_size=10)
    sampler = DistributedMaterializedShardSampler(
        dataset,
        seed=7,
        block_size=1,
        interleave_window=4,
    )

    indices = list(iter(sampler))
    first_spans = [_span_id(index, dataset.span_size) for index in indices[:4]]
    second_spans = [_span_id(index, dataset.span_size) for index in indices[4:8]]

    assert len(set(first_spans)) == 4
    assert len(set(second_spans)) == 4
    assert sorted(indices) == list(range(len(dataset)))


def test_materialized_sampler_partitions_shards_across_ranks(monkeypatch) -> None:
    dataset = FakeMaterializedDataset(span_count=8, span_size=10)
    per_rank = []
    for rank in (0, 1):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", "2")
        sampler = DistributedMaterializedShardSampler(
            dataset,
            seed=11,
            block_size=2,
            interleave_window=3,
        )
        per_rank.append(list(iter(sampler)))

    assert set(per_rank[0]).isdisjoint(per_rank[1])
    assert sorted(per_rank[0] + per_rank[1]) == list(range(len(dataset)))
