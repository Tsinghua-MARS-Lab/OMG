from __future__ import annotations

from omg.data.datamodule import DistributedWeightedSampler


def test_distributed_weighted_sampler_has_equal_rank_lengths(monkeypatch):
    weights = [1.0] * 101
    rank_indices = []
    for rank in range(4):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", str(rank))
        sampler = DistributedWeightedSampler(weights, num_samples=101, seed=7)
        indices = list(sampler)
        assert len(indices) == 26
        rank_indices.append(indices)

    assert rank_indices[0] != rank_indices[1]


def test_distributed_weighted_sampler_respects_dataset_mass(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    weights = [0.7 / 100.0] * 100 + [0.3 / 100.0] * 100
    sampler = DistributedWeightedSampler(weights, num_samples=20_000, seed=11)

    second_dataset_fraction = sum(index >= 100 for index in sampler) / len(sampler)

    assert 0.28 <= second_dataset_fraction <= 0.32


def test_distributed_weighted_sampler_set_epoch_changes_stream(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    sampler = DistributedWeightedSampler([1.0] * 50, num_samples=50, seed=3)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)

    assert len(epoch0) == len(epoch1) == 25
    assert epoch0 != epoch1
