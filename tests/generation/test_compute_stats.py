from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch

from omg.cli.generation.compute_stats import _update_moments


def test_update_moments_accumulates_float32_inputs_in_float64() -> None:
    values = torch.tensor(
        [[10000.0, -10000.0], [10000.25, -9999.75], [9999.75, -10000.25]],
        dtype=torch.float32,
    )
    count, mean, m2 = _update_moments(0, None, None, values[:2])
    count, mean, m2 = _update_moments(count, mean, m2, values[2:])

    expected = values.double()
    assert mean.dtype == torch.float64
    assert m2.dtype == torch.float64
    torch.testing.assert_close(mean, expected.mean(dim=0), rtol=0.0, atol=1e-12)
    torch.testing.assert_close((m2 / (count - 1)).sqrt(), expected.std(dim=0), rtol=0.0, atol=1e-12)


def test_compute_stats_allows_an_empty_distributed_partition(monkeypatch) -> None:
    module = importlib.import_module("omg.cli.generation.compute_stats")

    class EmptyDataset:
        def iter_stats_batches(self, **kwargs):
            return iter(())

    configs = {
        "data.yaml": {"dataset_opts": {"train": {"fake": {"_target_": "fake.Dataset"}}}},
        "representation.yaml": {
            "feat_dim": 125,
            "rotation_representation": "rot6d",
            "num_prev_states": 10,
            "canonical_frame_idx": 9,
            "sequence_length": 60,
        },
        "paths.yaml": {},
    }
    monkeypatch.setattr(module, "_load_yaml", lambda path: configs[Path(path).name])
    monkeypatch.setattr(module, "instantiate", lambda cfg: EmptyDataset())

    def contribute_other_rank(aggregate, op):
        del op
        aggregate.zero_()
        aggregate[0] = 2
        aggregate[1] = 2
        aggregate[2 : 2 + 125] = 2
        aggregate[2 + 125 : 2 + 250] = 2
        aggregate[2 + 250 + 3] = 2

    monkeypatch.setattr(module.dist, "all_reduce", contribute_other_rank)
    args = argparse.Namespace(
        data_config=Path("data.yaml"),
        representation_config=Path("representation.yaml"),
        paths_config=Path("paths.yaml"),
        split="train",
        batch_size=2,
        max_samples=1,
        device="cpu",
        episode_batch_frames=32,
        rank=0,
        world_size=2,
        std_min=1e-6,
    )
    stats = module.compute_stats(args)
    assert stats["count"] == 2
    assert stats["mean"] == [1.0] * 125
    assert stats["default_root_quat"] == [1.0, 0.0, 0.0, 0.0]
