from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from omg.benchmarks.runners.common import (
    SampleRecord,
    _cfg_output_name,
    _dataset_indices,
    _load_sample_records,
    _motion_input_dim,
    _parse_cfg_scale_value,
    _resolve_sample_records,
    _write_sample_records,
    select_condition_records,
)
from omg.benchmarks.metrics import multimodality
from omg.benchmarks.protocol import benchmark_condition_cohorts, benchmark_source_datasets
from omg.benchmarks.runners.text import (
    _retrieval_distances_and_ranks,
    _sample_metric_rows,
    _sample_rankings,
    _select_sample_records,
    _text_retrieval_summary,
)


class DummyDataset:
    def __init__(self, length: int, prefix: str):
        self.length = length
        self.prefix = prefix

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        return {"caption": f"{self.prefix}-{index}", "meta": {"index": index}}


class StableDummyDataset(DummyDataset):
    def sample_identity(self, index: int):
        return {
            "schema": "omg.benchmark.sample.v2",
            "repo_id": "THU-MARS/OMG-Data",
            "revision": "test-revision",
            "split": "test",
            "episode_index": 100 + int(index),
            "window_start": 0,
            "num_frames": 60,
            "source_dataset": self.prefix,
            "source_id": f"source-{index}",
            "segment_index": 0,
            "source_start_frame": 0,
            "source_end_frame": 60,
        }

    def resolve_identity(self, identity):
        index = int(identity["episode_index"]) - 100
        expected = self.sample_identity(index)
        assert identity == expected
        return index


class DummyEpisodeWindowDataset:
    def __init__(self, captions: list[str], window_counts: list[int]):
        assert len(captions) == len(window_counts)
        self.captions = captions
        self.window_offsets = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(window_counts, dtype=np.int64)]
        )
        self.getitem_calls = 0

    def __len__(self):
        return int(self.window_offsets[-1])

    def __getitem__(self, index: int):
        self.getitem_calls += 1
        episode = int(np.searchsorted(self.window_offsets, index, side="right") - 1)
        return {"caption": self.captions[episode], "meta": {"index": index}}


class DummyConditionDataset:
    def __init__(self, masks):
        self.masks = masks

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index: int):
        valid = self.masks[index]
        return {
            "audio_features": torch.ones(len(valid), 35),
            "mask": {"has_audio": torch.tensor(valid, dtype=torch.bool)},
        }

    def sample_has_condition(self, index: int, condition: str, *, num_frames: int):
        assert condition == "audio"
        valid = self.masks[index]
        return len(valid) >= num_frames and all(valid[:num_frames])

    def global_index(self, index: int):
        return index


def test_lerobot_config_pins_public_dataset_revision():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="train",
            overrides=["data=omg_data_lerobot", "logger=none", "trainer=1gpu"],
        )
    test_opts = cfg.data.dataset_opts.test
    assert list(test_opts) == ["omg_lerobot_test"]
    dataset_cfg = test_opts.omg_lerobot_test
    assert dataset_cfg.repo_id == "THU-MARS/OMG-Data"
    assert dataset_cfg.revision == "6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7"


def test_lerobot_omnimodal_config_enables_all_conditions():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="train",
            overrides=["data=omg_data_lerobot_omnimodal", "logger=none", "trainer=1gpu"],
        )
    dataset_cfg = cfg.data.dataset_opts.test.omg_lerobot_omnimodal_test
    assert dataset_cfg.use_text is True
    assert dataset_cfg.use_audio is True
    assert dataset_cfg.use_human_motion is True


def test_benchmark_cohorts_preserve_release_protocol():
    text = benchmark_condition_cohorts("text", "test")
    audio = benchmark_condition_cohorts("audio", "test")
    humanref = benchmark_condition_cohorts("humanref", "test")
    assert len(text) == 12
    assert len(audio) == 5
    assert len(humanref) == 11
    assert text["amass"] == ("amass_test",)
    assert audio["aioz_gdance"] == ("aioz_gdance_test",)
    assert humanref["beat2"] == (
        "beat2_chinese_test",
        "beat2_english_test",
        "beat2_japanese_test",
        "beat2_spanish_test",
    )
    assert "choreomaster_test" not in benchmark_source_datasets("audio", "test")
    assert set(humanref["beat2"]) <= set(benchmark_source_datasets("humanref", "test"))


def test_select_condition_records_requires_full_condition_window():
    datasets = {"audio": DummyConditionDataset([[True, True, True], [True, False, True], [True, True]])}
    records = select_condition_records(
        datasets,
        num_samples=1,
        seed=0,
        num_frames=3,
        condition="audio",
    )
    assert records == [SampleRecord(dataset="audio", index=0, global_index=0)]


def test_select_sample_records_is_deterministic_and_weighted_by_lengths():
    datasets = {"a": DummyDataset(2, "a"), "b": DummyDataset(3, "b")}
    first = _select_sample_records(datasets, num_samples=4, seed=7)
    second = _select_sample_records(datasets, num_samples=4, seed=7)
    assert first == second
    assert len({(record.dataset, record.index) for record in first}) == 4
    assert all(0 <= record.index < len(datasets[record.dataset]) for record in first)


def test_select_sample_records_clamps_to_available_caption_samples():
    datasets = {"a": DummyDataset(2, "a")}
    records = _select_sample_records(datasets, num_samples=3, seed=0)
    assert len(records) == 2
    assert {(record.dataset, record.index) for record in records} == {("a", 0), ("a", 1)}


def test_select_sample_records_uses_exact_episode_window_spans():
    datasets = {
        "a": DummyEpisodeWindowDataset(["walk", "", "turn"], [2, 3, 1]),
        "b": DummyEpisodeWindowDataset(["jump", "sit"], [2, 2]),
    }
    exhaustive_candidates = [
        SampleRecord(dataset="a", index=0, global_index=0),
        SampleRecord(dataset="a", index=1, global_index=1),
        SampleRecord(dataset="a", index=5, global_index=5),
        SampleRecord(dataset="b", index=0, global_index=6),
        SampleRecord(dataset="b", index=1, global_index=7),
        SampleRecord(dataset="b", index=2, global_index=8),
        SampleRecord(dataset="b", index=3, global_index=9),
    ]
    selected = np.sort(np.random.default_rng(17).choice(len(exhaustive_candidates), size=5, replace=False))

    records = _select_sample_records(datasets, num_samples=5, seed=17)

    assert records == [exhaustive_candidates[int(index)] for index in selected]
    assert all(dataset.getitem_calls == 0 for dataset in datasets.values())


def test_sample_records_round_trip(tmp_path):
    datasets = {"a": StableDummyDataset(2, "a")}
    records = [SampleRecord(dataset="a", index=1, global_index=1)]
    path = tmp_path / "samples.jsonl"
    _write_sample_records(path, records, datasets)
    loaded = _load_sample_records(path)
    assert loaded[0].dataset == "a"
    assert loaded[0].index is None
    assert loaded[0].caption == "a-1"
    assert loaded[0].meta == {"index": 1}
    resolved = _resolve_sample_records(loaded, datasets)
    assert resolved[0].index == 1


def test_motion_input_dim_accepts_evaluator_pose_keys():
    assert _motion_input_dim("qpos_36") == 36
    assert _motion_input_dim("body_pos_local") > _motion_input_dim("qpos_36")
    assert _motion_input_dim("body_link_pos_local") == _motion_input_dim("body_pos_local") - 3


def test_parse_cfg_scale_values_and_output_names():
    assert _parse_cfg_scale_value("default") is None
    assert _parse_cfg_scale_value("none") is None
    assert _parse_cfg_scale_value("2.5") == 2.5
    assert _cfg_output_name(None) == "cfg_default"
    assert _cfg_output_name(2.5) == "cfg_2p5"


def test_dataset_indices_preserve_dataset_membership():
    groups = _dataset_indices(["a", "b", "a", "c"])
    assert groups["a"].tolist() == [0, 2]
    assert groups["b"].tolist() == [1]
    assert groups["c"].tolist() == [3]


def test_text_retrieval_summary_and_sample_rankings():
    text = np.eye(64, dtype=np.float32)
    motion = text.copy()
    motion[32] = text[33]
    distances, ranks = _retrieval_distances_and_ranks(motion[:3], text[:3])
    assert distances.shape == (3,)
    assert ranks.tolist() == [1, 1, 1]
    summary = _text_retrieval_summary(motion, text)
    assert summary["retrieval_batch_size"] == 32
    assert summary["retrieval_num_batches"] == 2
    assert summary["retrieval_num_samples"] == 64
    assert summary["r_precision"] == [63 / 64, 1.0, 1.0]
    rows = _sample_metric_rows(
        captions=[f"caption {idx}" for idx in range(64)],
        dataset_names=["d0"] * 32 + ["d1"] * 32,
        dataset_indices=list(range(64)),
        physical_values={
            "contact_sliding_speed": np.linspace(0.1, 0.3, 64),
            "foot_ground_error": np.linspace(0.0, 0.02, 64),
            "body_jerk_mean": np.linspace(1.0, 3.0, 64),
        },
        generated_embeddings=motion,
        reference_embeddings=text,
        text_embeddings=text,
    )
    assert rows[0]["text_retrieval_batch_size"] == 32
    assert rows[0]["generated_r_at_1"] is True
    assert rows[32]["generated_text_rank"] == 2
    rankings = _sample_rankings(rows, limit=2)
    assert len(rankings["best_generated"]) == 2
    assert len(rankings["worst_generated"]) == 2


def test_multimodality_metric_from_embeddings():
    repeated = np.asarray(
        [
            [[0.0, 0.0], [3.0, 4.0]],
            [[1.0, 1.0], [1.0, 3.0]],
        ],
        dtype=np.float32,
    )
    summary = multimodality(repeated, num_pairs=4, seed=0)
    assert summary["num_texts"] == 2
    assert summary["repeats"] == 2
    assert summary["num_pairs"] == 4
    assert summary["mean"] == pytest.approx(3.5)
    assert summary["min"] == pytest.approx(2.0)
    assert summary["max"] == pytest.approx(5.0)
