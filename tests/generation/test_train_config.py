from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from omg.cli.generation.train import _trainer_config


GENERATION_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "generation"


@pytest.mark.parametrize(
    ("exp_name", "train_batch_size", "accumulate_grad_batches"),
    [
        ("base", 192, 1),
        ("50m", 192, 1),
        ("100m", 192, 1),
        ("300m", 128, 1),
        ("500m", 64, 3),
        ("1b", 32, 6),
    ],
)
def test_experiment_presets_share_omnimodal_sinusoidal_training_protocol(
    exp_name: str,
    train_batch_size: int,
    accumulate_grad_batches: int,
):
    with initialize_config_dir(config_dir=str(GENERATION_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=[f"exp={exp_name}"])

    assert all(float(value) == 0.0 for key, value in cfg.loss.items() if key != "_target_")
    assert cfg.model.history_pos_encoding == "sinusoidal"
    assert cfg.model.use_audio is True
    assert cfg.model.use_human_motion is True
    assert cfg.denoiser.self_attention_qk_norm is True
    assert cfg.denoiser.cross_attention_qk_norm is True
    assert cfg.trainer.devices == 4
    assert cfg.trainer.max_steps == 180000
    assert cfg.trainer.accumulate_grad_batches == accumulate_grad_batches
    assert cfg.model.optimizer.lr == 1.0e-4
    assert cfg.model.scheduler.t_max == 180000
    assert cfg.data.loader_opts.train.batch_size == train_batch_size
    assert cfg.data.loader_opts.train.num_workers == 2
    assert cfg.data.loader_opts.train.prefetch_factor == 2
    assert list(cfg.data.dataset_opts.train) == ["omg_data_materialized_omnimodal_train"]


def test_trainer_config_disables_sampler_replacement_for_custom_train_sampler():
    cfg = OmegaConf.create({"trainer": {"accelerator": "gpu", "devices": 8, "strategy": "ddp"}})
    datamodule = SimpleNamespace(train_sampler=object())

    trainer_cfg = _trainer_config(cfg, datamodule)

    assert trainer_cfg["use_distributed_sampler"] is False


def test_trainer_config_preserves_explicit_sampler_replacement_setting():
    cfg = OmegaConf.create(
        {"trainer": {"accelerator": "gpu", "devices": 8, "strategy": "ddp", "use_distributed_sampler": True}}
    )
    datamodule = SimpleNamespace(train_sampler=object())

    trainer_cfg = _trainer_config(cfg, datamodule)

    assert trainer_cfg["use_distributed_sampler"] is True
