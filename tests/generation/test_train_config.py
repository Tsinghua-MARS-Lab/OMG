from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from omg.cli.generation.train import _trainer_config


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
