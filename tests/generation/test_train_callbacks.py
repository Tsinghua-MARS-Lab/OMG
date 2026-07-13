from __future__ import annotations

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor

from omg.cli.generation.train import _callbacks


def test_callbacks_skip_logger_dependent_callbacks_without_logger() -> None:
    cfg = OmegaConf.create(
        {
            "callbacks": {
                "lr_monitor": {
                    "_target_": "pytorch_lightning.callbacks.LearningRateMonitor",
                    "requires_logger": True,
                    "logging_interval": "step",
                }
            }
        }
    )
    assert _callbacks(cfg, logger_enabled=False) == []
    callbacks = _callbacks(cfg, logger_enabled=True)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], LearningRateMonitor)
