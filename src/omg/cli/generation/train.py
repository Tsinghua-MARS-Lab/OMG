from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from omg.core.logging import Log


_CKPT_PATH_OVERRIDE_KEYS = {"ckpt_path", "init_weights_only_ckpt"}


def _quote_checkpoint_path_overrides(argv: list[str]) -> list[str]:
    # Hydra treats unquoted "=" in override values as grammar, but checkpoint filenames often contain it.
    quoted: list[str] = []
    for arg in argv:
        if "=" not in arg or arg.startswith("-"):
            quoted.append(arg)
            continue
        key, value = arg.split("=", 1)
        normalized_key = key.lstrip("+")
        if normalized_key not in _CKPT_PATH_OVERRIDE_KEYS or not value:
            quoted.append(arg)
            continue
        if value[0] in {"'", '"'}:
            quoted.append(arg)
            continue
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        quoted.append(f'{key}="{escaped}"')
    return quoted


def _prepare_argv_for_hydra() -> None:
    sys.argv = _quote_checkpoint_path_overrides(sys.argv)


def _callbacks(cfg: DictConfig, *, logger_enabled: bool):
    callbacks = []
    if "callbacks" not in cfg or cfg.callbacks is None:
        return callbacks
    for _, cb_cfg in cfg.callbacks.items():
        if cb_cfg is not None and "_target_" in cb_cfg:
            resolved = OmegaConf.create(OmegaConf.to_container(cb_cfg, resolve=True))
            requires_logger = bool(resolved.pop("requires_logger", False))
            if requires_logger and not logger_enabled:
                continue
            callbacks.append(instantiate(resolved))
    return callbacks


def _trainer_plugins(cfg: DictConfig):
    plugins = []
    checkpoint_io = cfg.get("checkpoint_io")
    if checkpoint_io is not None and "_target_" in checkpoint_io:
        plugins.append(instantiate(checkpoint_io))
    return plugins


def _trainer_config(cfg: DictConfig, datamodule) -> dict:
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    if getattr(datamodule, "train_sampler", None) is not None and "use_distributed_sampler" not in trainer_cfg:
        trainer_cfg["use_distributed_sampler"] = False
        Log.info("[Trainer]: disabled Lightning distributed sampler replacement for custom train sampler")
    return trainer_cfg


def _load_weights_only_checkpoint(
    model: pl.LightningModule,
    checkpoint_path: str | os.PathLike[str],
    *,
    strict: bool = True,
) -> None:
    path = Path(checkpoint_path)
    checkpoint: Any = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Weights-only checkpoint does not contain a Lightning state_dict: {path}")
    state_dict = checkpoint["state_dict"]
    adapted_mismatched: list[str] = []
    skipped_mismatched: list[str] = []
    if not bool(strict):
        current_state = model.state_dict()
        filtered_state = {}
        for key, value in state_dict.items():
            current_value = current_state.get(key)
            if current_value is not None and tuple(current_value.shape) != tuple(value.shape):
                detail = f"{key}: checkpoint={tuple(value.shape)} model={tuple(current_value.shape)}"
                if value.ndim != current_value.ndim:
                    skipped_mismatched.append(detail)
                    continue
                adapted = current_value.clone()
                overlap = tuple(
                    slice(0, min(int(checkpoint_size), int(model_size)))
                    for checkpoint_size, model_size in zip(value.shape, current_value.shape, strict=True)
                )
                adapted[overlap] = value[overlap].to(device=adapted.device, dtype=adapted.dtype)
                filtered_state[key] = adapted
                adapted_mismatched.append(detail)
                continue
            filtered_state[key] = value
        state_dict = filtered_state
    missing, unexpected = model.load_state_dict(state_dict, strict=bool(strict))
    if bool(strict) and (missing or unexpected):
        raise RuntimeError(
            f"Failed to load weights-only checkpoint strictly from {path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if missing or unexpected or adapted_mismatched or skipped_mismatched:
        Log.info(
            "[Checkpoint]: initialized model weights non-strictly from %s missing=%s unexpected=%s "
            "adapted_mismatched=%s skipped_mismatched=%s",
            path,
            missing,
            unexpected,
            adapted_mismatched,
            skipped_mismatched,
        )
    else:
        Log.info("[Checkpoint]: initialized model weights from %s", path)


def _log_stage(stage: str, start: float | None = None) -> float:
    now = time.perf_counter()
    if start is None:
        Log.info("[TrainStage] %s:start", stage)
    else:
        Log.info("[TrainStage] %s:done elapsed_sec=%.3f", stage, now - start)
    return now


def run(cfg: DictConfig) -> None:
    Log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    stage_start = _log_stage("seed")
    pl.seed_everything(int(cfg.seed), workers=True)
    _log_stage("seed", stage_start)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    stage_start = _log_stage("instantiate_datamodule")
    datamodule = instantiate(cfg.data, _recursive_=False)
    Log.info("[TrainStage] instantiate_datamodule:class=%s", type(datamodule).__name__)
    _log_stage("instantiate_datamodule", stage_start)

    stage_start = _log_stage("instantiate_model")
    model = instantiate(cfg.model)
    Log.info("[TrainStage] instantiate_model:class=%s", type(model).__name__)
    Log.info(
        "[INFO] 🧭 CONDITION_INJECTION=%s (training; model.frame_cond_injection=%s)",
        getattr(model, "frame_cond_injection", None),
        getattr(model, "frame_cond_injection", None),
    )
    _log_stage("instantiate_model", stage_start)

    init_weights_only_ckpt = cfg.get("init_weights_only_ckpt")
    if init_weights_only_ckpt:
        if cfg.get("ckpt_path"):
            raise ValueError("Use either ckpt_path for full resume or init_weights_only_ckpt for weight initialization, not both")
        stage_start = _log_stage("load_weights_only_checkpoint")
        _load_weights_only_checkpoint(
            model,
            init_weights_only_ckpt,
            strict=bool(cfg.get("init_weights_strict", True)),
        )
        _log_stage("load_weights_only_checkpoint", stage_start)

    logger = False
    if cfg.get("logger", {}).get("enabled", False):
        stage_start = _log_stage("instantiate_logger")
        logger_cfg = OmegaConf.create(OmegaConf.to_container(cfg.logger, resolve=True))
        logger_cfg.pop("enabled", None)
        logger = instantiate(logger_cfg)
        Log.info("[TrainStage] instantiate_logger:class=%s", type(logger).__name__)
        _log_stage("instantiate_logger", stage_start)

    stage_start = _log_stage("build_trainer_config")
    trainer_cfg = _trainer_config(cfg, datamodule)
    plugins = _trainer_plugins(cfg)
    if plugins:
        existing_plugins = trainer_cfg.pop("plugins", None)
        if existing_plugins is None:
            trainer_cfg["plugins"] = plugins
        elif isinstance(existing_plugins, list):
            trainer_cfg["plugins"] = [*existing_plugins, *plugins]
        else:
            trainer_cfg["plugins"] = [existing_plugins, *plugins]
    _log_stage("build_trainer_config", stage_start)

    stage_start = _log_stage("instantiate_callbacks")
    callbacks = _callbacks(cfg, logger_enabled=logger is not False)
    Log.info("[TrainStage] instantiate_callbacks:classes=%s", [type(callback).__name__ for callback in callbacks])
    _log_stage("instantiate_callbacks", stage_start)

    stage_start = _log_stage("instantiate_trainer")
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, default_root_dir=cfg.output_dir, **trainer_cfg)
    _log_stage("instantiate_trainer", stage_start)

    if cfg.task == "fit":
        stage_start = _log_stage("trainer_fit")
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        _log_stage("trainer_fit", stage_start)
    elif cfg.task == "validate":
        stage_start = _log_stage("trainer_validate")
        trainer.validate(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        _log_stage("trainer_validate", stage_start)
    else:
        raise ValueError(f"Unsupported task: {cfg.task}")


@hydra.main(version_base="1.3", config_path="../../../../configs/generation", config_name="train")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _prepare_argv_for_hydra()
    main()
