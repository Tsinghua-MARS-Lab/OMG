from pathlib import Path

from hydra import compose, initialize_config_dir


def test_compose_transformer():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train", overrides=["exp=base", "logger=none", "trainer=1gpu"])
    assert cfg.model._target_.endswith("MotionGenerator")
    assert cfg.denoiser._target_.endswith("MotionTransformerDenoiser")
    assert cfg.model.text_encoder.model_name == f"{cfg.paths.repo_root}/models/t5-base-local"
    assert cfg.model.scheduler.type == "linear_warmup_cosine"
    assert cfg.model.scheduler.warmup_steps == 2000


def test_compose_300m_diffusion_only():
    config_dir = str(Path(__file__).resolve().parents[2] / "configs" / "generation")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train", overrides=["exp=300m", "logger=none", "trainer=8gpu"])
    assert cfg.denoiser._target_.endswith("MotionTransformerDenoiser")
    assert cfg.loss.simple_root_pos == 0.0
    assert cfg.loss.seam_body_pos == 0.0
    assert cfg.model.use_audio is True
    assert cfg.model.use_human_motion is True
    assert cfg.trainer.devices == 8
    assert cfg.trainer.strategy == "ddp_find_unused_parameters_true"
