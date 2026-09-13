from types import SimpleNamespace
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from omg.generation.architecture import apply_checkpoint_architecture_config, build_model_architecture_contract, MODEL_ARCHITECTURE_KEY
from omg.generation.export.onnx import _ieee_fp32_validation
from omg.pipeline.planner import OnnxDiffusionPlanner


def test_checkpoint_config_and_conflicts():
    m = SimpleNamespace(denoiser=SimpleNamespace(self_attention_qk_norm=True, cross_attention_qk_norm=True), history_pos_encoding="sinusoidal", frame_cond_injection="per_layer_film")
    c = {MODEL_ARCHITECTURE_KEY: build_model_architecture_contract(m)}
    cfg = OmegaConf.create({"denoiser": {"self_attention_qk_norm": False, "cross_attention_qk_norm": False}, "model": {"history_pos_encoding": "none", "frame_cond_injection": "per_layer_film"}})
    with pytest.raises(RuntimeError, match="conflicts"):
        apply_checkpoint_architecture_config(cfg, c, explicit_overrides=["model.history_pos_encoding=none"])
    apply_checkpoint_architecture_config(cfg, c)
    assert cfg.model.history_pos_encoding == "sinusoidal"
    assert cfg.denoiser.self_attention_qk_norm is True


def test_legacy_requires_both_declarations():
    cfg = OmegaConf.create({})
    with pytest.raises(RuntimeError, match="legacy-attention-contract"):
        apply_checkpoint_architecture_config(cfg, {})
    with pytest.raises(RuntimeError, match="history_pos_encoding"):
        apply_checkpoint_architecture_config(cfg, {}, legacy_attention_contract="none")


def test_text_cache_reuses_and_is_bounded():
    p = object.__new__(OnnxDiffusionPlanner)
    p._text_condition_cache = {}
    calls = []
    p._encode_text = lambda text: (calls.append(text), text)
    assert p._text_conditions("same") == p._text_conditions("same")
    assert calls == ["same"]
    for i in range(140):
        p._text_conditions(str(i))
    assert len(p._text_condition_cache) == 128


def test_fp32_precision_restored_on_error():
    old = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        @_ieee_fp32_validation
        def fail():
            assert torch.get_float32_matmul_precision() == "highest"
            raise ValueError("test")
        with pytest.raises(ValueError):
            fail()
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(old)


def test_text_cfg_routes_to_one_joint_call():
    p = object.__new__(OnnxDiffusionPlanner)
    calls = []
    p._onnx_joint_cfg_pred = lambda *args, **kwargs: calls.append((args, kwargs)) or np.ones(1)
    x = np.zeros((1, 2, 3), dtype=np.float32)
    result = p._onnx_cfg_pred(x, 1, None, None, {}, {}, 1., cfg_text_scale=2.5, cfg_audio_scale=0., cfg_human_scale=0.)
    assert len(calls) == 1
    assert calls[0][0][-1] == 2.5
    assert result[0] == 1
