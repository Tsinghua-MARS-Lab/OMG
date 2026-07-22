from __future__ import annotations

from types import SimpleNamespace

import pytest

from omg.generation.architecture import (
    MODEL_ARCHITECTURE_KEY,
    build_model_architecture_contract,
    validate_checkpoint_architecture_contract,
)
from omg.generation.models.motion_generator import MotionGenerator


def _model(*, self_qk_norm: bool, cross_qk_norm: bool):
    denoiser = SimpleNamespace(
        self_attention_qk_norm=self_qk_norm,
        cross_attention_qk_norm=cross_qk_norm,
    )
    return SimpleNamespace(
        denoiser=denoiser,
        frame_cond_injection="per_layer_film",
    )


def test_checkpoint_hook_records_attention_architecture():
    model = _model(self_qk_norm=False, cross_qk_norm=True)
    checkpoint = {}
    MotionGenerator.on_save_checkpoint(model, checkpoint)
    assert checkpoint[MODEL_ARCHITECTURE_KEY]["attention"] == {
        "rotary_self_attention_qk_norm": False,
        "cross_attention_qk_norm": True,
    }


def test_recorded_checkpoint_contract_validates_matching_model():
    model = _model(self_qk_norm=True, cross_qk_norm=True)
    checkpoint = {MODEL_ARCHITECTURE_KEY: build_model_architecture_contract(model)}
    assert validate_checkpoint_architecture_contract(checkpoint, model)["attention"] == {
        "rotary_self_attention_qk_norm": True,
        "cross_attention_qk_norm": True,
    }


def test_recorded_checkpoint_contract_rejects_semantic_mismatch():
    trained = _model(self_qk_norm=False, cross_qk_norm=True)
    current = _model(self_qk_norm=True, cross_qk_norm=True)
    checkpoint = {MODEL_ARCHITECTURE_KEY: build_model_architecture_contract(trained)}
    with pytest.raises(RuntimeError, match="do not match"):
        validate_checkpoint_architecture_contract(checkpoint, current)


def test_legacy_checkpoint_requires_explicit_attention_contract():
    model = _model(self_qk_norm=False, cross_qk_norm=True)
    with pytest.raises(RuntimeError, match="--legacy-attention-contract"):
        validate_checkpoint_architecture_contract({}, model)


def test_legacy_cross_only_contract_accepts_matching_model():
    model = _model(self_qk_norm=False, cross_qk_norm=True)
    contract = validate_checkpoint_architecture_contract(
        {},
        model,
        legacy_attention_contract="cross-only",
    )
    assert contract["attention"] == {
        "rotary_self_attention_qk_norm": False,
        "cross_attention_qk_norm": True,
    }


def test_legacy_contract_rejects_wrong_instantiated_architecture():
    model = _model(self_qk_norm=True, cross_qk_norm=True)
    with pytest.raises(RuntimeError, match="do not match"):
        validate_checkpoint_architecture_contract(
            {},
            model,
            legacy_attention_contract="cross-only",
        )
