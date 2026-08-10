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
        history_pos_encoding="none",
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


def test_recorded_checkpoint_contract_records_history_position_encoding():
    model = _model(self_qk_norm=True, cross_qk_norm=True)
    model.history_pos_encoding = "sinusoidal"
    contract = build_model_architecture_contract(model)
    assert contract["history_pos_encoding"] == "sinusoidal"
    checkpoint = {MODEL_ARCHITECTURE_KEY: contract}
    assert validate_checkpoint_architecture_contract(checkpoint, model)["history_pos_encoding"] == "sinusoidal"


def test_recorded_checkpoint_contract_rejects_history_position_encoding_mismatch():
    trained = _model(self_qk_norm=True, cross_qk_norm=True)
    trained.history_pos_encoding = "sinusoidal"
    current = _model(self_qk_norm=True, cross_qk_norm=True)
    checkpoint = {MODEL_ARCHITECTURE_KEY: build_model_architecture_contract(trained)}
    with pytest.raises(RuntimeError, match="history positional encoding"):
        validate_checkpoint_architecture_contract(checkpoint, current)


def test_older_architecture_contract_without_history_position_encoding_remains_valid():
    model = _model(self_qk_norm=True, cross_qk_norm=True)
    contract = build_model_architecture_contract(model)
    contract["version"] = 1
    contract.pop("history_pos_encoding")
    checkpoint = {MODEL_ARCHITECTURE_KEY: contract}
    assert validate_checkpoint_architecture_contract(checkpoint, model)["history_pos_encoding"] == "none"


def test_current_architecture_contract_requires_history_position_encoding():
    model = _model(self_qk_norm=True, cross_qk_norm=True)
    contract = build_model_architecture_contract(model)
    contract.pop("history_pos_encoding")
    checkpoint = {MODEL_ARCHITECTURE_KEY: contract}
    with pytest.raises(RuntimeError, match="missing history_pos_encoding"):
        validate_checkpoint_architecture_contract(checkpoint, model)


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
