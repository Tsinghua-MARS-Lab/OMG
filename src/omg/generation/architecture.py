from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MODEL_ARCHITECTURE_KEY = "omg_model_architecture"
MODEL_ARCHITECTURE_FORMAT = "omg.model_architecture"
MODEL_ARCHITECTURE_VERSION = 2
SUPPORTED_MODEL_ARCHITECTURE_VERSIONS = {1, MODEL_ARCHITECTURE_VERSION}

LEGACY_ATTENTION_CONTRACTS = {
    "none": {
        "rotary_self_attention_qk_norm": False,
        "cross_attention_qk_norm": False,
    },
    "cross-only": {
        "rotary_self_attention_qk_norm": False,
        "cross_attention_qk_norm": True,
    },
    "self-only": {
        "rotary_self_attention_qk_norm": True,
        "cross_attention_qk_norm": False,
    },
    "self-and-cross": {
        "rotary_self_attention_qk_norm": True,
        "cross_attention_qk_norm": True,
    },
}


def apply_checkpoint_architecture_config(cfg, checkpoint, *, explicit_overrides=()):
    """Resolve semantic architecture fields before instantiation; reject user conflicts."""
    from omegaconf import OmegaConf

    recorded = checkpoint.get(MODEL_ARCHITECTURE_KEY)
    if recorded is None:
        return  # Legacy declarations are handled separately, never inferred from tensors.
    _validate_contract_shape(recorded)
    values = {
        "denoiser.self_attention_qk_norm": recorded["attention"]["rotary_self_attention_qk_norm"],
        "denoiser.cross_attention_qk_norm": recorded["attention"]["cross_attention_qk_norm"],
        "model.history_pos_encoding": recorded.get("history_pos_encoding", "none"),
        "model.frame_cond_injection": recorded["frame_cond_injection"],
    }
    explicit = {item.split("=", 1)[0].lstrip("+") for item in explicit_overrides}
    for key, value in values.items():
        if key in explicit and OmegaConf.select(cfg, key) != value:
            raise RuntimeError(f"Explicit {key} conflicts with checkpoint: {OmegaConf.select(cfg, key)!r} != {value!r}")
        OmegaConf.update(cfg, key, value)


def build_model_architecture_contract(model: Any) -> dict[str, Any]:
    denoiser = model.denoiser
    return {
        "format": MODEL_ARCHITECTURE_FORMAT,
        "version": MODEL_ARCHITECTURE_VERSION,
        "denoiser_type": f"{type(denoiser).__module__}.{type(denoiser).__qualname__}",
        "frame_cond_injection": str(getattr(model, "frame_cond_injection", "")),
        "history_pos_encoding": str(getattr(model, "history_pos_encoding", "none")),
        "attention": {
            "rotary_self_attention_qk_norm": bool(
                getattr(denoiser, "self_attention_qk_norm", False)
            ),
            "cross_attention_qk_norm": bool(
                getattr(denoiser, "cross_attention_qk_norm", False)
            ),
        },
    }


def _validate_contract_shape(contract: Mapping[str, Any]) -> None:
    if contract.get("format") != MODEL_ARCHITECTURE_FORMAT:
        raise RuntimeError(
            "Unsupported checkpoint architecture format: "
            f"{contract.get('format')!r}; expected {MODEL_ARCHITECTURE_FORMAT!r}"
        )
    version = int(contract.get("version", -1))
    if version not in SUPPORTED_MODEL_ARCHITECTURE_VERSIONS:
        raise RuntimeError(
            "Unsupported checkpoint architecture version: "
            f"{contract.get('version')!r}; expected one of "
            f"{sorted(SUPPORTED_MODEL_ARCHITECTURE_VERSIONS)}"
        )
    if version >= 2 and "history_pos_encoding" not in contract:
        raise RuntimeError("Checkpoint architecture contract is missing history_pos_encoding")
    attention = contract.get("attention")
    if not isinstance(attention, Mapping):
        raise RuntimeError("Checkpoint architecture contract is missing the attention mapping")
    missing = {
        "rotary_self_attention_qk_norm",
        "cross_attention_qk_norm",
    } - set(attention)
    if missing:
        raise RuntimeError(f"Checkpoint architecture attention contract is missing keys: {sorted(missing)}")


def validate_checkpoint_architecture_contract(
    checkpoint: Mapping[str, Any],
    model: Any,
    *,
    legacy_attention_contract: str | None = None,
) -> dict[str, Any]:
    actual = build_model_architecture_contract(model)
    recorded = checkpoint.get(MODEL_ARCHITECTURE_KEY)

    if recorded is None:
        if legacy_attention_contract is None:
            choices = ", ".join(LEGACY_ATTENTION_CONTRACTS)
            raise RuntimeError(
                "Checkpoint has no OMG architecture contract. Attention normalization changes no parameter "
                "shapes, so strict state-dict loading cannot detect this semantic mismatch. Re-run with "
                f"--legacy-attention-contract {{{choices}}} and matching Hydra denoiser overrides."
            )
        if legacy_attention_contract not in LEGACY_ATTENTION_CONTRACTS:
            raise ValueError(f"Unknown legacy attention contract: {legacy_attention_contract!r}")
        expected_attention = LEGACY_ATTENTION_CONTRACTS[legacy_attention_contract]
        source = f"legacy declaration {legacy_attention_contract!r}"
    else:
        if not isinstance(recorded, Mapping):
            raise RuntimeError(f"Checkpoint {MODEL_ARCHITECTURE_KEY} must be a mapping")
        _validate_contract_shape(recorded)
        expected_attention = {
            key: bool(recorded["attention"][key])
            for key in (
                "rotary_self_attention_qk_norm",
                "cross_attention_qk_norm",
            )
        }
        source = "checkpoint architecture contract"
        for key in ("denoiser_type", "frame_cond_injection"):
            if str(recorded.get(key, "")) != str(actual[key]):
                raise RuntimeError(
                    f"Instantiated model {key} does not match the checkpoint architecture contract: "
                    f"expected={recorded.get(key)!r}, actual={actual[key]!r}"
                )
        recorded_history_pos_encoding = recorded.get("history_pos_encoding")
        if recorded_history_pos_encoding is not None:
            actual_history_pos_encoding = actual["history_pos_encoding"]
            if str(recorded_history_pos_encoding) != str(actual_history_pos_encoding):
                raise RuntimeError(
                    "Instantiated model history positional encoding does not match the checkpoint "
                    f"architecture contract: expected={recorded_history_pos_encoding!r}, "
                    f"actual={actual_history_pos_encoding!r}."
                )

    if actual["attention"] != expected_attention:
        raise RuntimeError(
            "Instantiated denoiser attention semantics do not match the "
            f"{source}: expected={expected_attention}, actual={actual['attention']}. "
            "Set denoiser.self_attention_qk_norm and denoiser.cross_attention_qk_norm to the checkpoint's "
            "training values before exporting."
        )
    return actual
