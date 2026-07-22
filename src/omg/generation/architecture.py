from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MODEL_ARCHITECTURE_KEY = "omg_model_architecture"
MODEL_ARCHITECTURE_FORMAT = "omg.model_architecture"
MODEL_ARCHITECTURE_VERSION = 1

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


def build_model_architecture_contract(model: Any) -> dict[str, Any]:
    denoiser = model.denoiser
    return {
        "format": MODEL_ARCHITECTURE_FORMAT,
        "version": MODEL_ARCHITECTURE_VERSION,
        "denoiser_type": f"{type(denoiser).__module__}.{type(denoiser).__qualname__}",
        "frame_cond_injection": str(getattr(model, "frame_cond_injection", "")),
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
    if int(contract.get("version", -1)) != MODEL_ARCHITECTURE_VERSION:
        raise RuntimeError(
            "Unsupported checkpoint architecture version: "
            f"{contract.get('version')!r}; expected {MODEL_ARCHITECTURE_VERSION}"
        )
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

    if actual["attention"] != expected_attention:
        raise RuntimeError(
            "Instantiated denoiser attention semantics do not match the "
            f"{source}: expected={expected_attention}, actual={actual['attention']}. "
            "Set denoiser.self_attention_qk_norm and denoiser.cross_attention_qk_norm to the checkpoint's "
            "training values before exporting."
        )
    return actual
