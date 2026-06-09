import sys
import types

import torch
import torch.nn as nn

if "pytorch_lightning" not in sys.modules:
    pl = types.ModuleType("pytorch_lightning")
    pl.LightningModule = nn.Module
    pl.LightningDataModule = object
    sys.modules["pytorch_lightning"] = pl
if "hydra" not in sys.modules:
    hydra = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")
    hydra_utils.instantiate = lambda cfg: cfg
    sys.modules["hydra"] = hydra
    sys.modules["hydra.utils"] = hydra_utils

from omg.generation.denoisers.transformer import MotionTransformerDenoiser
from omg.generation.models.motion_generator import MotionGenerator


class DummyRepresentation:
    feat_dim = 4
    sequence_length = 3
    mean = torch.zeros(1)

    def normalize_features(self, value):
        return value


class DummyTextEncoder(nn.Module):
    def __init__(self, output_dim=8):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, captions, has_text, force_null_text, device):
        value = 0.0 if force_null_text else 1.0
        context = torch.full((len(captions), 1, self.output_dim), value, device=device)
        context = context * has_text.to(device=device, dtype=context.dtype)[:, None, None]
        return {"context": context, "mask": has_text[:, None].to(device=device)}


class ConstantProjector(nn.Module):
    def __init__(self, hidden_dim, value):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.value = float(value)

    def forward(self, x):
        return x.new_full((x.shape[0], x.shape[1], self.hidden_dim), self.value)


def _model(use_audio=False, use_human_motion=False, **kwargs):
    denoiser = MotionTransformerDenoiser(
        input_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        text_dim=8,
        dropout=0.0,
    )
    model = MotionGenerator(
        representation=DummyRepresentation(),
        denoiser=denoiser,
        diffusion=object(),
        loss=object(),
        text_encoder=DummyTextEncoder(),
        condition_dim=8,
        use_audio=use_audio,
        use_human_motion=use_human_motion,
        audio_mask_prob=kwargs.pop("audio_mask_prob", 0.0),
        human_motion_mask_prob=kwargs.pop("human_motion_mask_prob", 0.0),
        history_mask_prob=kwargs.pop("history_mask_prob", 0.0),
        **kwargs,
    )
    if use_audio:
        model.audio_embedder = ConstantProjector(8, 2.0)
    if use_human_motion:
        model.human_motion_embedder = ConstantProjector(8, 3.0)
    return model


def _batch(batch_size=2, seq_len=3):
    return {
        "B": batch_size,
        "history_features": torch.zeros(batch_size, 2, 4),
        "caption": ["walk"] * batch_size,
        "has_text": torch.ones(batch_size, dtype=torch.bool),
        "audio_features": torch.ones(batch_size, seq_len, 35),
        "human_motion": torch.ones(batch_size, seq_len, 66),
        "mask": {
            "valid": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "has_audio": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "has_human_motion": torch.ones(batch_size, seq_len, dtype=torch.bool),
        },
    }


def _denoiser_forward(model, batch):
    conditions = model._conditions(batch)
    x = torch.zeros(2, 3, 4)
    t = torch.zeros(2, dtype=torch.long)
    valid = batch["mask"]["valid"]
    out = model.denoiser(x, t, conditions, valid)
    assert out.shape == x.shape
    if "frame_cond" in conditions:
        assert conditions["frame_cond"].shape == (2, 3, 8)
    if "audio_cond" in conditions:
        assert conditions["audio_cond"].shape == (2, 3, 8)
        assert conditions["audio_mask"].shape == (2, 3)
    if "human_motion_cond" in conditions:
        assert conditions["human_motion_cond"].shape == (2, 3, 8)
        assert conditions["human_motion_mask"].shape == (2, 3)
    return conditions


def test_guided_transformer_forwards_with_text_only():
    model = _model()
    _denoiser_forward(model, _batch())


def test_guided_transformer_forwards_with_audio_only():
    model = _model(use_audio=True)
    batch = _batch()
    batch["has_text"] = torch.zeros(2, dtype=torch.bool)
    conditions = _denoiser_forward(model, batch)
    assert "frame_cond" not in conditions
    assert torch.allclose(conditions["audio_cond"], torch.full((2, 3, 8), 2.0))
    assert conditions["audio_mask"].all()


def test_guided_transformer_forwards_with_human_motion_only():
    model = _model(use_human_motion=True)
    batch = _batch()
    batch["has_text"] = torch.zeros(2, dtype=torch.bool)
    conditions = _denoiser_forward(model, batch)
    assert "frame_cond" not in conditions
    assert torch.allclose(conditions["human_motion_cond"], torch.full((2, 3, 8), 3.0))
    assert conditions["human_motion_mask"].all()


def test_guided_transformer_forwards_with_text_audio_and_human_motion():
    model = _model(use_audio=True, use_human_motion=True)
    conditions = _denoiser_forward(model, _batch())
    assert torch.allclose(conditions["audio_cond"], torch.full((2, 3, 8), 2.0))
    assert torch.allclose(conditions["human_motion_cond"], torch.full((2, 3, 8), 3.0))


def test_force_null_text_preserves_audio_and_human_motion_frame_conditions():
    model = _model(use_audio=True, use_human_motion=True)
    model.eval()
    batch = _batch()
    conditions = model._conditions(batch, force_null_text=False)
    null_text_conditions = model._conditions(batch, force_null_text=True)
    assert torch.allclose(conditions["audio_cond"], null_text_conditions["audio_cond"])
    assert torch.allclose(conditions["human_motion_cond"], null_text_conditions["human_motion_cond"])
    assert not torch.allclose(conditions["text_context"], null_text_conditions["text_context"])


def test_modality_dropout_is_independent():
    model = _model(
        use_audio=True,
        use_human_motion=True,
        text_mask_prob=1.0,
        audio_mask_prob=0.0,
        human_motion_mask_prob=0.0,
    )
    model.train()
    conditions = model._conditions(_batch())
    assert torch.allclose(conditions["audio_cond"], torch.full((2, 3, 8), 2.0))
    assert torch.allclose(conditions["human_motion_cond"], torch.full((2, 3, 8), 3.0))
    assert not conditions["text_mask"].any()


def test_history_dropout_masks_extra_tokens():
    model = _model(history_mask_prob=1.0)
    model.history_projector = ConstantProjector(8, 4.0)
    model.train()
    conditions = model._conditions(_batch())
    assert torch.allclose(conditions["extra_tokens"], torch.zeros(2, 2, 8))


def test_audio_and_human_motion_masks_are_required_when_enabled():
    model = _model(use_audio=True, use_human_motion=True)
    batch = _batch()
    del batch["mask"]["has_audio"]
    try:
        model._conditions(batch)
    except ValueError as exc:
        assert "mask['has_audio'] is required" in str(exc)
    else:
        raise AssertionError("Expected missing has_audio mask to raise")


def test_sum_to_time_mode_remains_supported():
    model = _model(use_audio=True, use_human_motion=True, frame_cond_injection="sum_to_time")
    conditions = _denoiser_forward(model, _batch())
    assert torch.allclose(conditions["frame_cond"], torch.full((2, 3, 8), 5.0))
    assert "audio_cond" not in conditions
    assert "human_motion_cond" not in conditions


def test_per_layer_film_pads_history_prefix_masks_false():
    model = _model(use_audio=True, use_human_motion=True, diffusion_target="history_future")
    conditions = model._conditions(_batch())
    assert conditions["audio_cond"].shape == (2, 5, 8)
    assert conditions["human_motion_cond"].shape == (2, 5, 8)
    assert not conditions["audio_mask"][:, :2].any()
    assert conditions["audio_mask"][:, 2:].all()
    assert not conditions["human_motion_mask"][:, :2].any()
    assert conditions["human_motion_mask"][:, 2:].all()


def test_per_layer_film_requires_masks_with_conditions():
    denoiser = MotionTransformerDenoiser(
        input_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        text_dim=8,
        dropout=0.0,
        frame_cond_injection="per_layer_film",
    )
    x = torch.zeros(2, 3, 4)
    t = torch.zeros(2, dtype=torch.long)
    conditions = {"audio_cond": torch.zeros(2, 3, 8)}
    try:
        denoiser(x, t, conditions, torch.ones(2, 3, dtype=torch.bool))
    except ValueError as exc:
        assert "audio_mask is required" in str(exc)
    else:
        raise AssertionError("Expected missing audio_mask to raise")


def test_per_layer_film_masks_adapter_bias():
    torch.manual_seed(0)
    denoiser = MotionTransformerDenoiser(
        input_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        text_dim=8,
        dropout=0.0,
        frame_cond_injection="per_layer_film",
    )
    denoiser.eval()
    layer = denoiser.layers[0]
    with torch.no_grad():
        layer.audio_film[-1].bias.fill_(0.25)
    x = torch.randn(2, 3, 4)
    t = torch.zeros(2, dtype=torch.long)
    valid = torch.ones(2, 3, dtype=torch.bool)
    no_audio = denoiser(x, t, {}, valid)
    masked_audio = denoiser(
        x,
        t,
        {
            "audio_cond": torch.zeros(2, 3, 8),
            "audio_mask": torch.zeros(2, 3, dtype=torch.bool),
        },
        valid,
    )
    active_audio = denoiser(
        x,
        t,
        {
            "audio_cond": torch.zeros(2, 3, 8),
            "audio_mask": torch.ones(2, 3, dtype=torch.bool),
        },
        valid,
    )
    assert torch.allclose(masked_audio, no_audio)
    assert not torch.allclose(active_audio, no_audio)


def test_per_layer_film_grads_follow_enabled_modalities():
    model = _model(use_audio=True, use_human_motion=False)
    layer = model.denoiser.layers[0]
    assert all(param.requires_grad for param in layer.audio_film.parameters())
    assert not any(param.requires_grad for param in layer.human_motion_film.parameters())
