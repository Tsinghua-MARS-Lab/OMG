from __future__ import annotations

import torch
import torch.nn as nn

from omg.generation.denoisers.transformer import _rotate_half
from omg.generation.export import DenoiserStepExportModel, build_export_metadata


class _FakeKinematics:
    kinematics_path = "assets/robots/g1/g1_kinematics.json"


class _FakeRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_dim = 3
        self.sequence_length = 5
        self.num_prev_states = 2
        self.canonical_frame_idx = 1
        self.stats_path = "assets/stats/fake.json"
        self.kinematics = _FakeKinematics()
        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("std", torch.ones(3))


class _FakeDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.objective = "pred_x0"
        self.ddim_eta = 0.0
        self.cfg_scale = 2.5
        self.register_buffer("sample_timestep_map", torch.tensor([0, 9], dtype=torch.long))
        self.register_buffer("sample_alphas_cumprod", torch.tensor([0.9, 0.5], dtype=torch.float32))
        self.register_buffer("sample_alphas_cumprod_prev", torch.tensor([1.0, 0.9], dtype=torch.float32))


class GuidedDiffusion(_FakeDiffusion):
    pass


class _FakeDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 4
        self.input_dim = 3
        self.text_proj = nn.Linear(6, 4)
        self.output = nn.Linear(3, 3)

    def forward(self, x, timesteps, cond_tokens, valid_mask=None):
        assert cond_tokens["extra_tokens"].shape[-1] == 4
        assert cond_tokens["text_context"].shape[-1] == 6
        assert timesteps.shape[:2] == x.shape[:2]
        out = self.output(x)
        if valid_mask is not None:
            out = out.masked_fill(~valid_mask.bool().unsqueeze(-1), 0.0)
        return out


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.representation = _FakeRepresentation()
        self.denoiser = _FakeDenoiser()
        self.diffusion = GuidedDiffusion()
        self.diffusion_target = "future"
        self.condition_dim = 4
        self.text_encoder = None
        self.history_projector = nn.Linear(3, 4)


def test_denoiser_step_export_wrapper_contract():
    model = _FakeModel().eval()
    wrapper = DenoiserStepExportModel(model).eval()
    pred = wrapper(
        torch.randn(1, 5, 3),
        torch.zeros(1, 5, dtype=torch.long),
        torch.ones(1, 5, dtype=torch.bool),
        torch.randn(1, 2, 3),
        torch.randn(1, 7, 6),
        torch.ones(1, 7, dtype=torch.bool),
    )
    assert pred.shape == (1, 5, 3)


def test_build_export_metadata_contract():
    metadata = build_export_metadata(_FakeModel(), opset=17, text_len=7)
    assert metadata["format"] == "omg.denoiser_step"
    assert metadata["diffusion_type"] == "GuidedDiffusion"
    assert metadata["diffusion_target"] == "future"
    assert metadata["objective"] == "pred_x0"
    assert metadata["sequence_length"] == 5
    assert metadata["num_prev_states"] == 2
    assert metadata["feat_dim"] == 3
    assert metadata["text_dim"] == 6
    assert metadata["text_max_length"] == 7
    assert metadata["batch_size"] == 2
    assert metadata["exporter"] == "dynamo"
    assert metadata["export_target"] == "tensorrt"
    assert metadata["tensorrt_compatible"] is True
    assert metadata["sample_timestep_map"] == [0, 9]



def test_export_cli_contract_format():
    from omg.cli.generation.export_onnx import _format_contract

    text = _format_contract(
        {
            "format": "omg.denoiser_step",
            "sequence_length": 60,
            "num_prev_states": 10,
            "feat_dim": 123,
            "text_dim": 768,
            "diffusion_type": "GuidedDiffusion",
        }
    )
    assert "format=omg.denoiser_step" in text
    assert "seq_len=60" in text
    assert "diffusion=GuidedDiffusion" in text



def test_rotate_half_matches_stack_flatten():
    x = torch.randn(2, 3, 4, 8)
    expected = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    assert torch.allclose(_rotate_half(x), expected)



def test_tensorrt_compatible_denoiser_matches_transformer_attention():
    from omg.generation.denoisers.transformer import MotionTransformerDenoiser, RotarySelfAttention
    from omg.generation.export.onnx import (
        TensorRTFriendlyMultiheadAttention,
        TensorRTFriendlyRotarySelfAttention,
        make_tensorrt_compatible_denoiser,
    )

    torch.manual_seed(0)
    denoiser = MotionTransformerDenoiser(
        input_dim=6,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        text_dim=10,
        dropout=0.0,
    ).eval()
    converted = make_tensorrt_compatible_denoiser(denoiser, sequence_length=5).eval()

    assert not any(isinstance(module, nn.MultiheadAttention) for module in converted.modules())
    assert not any(isinstance(module, RotarySelfAttention) for module in converted.modules())
    assert any(isinstance(module, TensorRTFriendlyMultiheadAttention) for module in converted.modules())
    assert any(isinstance(module, TensorRTFriendlyRotarySelfAttention) for module in converted.modules())

    x = torch.randn(1, 5, 6)
    timesteps = torch.zeros(1, 5, dtype=torch.long)
    conditions = {
        "extra_tokens": torch.randn(1, 3, 8),
        "text_context": torch.randn(1, 4, 10),
        "text_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    with torch.no_grad():
        expected = denoiser(x, timesteps, conditions, valid_mask=None)
        actual = converted(x, timesteps, conditions, valid_mask=None)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
