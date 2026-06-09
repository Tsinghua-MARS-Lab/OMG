from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _install_lightweight_import_fallbacks() -> None:
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


class DummyRepresentation:
    def __init__(self, feat_dim: int, seq_len: int):
        self.feat_dim = int(feat_dim)
        self.sequence_length = int(seq_len)
        self.num_prev_states = 2
        self.mean = torch.zeros(1)

    def normalize_features(self, value: torch.Tensor) -> torch.Tensor:
        return value


class DummyTextEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, captions, has_text, force_null_text, device):
        value = 0.0 if force_null_text else 1.0
        context = torch.full((len(captions), 1, self.output_dim), value, device=device)
        context = context * has_text.to(device=device, dtype=context.dtype)[:, None, None]
        return {"context": context, "mask": has_text[:, None].to(device=device)}


def _batch(batch_size: int, seq_len: int, feat_dim: int, device: torch.device, text: bool, audio: bool, human_motion: bool):
    return {
        "B": batch_size,
        "history_features": torch.zeros(batch_size, 2, feat_dim, device=device),
        "caption": ["walk forward"] * batch_size if text else [""] * batch_size,
        "has_text": torch.full((batch_size,), bool(text), dtype=torch.bool, device=device),
        "audio_features": torch.randn(batch_size, seq_len, 35, device=device),
        "human_motion": torch.randn(batch_size, seq_len, 66, device=device),
        "mask": {
            "valid": torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
            "has_audio": torch.full((batch_size, seq_len), bool(audio), dtype=torch.bool, device=device),
            "has_human_motion": torch.full((batch_size, seq_len), bool(human_motion), dtype=torch.bool, device=device),
        },
    }


def _run_case(model, batch, name: str) -> None:
    x = torch.randn(
        batch["mask"]["valid"].shape[0],
        batch["mask"]["valid"].shape[1],
        model.representation.feat_dim,
        device=next(model.parameters()).device,
    )
    timesteps = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    conditions = model._conditions(batch)
    out = model.denoiser(x, timesteps, conditions, batch["mask"]["valid"])
    frame_cond = conditions.get("frame_cond")
    text_mask = conditions.get("text_mask")
    print(f"[INFO] case={name} output.shape={tuple(out.shape)}")
    print(f"[INFO] case={name} text_context.shape={tuple(conditions['text_context'].shape)}")
    print(f"[INFO] case={name} text_mask_true_ratio={float(text_mask.float().mean().item()):.4f}")
    if frame_cond is None:
        print(f"[INFO] case={name} frame_cond=None")
    else:
        print(
            f"[INFO] case={name} frame_cond.shape={tuple(frame_cond.shape)} "
            f"mean_abs={float(frame_cond.abs().mean().item()):.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run lightweight guided diffusion + MotionTransformerDenoiser condition-forward checks with synthetic data."
        )
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--feat-dim", type=int, default=123)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args()

    sys.path.insert(0, str(_repo_root() / "src"))
    _install_lightweight_import_fallbacks()
    from omg.generation.denoisers.transformer import MotionTransformerDenoiser
    from omg.generation.models.motion_generator import MotionGenerator

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    torch.manual_seed(0)
    print("[INFO] Guided condition forward sanity check")
    print(f"[INFO] device={device} batch_size={args.batch_size} seq_len={args.seq_len}")
    denoiser = MotionTransformerDenoiser(
        input_dim=args.feat_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        text_dim=args.hidden_dim,
        dropout=0.0,
    )
    model = MotionGenerator(
        representation=DummyRepresentation(args.feat_dim, args.seq_len),
        denoiser=denoiser,
        diffusion=object(),
        loss=object(),
        text_encoder=DummyTextEncoder(args.hidden_dim),
        condition_dim=args.hidden_dim,
        text_mask_prob=0.1,
        audio_mask_prob=0.1,
        human_motion_mask_prob=0.1,
        history_mask_prob=0.0,
        use_audio=True,
        use_human_motion=True,
    ).to(device)
    model.eval()

    cases = {
        "text_only": dict(text=True, audio=False, human_motion=False),
        "audio_only": dict(text=False, audio=True, human_motion=False),
        "human_motion_only": dict(text=False, audio=False, human_motion=True),
        "text_audio_human_motion": dict(text=True, audio=True, human_motion=True),
    }
    for name, flags in cases.items():
        batch = _batch(args.batch_size, args.seq_len, args.feat_dim, device, **flags)
        _run_case(model, batch, name)

    cfg_batch = _batch(args.batch_size, args.seq_len, args.feat_dim, device, text=True, audio=True, human_motion=True)
    conditions = model._conditions(cfg_batch, force_null_text=False)
    null_text = model._conditions(cfg_batch, force_null_text=True)
    diff = (conditions["frame_cond"] - null_text["frame_cond"]).abs().max().item()
    print(f"[INFO] cfg_check force_null_text_frame_cond_max_abs_diff={diff:.8f}")
    if diff != 0.0:
        raise AssertionError("force_null_text changed frame_cond; audio/human_motion should be preserved")
    print("[INFO] Condition forward sanity check finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
