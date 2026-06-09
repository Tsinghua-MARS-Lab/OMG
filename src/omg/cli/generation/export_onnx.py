from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from omg.generation.export import export_denoiser_step_onnx, metadata_sidecar_path


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "generation"


def _load_model(cfg, ckpt_path: str | Path, device: torch.device, *, strict: bool):
    model = instantiate(cfg.model)
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    incompatible = model.load_state_dict(state_dict, strict=bool(strict))
    if not strict:
        print(
            "[INFO] Loaded checkpoint with strict=False: "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )
    return model.to(device).eval()


def _default_output_path(exp: str, ckpt_path: str | Path) -> Path:
    ckpt = Path(ckpt_path)
    return Path("models") / "generation" / "onnx" / exp / f"{ckpt.stem}_denoiser_step.onnx"


def _format_contract(metadata: dict) -> str:
    return (
        f"format={metadata['format']} seq_len={metadata['sequence_length']} "
        f"history={metadata['num_prev_states']} feat_dim={metadata['feat_dim']} "
        f"text_dim={metadata['text_dim']} audio={bool(metadata.get('use_audio', False))} "
        f"audio_dim={metadata.get('audio_dim')} humanref={bool(metadata.get('use_human_motion', False))} "
        f"human_dim={metadata.get('human_motion_dim')} diffusion={metadata['diffusion_type']}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an OMG diffusion denoiser step to ONNX.")
    parser.add_argument("--ckpt_path", required=True, help="Training checkpoint to export.")
    parser.add_argument("--exp", required=True, help="Hydra generation experiment name used by the checkpoint.")
    parser.add_argument("--output", default=None, help="Output .onnx path. Defaults under models/generation/onnx/<exp>.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Device used while tracing.")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--text_len", type=int, default=None, help="Override exported text context length.")
    parser.add_argument("--batch_size", type=int, default=2, help="Fixed exported ONNX batch size. Use 2 for batched CFG TensorRT inference.")
    parser.add_argument("--legacy-exporter", action="store_true", help="Use the legacy torch.onnx tracer instead of the dynamo exporter.")
    parser.add_argument("--allow-partial-ckpt", action="store_true", help="Load checkpoint with strict=False before exporting.")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides, e.g. model.text_encoder.model_name=models/t5-base-local")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    device = torch.device(args.device)
    output = Path(args.output) if args.output is not None else _default_output_path(args.exp, args.ckpt_path)

    with initialize_config_dir(config_dir=str(_config_dir()), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"exp={args.exp}",
                "logger=none",
                "trainer=1gpu",
                *args.overrides,
            ],
        )

    model = _load_model(cfg, args.ckpt_path, device=device, strict=not bool(args.allow_partial_ckpt))
    metadata = export_denoiser_step_onnx(
        model,
        output,
        opset=args.opset,
        text_len=args.text_len,
        batch_size=args.batch_size,
        dynamo=not args.legacy_exporter,
    )
    print(f"onnx={output.resolve()}")
    print(f"metadata={metadata_sidecar_path(output).resolve()}")
    print(f"contract={_format_contract(metadata)}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
