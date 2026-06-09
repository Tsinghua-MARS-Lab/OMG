from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from omg.generation.denoisers.transformer import RotarySelfAttention

EXPORT_METADATA_KEY = "omg_export_metadata"
EXPORT_FORMAT = "omg.denoiser_step"
EXPORT_FORMAT_VERSION = 1
TENSORRT_BLOCKED_ONNX_OPS = frozenset({"SplitToSequence", "SequenceAt", "ConcatFromSequence"})


def _rotate_half_export(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)


class TensorRTFriendlyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

    @classmethod
    def from_mha(cls, module: nn.MultiheadAttention) -> TensorRTFriendlyMultiheadAttention:
        if module.kdim != module.embed_dim or module.vdim != module.embed_dim:
            raise ValueError("TensorRT diffusion export requires kdim == vdim == embed_dim for cross-attention")
        if module.bias_k is not None or module.bias_v is not None or module.add_zero_attn:
            raise ValueError("TensorRT diffusion export does not support bias_k, bias_v, or add_zero_attn")
        if module.in_proj_weight is None:
            raise ValueError("TensorRT diffusion export requires packed MultiheadAttention in_proj_weight")

        converted = cls(module.embed_dim, module.num_heads, bias=module.in_proj_bias is not None)
        q_weight, k_weight, v_weight = module.in_proj_weight.detach().chunk(3, dim=0)
        converted.q_proj.weight.data.copy_(q_weight)
        converted.k_proj.weight.data.copy_(k_weight)
        converted.v_proj.weight.data.copy_(v_weight)
        if module.in_proj_bias is not None:
            q_bias, k_bias, v_bias = module.in_proj_bias.detach().chunk(3, dim=0)
            converted.q_proj.bias.data.copy_(q_bias)
            converted.k_proj.bias.data.copy_(k_bias)
            converted.v_proj.bias.data.copy_(v_bias)
        converted.out_proj.weight.data.copy_(module.out_proj.weight.detach())
        if module.out_proj.bias is not None:
            converted.out_proj.bias.data.copy_(module.out_proj.bias.detach())
        return converted

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, None]:
        if need_weights:
            raise ValueError("TensorRT-friendly export attention does not produce attention weights")
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        q = self.q_proj(query).reshape(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(batch_size, 1, 1, key_len, device=query.device, dtype=query.dtype)
            attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :].bool(), torch.finfo(query.dtype).min)

        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        h = h.transpose(1, 2).reshape(batch_size, query_len, self.embed_dim)
        return self.out_proj(h), None


class TensorRTFriendlyRotarySelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, sequence_length: int, inv_freq: torch.Tensor):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.head_dim = self.hidden_dim // self.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even attention head dim, got {self.head_dim}")
        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        positions = torch.arange(int(sequence_length), dtype=torch.float32)
        rope_freq = inv_freq.detach().to(device=positions.device, dtype=torch.float32)
        freqs = torch.einsum("t,d->td", positions, rope_freq)
        cos = torch.repeat_interleave(freqs.cos(), repeats=2, dim=-1)
        sin = torch.repeat_interleave(freqs.sin(), repeats=2, dim=-1)
        self.register_buffer("cos", cos.unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin", sin.unsqueeze(0).unsqueeze(0), persistent=False)

    @classmethod
    def from_rotary_self_attention(
        cls,
        module: RotarySelfAttention,
        *,
        sequence_length: int,
    ) -> TensorRTFriendlyRotarySelfAttention:
        converted = cls(
            hidden_dim=module.hidden_dim,
            num_heads=module.num_heads,
            sequence_length=sequence_length,
            inv_freq=module.rope.inv_freq.detach(),
        )
        converted.qkv.load_state_dict(module.qkv.state_dict())
        converted.out_proj.load_state_dict(module.out_proj.state_dict())
        return converted

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0, :, :].transpose(1, 2)
        k = qkv[:, :, 1, :, :].transpose(1, 2)
        v = qkv[:, :, 2, :, :].transpose(1, 2)
        cos = self.cos[:, :, :seq_len, :].to(dtype=q.dtype)
        sin = self.sin[:, :, :seq_len, :].to(dtype=q.dtype)
        q = (q * cos) + (_rotate_half_export(q) * sin)
        k = (k * cos) + (_rotate_half_export(k) * sin)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(batch_size, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :].bool(), torch.finfo(x.dtype).min)

        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        h = h.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        h = self.out_proj(h)
        if key_padding_mask is not None:
            h = h.masked_fill(key_padding_mask.unsqueeze(-1).bool(), 0.0)
        return h


def _replace_export_attention(module: nn.Module, *, sequence_length: int) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, TensorRTFriendlyMultiheadAttention.from_mha(child))
        elif isinstance(child, RotarySelfAttention):
            setattr(
                module,
                name,
                TensorRTFriendlyRotarySelfAttention.from_rotary_self_attention(child, sequence_length=sequence_length),
            )
        else:
            _replace_export_attention(child, sequence_length=sequence_length)


def make_tensorrt_compatible_denoiser(denoiser: nn.Module, *, sequence_length: int) -> nn.Module:
    converted = copy.deepcopy(denoiser).eval()
    _replace_export_attention(converted, sequence_length=sequence_length)
    return converted


def _to_list(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def _as_repo_or_abs_path(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def metadata_sidecar_path(onnx_path: str | Path) -> Path:
    path = Path(onnx_path)
    return path.with_suffix(path.suffix + ".meta.json")


class DenoiserStepExportModel(nn.Module):
    """ONNX boundary for one classifier-free-guided diffusion denoiser call.

    The text encoder is intentionally outside this graph. The graph owns the
    trainable motion-conditioning components: history normalization, frame
    condition embedders, and the denoiser.
    """

    def __init__(self, motion_model: nn.Module):
        super().__init__()
        self.history_projector = motion_model.history_projector
        self.use_audio = bool(getattr(motion_model, "use_audio", False))
        self.audio_dim = int(getattr(motion_model, "audio_dim", 0))
        self.use_human_motion = bool(getattr(motion_model, "use_human_motion", False))
        self.human_motion_dim = int(getattr(motion_model, "human_motion_dim", 0))
        self.frame_cond_injection = str(getattr(motion_model, "frame_cond_injection", "sum_to_time"))
        self.audio_embedder = motion_model.audio_embedder if self.use_audio else None
        self.human_motion_embedder = motion_model.human_motion_embedder if self.use_human_motion else None
        if self.use_audio and self.audio_embedder is None:
            raise ValueError("use_audio=True requires motion_model.audio_embedder for ONNX export")
        if self.use_human_motion and self.human_motion_embedder is None:
            raise ValueError("use_human_motion=True requires motion_model.human_motion_embedder for ONNX export")
        self.denoiser = make_tensorrt_compatible_denoiser(
            motion_model.denoiser,
            sequence_length=int(motion_model.representation.sequence_length),
        )
        self.register_buffer("feature_mean", motion_model.representation.mean.detach().clone(), persistent=False)
        self.register_buffer("feature_std", motion_model.representation.std.detach().clone(), persistent=False)

    def _add_frame_condition(
        self,
        conditions: dict[str, torch.Tensor],
        frame_cond: torch.Tensor | None,
        *,
        condition_key: str,
        mask_key: str,
        cond: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        cond = cond * mask.unsqueeze(-1).to(dtype=cond.dtype)
        if self.frame_cond_injection == "sum_to_time":
            return cond if frame_cond is None else frame_cond + cond
        conditions[condition_key] = cond
        if self.frame_cond_injection in {"per_layer_film", "control_local_attn"}:
            conditions[mask_key] = mask
        return frame_cond

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        valid_mask: torch.Tensor,
        history_features: torch.Tensor,
        text_context: torch.Tensor,
        text_mask: torch.Tensor,
        audio_features: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        human_motion: torch.Tensor | None = None,
        human_motion_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        history_norm = (history_features - self.feature_mean.to(history_features)) / self.feature_std.to(history_features)
        history_tokens = self.history_projector(history_norm)
        extra_tokens = [history_tokens]
        conditions = {
            "text_context": text_context,
            "text_mask": text_mask.bool(),
            "extra_tokens": torch.cat(extra_tokens, dim=1),
        }
        frame_cond = None
        if self.use_audio:
            if audio_features is None or audio_mask is None:
                raise ValueError("audio_features and audio_mask are required for audio ONNX export")
            assert self.audio_embedder is not None
            frame_cond = self._add_frame_condition(
                conditions,
                frame_cond,
                condition_key="audio_cond",
                mask_key="audio_mask",
                cond=self.audio_embedder(audio_features.to(dtype=history_features.dtype)),
                mask=audio_mask.bool(),
            )
        if self.use_human_motion:
            if human_motion is None or human_motion_mask is None:
                raise ValueError("human_motion and human_motion_mask are required for human-reference ONNX export")
            assert self.human_motion_embedder is not None
            frame_cond = self._add_frame_condition(
                conditions,
                frame_cond,
                condition_key="human_motion_cond",
                mask_key="human_motion_mask",
                cond=self.human_motion_embedder(human_motion.to(dtype=history_features.dtype)),
                mask=human_motion_mask.bool(),
            )
        if frame_cond is not None:
            conditions["frame_cond"] = frame_cond
        pred = self.denoiser(x, timesteps, conditions, valid_mask=None)
        # The exported planner contract samples fixed full chunks. Keep
        # valid_mask as an ABI input without routing it through attention masks.
        valid_mask_anchor = valid_mask.to(dtype=pred.dtype).sum() * 0.0
        return pred + valid_mask_anchor


def _infer_text_dim(model: nn.Module) -> int:
    text_proj = getattr(model.denoiser, "text_proj", None)
    if text_proj is None or not hasattr(text_proj, "in_features"):
        raise ValueError("Cannot infer denoiser text_dim; expected denoiser.text_proj.in_features")
    return int(text_proj.in_features)


def _infer_hidden_dim(model: nn.Module) -> int:
    if hasattr(model.denoiser, "hidden_dim"):
        return int(model.denoiser.hidden_dim)
    if hasattr(model, "condition_dim"):
        return int(model.condition_dim)
    raise ValueError("Cannot infer hidden_dim; expected denoiser.hidden_dim or model.condition_dim")


def build_export_metadata(
    model: nn.Module,
    *,
    opset: int,
    text_len: int | None = None,
    batch_size: int = 2,
    dynamo: bool = True,
) -> dict[str, Any]:
    diffusion = model.diffusion
    diffusion_name = diffusion.__class__.__name__
    if diffusion_name != "GuidedDiffusion":
        raise ValueError(f"ONNX diffusion-only planner currently supports GuidedDiffusion, got {diffusion_name}")
    diffusion_target = str(getattr(model, "diffusion_target", ""))
    if diffusion_target != "future":
        raise ValueError(f"ONNX diffusion-only planner supports diffusion_target=future, got {diffusion_target}")

    representation = model.representation
    text_encoder = getattr(model, "text_encoder", None)
    max_text_len = int(text_len or getattr(text_encoder, "max_length", 50))
    metadata = {
        "format": EXPORT_FORMAT,
        "format_version": EXPORT_FORMAT_VERSION,
        "opset": int(opset),
        "exporter": "dynamo" if dynamo else "legacy",
        "export_target": "tensorrt",
        "tensorrt_compatible": True,
        "diffusion_type": diffusion_name,
        "diffusion_target": diffusion_target,
        "objective": str(getattr(diffusion, "objective", "")),
        "ddim_eta": float(getattr(diffusion, "ddim_eta", 0.0)),
        "cfg_scale": float(getattr(diffusion, "cfg_scale", 1.0)),
        "sample_timestep_map": _to_list(diffusion.sample_timestep_map),
        "sample_alphas_cumprod": _to_list(diffusion.sample_alphas_cumprod),
        "sample_alphas_cumprod_prev": _to_list(diffusion.sample_alphas_cumprod_prev),
        "feat_dim": int(representation.feat_dim),
        "rotation_representation": str(getattr(representation, "rotation_representation", "quat")),
        "sequence_length": int(representation.sequence_length),
        "num_prev_states": int(representation.num_prev_states),
        "canonical_frame_idx": int(representation.canonical_frame_idx),
        "condition_dim": int(getattr(model, "condition_dim", _infer_hidden_dim(model))),
        "hidden_dim": _infer_hidden_dim(model),
        "text_dim": _infer_text_dim(model),
        "text_max_length": max_text_len,
        "frame_cond_injection": str(getattr(model, "frame_cond_injection", "sum_to_time")),
        "use_audio": bool(getattr(model, "use_audio", False)),
        "audio_dim": int(getattr(model, "audio_dim", 0)),
        "use_human_motion": bool(getattr(model, "use_human_motion", False)),
        "human_motion_dim": int(getattr(model, "human_motion_dim", 0)),
        "text_encoder_model": None if text_encoder is None else str(getattr(text_encoder, "model_name", "")),
        "batch_size": int(batch_size),
        "stats_path": _as_repo_or_abs_path(getattr(representation, "stats_path", None)),
        "kinematics_path": _as_repo_or_abs_path(getattr(representation.kinematics, "kinematics_path", None)),
    }
    return metadata


def save_export_metadata(onnx_path: str | Path, metadata: dict[str, Any]) -> Path:
    sidecar = metadata_sidecar_path(onnx_path)
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def embed_export_metadata(onnx_path: str | Path, metadata: dict[str, Any]) -> None:
    import onnx

    path = Path(onnx_path)
    external_data_path = Path(str(path) + ".data")
    # Avoid materializing >2GB weights into the in-memory proto when the export
    # already uses ONNX external data.
    model_proto = onnx.load(path, load_external_data=False)
    retained = [prop for prop in model_proto.metadata_props if prop.key != EXPORT_METADATA_KEY]
    del model_proto.metadata_props[:]
    model_proto.metadata_props.extend(retained)
    prop = model_proto.metadata_props.add()
    prop.key = EXPORT_METADATA_KEY
    prop.value = json.dumps(metadata, sort_keys=True)
    if external_data_path.exists():
        onnx.save_model(
            model_proto,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path.name,
        )
    else:
        onnx.save(model_proto, path)


def find_tensorrt_blocking_ops(onnx_path: str | Path) -> dict[str, int]:
    import onnx

    model_proto = onnx.load(Path(onnx_path))
    counts: dict[str, int] = {}
    for node in model_proto.graph.node:
        if node.op_type in TENSORRT_BLOCKED_ONNX_OPS:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def validate_tensorrt_compatible_onnx(onnx_path: str | Path) -> None:
    blocking = find_tensorrt_blocking_ops(onnx_path)
    if blocking:
        details = ", ".join(f"{op}={count}" for op, count in sorted(blocking.items()))
        raise RuntimeError(f"TensorRT-incompatible ONNX sequence ops remain in {onnx_path}: {details}")


def load_export_metadata(onnx_path: str | Path, metadata_path: str | Path | None = None) -> dict[str, Any]:
    if metadata_path is not None:
        path = Path(metadata_path)
        return json.loads(path.read_text(encoding="utf-8"))

    sidecar = metadata_sidecar_path(onnx_path)
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))

    import onnx

    model_proto = onnx.load(Path(onnx_path))
    for prop in model_proto.metadata_props:
        if prop.key == EXPORT_METADATA_KEY:
            return json.loads(prop.value)
    raise FileNotFoundError(f"No OMG export metadata found for {onnx_path}")


def export_denoiser_step_onnx(
    model: nn.Module,
    output_path: str | Path,
    *,
    opset: int = 18,
    text_len: int | None = None,
    batch_size: int = 2,
    dynamo: bool = True,
) -> dict[str, Any]:
    if dynamo and int(opset) < 18:
        raise ValueError("The torch dynamo ONNX exporter requires opset >= 18")
    model.eval()
    device = next(model.parameters()).device
    metadata = build_export_metadata(
        model,
        opset=opset,
        text_len=text_len,
        batch_size=batch_size,
        dynamo=dynamo,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    batch = int(batch_size)
    if batch <= 0:
        raise ValueError(f"batch_size must be positive, got {batch}")
    seq_len = int(metadata["sequence_length"])
    history_len = int(metadata["num_prev_states"])
    feat_dim = int(metadata["feat_dim"])
    text_dim = int(metadata["text_dim"])
    max_text_len = int(metadata["text_max_length"])

    wrapper = DenoiserStepExportModel(model).to(device).eval()
    args = (
        torch.randn(batch, seq_len, feat_dim, device=device, dtype=torch.float32),
        torch.zeros(batch, seq_len, device=device, dtype=torch.long),
        torch.ones(batch, seq_len, device=device, dtype=torch.bool),
        torch.randn(batch, history_len, feat_dim, device=device, dtype=torch.float32),
        torch.randn(batch, max_text_len, text_dim, device=device, dtype=torch.float32),
        torch.ones(batch, max_text_len, device=device, dtype=torch.bool),
    )
    input_names = [
        "x",
        "timesteps",
        "valid_mask",
        "history_features",
        "text_context",
        "text_mask",
    ]
    kwargs: dict[str, torch.Tensor] = {}
    if bool(metadata.get("use_audio", False)):
        audio_dim = int(metadata["audio_dim"])
        if audio_dim <= 0:
            raise ValueError(f"Audio ONNX export requires positive audio_dim, got {audio_dim}")
        kwargs["audio_features"] = torch.randn(batch, seq_len, audio_dim, device=device, dtype=torch.float32)
        kwargs["audio_mask"] = torch.ones(batch, seq_len, device=device, dtype=torch.bool)
        input_names.extend(["audio_features", "audio_mask"])
    if bool(metadata.get("use_human_motion", False)):
        human_motion_dim = int(metadata["human_motion_dim"])
        if human_motion_dim <= 0:
            raise ValueError(f"Human-reference ONNX export requires positive human_motion_dim, got {human_motion_dim}")
        kwargs["human_motion"] = torch.randn(batch, seq_len, human_motion_dim, device=device, dtype=torch.float32)
        kwargs["human_motion_mask"] = torch.ones(batch, seq_len, device=device, dtype=torch.bool)
        input_names.extend(["human_motion", "human_motion_mask"])
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            output,
            kwargs=kwargs,
            input_names=input_names,
            output_names=["pred"],
            opset_version=int(opset),
            do_constant_folding=True,
            dynamo=bool(dynamo),
        )
    save_export_metadata(output, metadata)
    embed_export_metadata(output, metadata)
    validate_tensorrt_compatible_onnx(output)
    return metadata
