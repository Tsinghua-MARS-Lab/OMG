from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from omg.generation.conditions.t5 import FrozenT5TextEncoder
from omg.generation.export import load_export_metadata
from omg.motion.representation import G1MotionRepresentation
from omg.runtime.onnx_providers import (
    DEFAULT_DIFFUSION_ONNX_PROVIDERS,
    DEFAULT_TENSORRT_ONNX_PROVIDERS,
    parse_onnx_providers,
    prepare_onnx_provider_runtime,
    validate_active_onnx_providers,
    validate_onnx_providers,
)
from omg.robots.g1.constants import QPOS_DIM


@dataclass(frozen=True)
class DiffusionContinuationState:
    latent_states: np.ndarray
    valid_steps: np.ndarray
    sample_timestep_map: np.ndarray
    canonical_root_pos: np.ndarray
    canonical_root_quat: np.ndarray


@dataclass(frozen=True)
class MotionPlan:
    qpos_36: np.ndarray
    motion_features: np.ndarray
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)
    continuation_state: DiffusionContinuationState | None = None


@dataclass(frozen=True)
class _ContinuationInit:
    x0: np.ndarray
    noise: np.ndarray
    start_step: int
    overlap_frames: int


@dataclass(frozen=True)
class _DitCacheStats:
    enabled: bool
    threshold: float | None
    warmup_steps: int
    max_consecutive: int
    executed_steps: int
    skipped_steps: int
    cosine_mean: float | None
    cosine_min: float | None
    cosine_max: float | None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "warmup_steps": self.warmup_steps,
            "max_consecutive": self.max_consecutive,
            "executed_steps": self.executed_steps,
            "skipped_steps": self.skipped_steps,
            "cosine_mean": self.cosine_mean,
            "cosine_min": self.cosine_min,
            "cosine_max": self.cosine_max,
        }


def _continuation_start_step(num_sampling_steps: int, continuation_steps: int) -> int:
    sampling_steps = int(num_sampling_steps)
    steps = int(continuation_steps)
    if sampling_steps <= 0:
        raise ValueError(f"num_sampling_steps must be positive, got {sampling_steps}")
    if steps <= 0 or steps > sampling_steps:
        raise ValueError(f"continuation_steps must be in [1, {sampling_steps}], got {steps}")
    return steps - 1


def _cosine_similarity_flat(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(left.shape[0], -1)
    b = np.asarray(right, dtype=np.float64).reshape(right.shape[0], -1)
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denominator > 0.0
    if not np.any(valid):
        return 0.0
    return float(np.mean(numerator[valid] / denominator[valid]))


def _dit_cache_stats(
    *,
    enabled: bool,
    threshold: float | None,
    warmup_steps: int,
    max_consecutive: int,
    executed_steps: int,
    skipped_steps: int,
    cosines: list[float],
) -> _DitCacheStats:
    if cosines:
        cosine_array = np.asarray(cosines, dtype=np.float64)
        cosine_mean = float(cosine_array.mean())
        cosine_min = float(cosine_array.min())
        cosine_max = float(cosine_array.max())
    else:
        cosine_mean = cosine_min = cosine_max = None
    return _DitCacheStats(
        enabled=bool(enabled),
        threshold=None if threshold is None else float(threshold),
        warmup_steps=int(warmup_steps),
        max_consecutive=int(max_consecutive),
        executed_steps=int(executed_steps),
        skipped_steps=int(skipped_steps),
        cosine_mean=cosine_mean,
        cosine_min=cosine_min,
        cosine_max=cosine_max,
    )


def _providers(value: Sequence[str] | str | None, metadata: dict[str, Any] | None = None) -> list[str]:
    if value is not None:
        return parse_onnx_providers(value)
    if metadata is not None and bool(metadata.get("tensorrt_compatible", False)):
        return list(DEFAULT_TENSORRT_ONNX_PROVIDERS)
    return list(DEFAULT_DIFFUSION_ONNX_PROVIDERS)


def _coerce_seed_qpos(qpos_36: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos_36, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != QPOS_DIM:
        raise ValueError(f"Expected seed qpos_36 shape (T,{QPOS_DIM}), got {qpos.shape}")
    if qpos.shape[0] <= 0:
        raise ValueError("Seed qpos_36 is empty")
    if not np.isfinite(qpos).all():
        raise ValueError("Seed qpos_36 contains non-finite values")
    return qpos.astype(np.float32, copy=False)


def _coerce_audio_feature_chunks(
    value: np.ndarray | tuple[np.ndarray, np.ndarray],
    *,
    sequence_length: int,
    audio_dim: int,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, tuple):
        features_value, mask_value = value
    else:
        features_value = value
        mask_value = None
    features = np.asarray(features_value, dtype=np.float32)
    seq_len = int(sequence_length)
    dim = int(audio_dim)
    frames = int(num_frames)
    if frames % seq_len != 0:
        raise ValueError(f"num_frames={frames} must be a multiple of sequence_length={seq_len}")
    chunks = frames // seq_len
    if features.ndim == 2:
        expected = (frames, dim)
        if features.shape != expected:
            raise ValueError(f"audio_features must have shape {expected}, got {features.shape}")
        features = features.reshape(chunks, seq_len, dim)
    elif features.ndim == 3:
        expected = (chunks, seq_len, dim)
        if features.shape != expected:
            raise ValueError(f"audio_features must have shape {expected}, got {features.shape}")
    else:
        raise ValueError(
            "audio_features must have shape (num_frames, audio_dim) or "
            f"(num_chunks, sequence_length, audio_dim), got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("audio_features contains non-finite values")
    if mask_value is None:
        mask = np.ones((chunks, seq_len), dtype=bool)
    else:
        mask = np.asarray(mask_value, dtype=bool)
        if mask.ndim == 1:
            if mask.shape != (frames,):
                raise ValueError(f"audio mask must have shape ({frames},), got {mask.shape}")
            mask = mask.reshape(chunks, seq_len)
        elif mask.ndim == 2:
            if mask.shape != (chunks, seq_len):
                raise ValueError(f"audio mask must have shape ({chunks}, {seq_len}), got {mask.shape}")
        else:
            raise ValueError(f"audio mask must have shape (T,) or (N,T), got {mask.shape}")
    return features.astype(np.float32, copy=False), mask.astype(bool, copy=False)


def _coerce_human_motion_chunks(
    value: np.ndarray | tuple[np.ndarray, np.ndarray],
    *,
    sequence_length: int,
    human_motion_dim: int,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, tuple):
        features_value, mask_value = value
    else:
        features_value = value
        mask_value = None
    features = np.asarray(features_value, dtype=np.float32)
    if features.ndim == 3:
        if features.shape[-1] != 3:
            raise ValueError(f"Expected human joints shape (T,J,3), got {features.shape}")
        features = features.reshape(features.shape[0], -1)
    seq_len = int(sequence_length)
    dim = int(human_motion_dim)
    frames = int(num_frames)
    if frames % seq_len != 0:
        raise ValueError(f"num_frames={frames} must be a multiple of sequence_length={seq_len}")
    chunks = frames // seq_len
    if features.ndim == 2:
        expected = (frames, dim)
        if features.shape != expected:
            raise ValueError(f"human_motion must have shape {expected}, got {features.shape}")
        features = features.reshape(chunks, seq_len, dim)
    elif features.ndim == 3:
        expected = (chunks, seq_len, dim)
        if features.shape != expected:
            raise ValueError(f"human_motion must have shape {expected}, got {features.shape}")
    else:
        raise ValueError(
            "human_motion must have shape (num_frames, human_motion_dim), "
            "(num_chunks, sequence_length, human_motion_dim), or (num_frames, joints, 3); "
            f"got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("human_motion contains non-finite values")
    if mask_value is None:
        mask = np.ones((chunks, seq_len), dtype=bool)
    else:
        mask = np.asarray(mask_value, dtype=bool)
        if mask.ndim == 1:
            if mask.shape != (frames,):
                raise ValueError(f"human motion mask must have shape ({frames},), got {mask.shape}")
            mask = mask.reshape(chunks, seq_len)
        elif mask.ndim == 2:
            if mask.shape != (chunks, seq_len):
                raise ValueError(f"human motion mask must have shape ({chunks}, {seq_len}), got {mask.shape}")
        else:
            raise ValueError(f"human motion mask must have shape (T,) or (N,T), got {mask.shape}")
    return features.astype(np.float32, copy=False), mask.astype(bool, copy=False)


def _torch_device(value: str | torch.device | None) -> torch.device:
    if value is None or str(value) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")
    return device


def _sync_torch_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _has_condition_signal(inputs: dict[str, np.ndarray] | None, mask_key: str) -> bool:
    if inputs is None:
        return False
    mask = inputs.get(mask_key)
    return mask is not None and bool(np.asarray(mask, dtype=bool).any())


class OnnxDiffusionPlanner:
    """Diffusion-only runtime planner backed by an exported OMG ONNX step graph."""

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        metadata_path: str | Path | None = None,
        providers: Sequence[str] | str | None = None,
        text_encoder_model: str | None = None,
        torch_device: str | torch.device | None = "auto",
        seed: int = 0,
        compile_history_encoder: bool | None = None,
        tensorrt_fp16: bool = False,
        tensorrt_engine_cache_path: str | Path | None = None,
        dit_cache: bool = False,
        dit_cache_threshold: float = 0.995,
        dit_cache_warmup_steps: int = 4,
        dit_cache_max_consecutive: int = 2,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            raise RuntimeError("OnnxDiffusionPlanner requires onnxruntime") from exc

        self.onnx_path = Path(onnx_path).expanduser().resolve()
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"Diffusion ONNX model not found: {self.onnx_path}")
        self.metadata = load_export_metadata(self.onnx_path, metadata_path=metadata_path)
        if self.metadata.get("format") != "omg.denoiser_step":
            raise ValueError(f"Unsupported diffusion ONNX metadata format: {self.metadata.get('format')}")
        if self.metadata.get("diffusion_type") != "GuidedDiffusion":
            raise ValueError(f"Unsupported diffusion type: {self.metadata.get('diffusion_type')}")
        if self.metadata.get("diffusion_target") != "future":
            raise ValueError(f"Unsupported diffusion target: {self.metadata.get('diffusion_target')}")
        if self.metadata.get("objective") != "pred_x0":
            raise ValueError(f"Unsupported diffusion objective: {self.metadata.get('objective')}")

        self.providers = _providers(providers, self.metadata)
        validate_onnx_providers(self.providers, ort.get_available_providers())
        prepare_onnx_provider_runtime(self.providers)
        self.tensorrt_fp16 = bool(tensorrt_fp16)
        self.tensorrt_engine_cache_path = None
        self.dit_cache = bool(dit_cache)
        self.dit_cache_threshold = float(dit_cache_threshold)
        self.dit_cache_warmup_steps = int(dit_cache_warmup_steps)
        self.dit_cache_max_consecutive = int(dit_cache_max_consecutive)
        if self.dit_cache:
            if not np.isfinite(self.dit_cache_threshold) or not (-1.0 <= self.dit_cache_threshold <= 1.0):
                raise ValueError(
                    "dit_cache_threshold must be finite and within [-1, 1], "
                    f"got {self.dit_cache_threshold}"
                )
            if self.dit_cache_warmup_steps < 0:
                raise ValueError(f"dit_cache_warmup_steps must be non-negative, got {self.dit_cache_warmup_steps}")
            if self.dit_cache_max_consecutive < 0:
                raise ValueError(
                    f"dit_cache_max_consecutive must be non-negative, got {self.dit_cache_max_consecutive}"
                )
        provider_options: list[dict[str, str]] = []
        for provider in self.providers:
            options: dict[str, str] = {}
            if provider == "TensorrtExecutionProvider":
                if self.tensorrt_fp16:
                    options["trt_fp16_enable"] = "True"
                if tensorrt_engine_cache_path is not None:
                    cache_path = Path(tensorrt_engine_cache_path).expanduser().resolve()
                    cache_path.mkdir(parents=True, exist_ok=True)
                    self.tensorrt_engine_cache_path = cache_path
                    options["trt_engine_cache_enable"] = "True"
                    options["trt_engine_cache_path"] = str(cache_path)
                    options["trt_timing_cache_enable"] = "True"
                    options["trt_timing_cache_path"] = str(cache_path)
            provider_options.append(options)
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=self.providers,
            provider_options=provider_options,
        )
        validate_active_onnx_providers(self.providers, self.session.get_providers())
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.torch_device = _torch_device(torch_device)
        self.rng = np.random.default_rng(int(seed))
        self._onnx_infer_seconds = 0.0
        self._onnx_run_seconds: list[float] = []
        self._dit_cache_similarity_seconds = 0.0
        self._diffusion_update_seconds = 0.0

        self.sequence_length = int(self.metadata["sequence_length"])
        self.num_prev_states = int(self.metadata["num_prev_states"])
        self.feat_dim = int(self.metadata["feat_dim"])
        self.text_dim = int(self.metadata["text_dim"])
        self.text_max_length = int(self.metadata["text_max_length"])
        self.use_audio = bool(self.metadata.get("use_audio", False)) or bool({"audio_features", "audio_cond"} & self.input_names)
        self.audio_dim = int(self.metadata.get("audio_dim", 35))
        self.use_human_motion = bool(self.metadata.get("use_human_motion", False)) or bool({"human_motion", "human_motion_cond"} & self.input_names)
        self.human_motion_dim = int(self.metadata.get("human_motion_dim", 66))
        self.frame_cond_injection = str(self.metadata.get("frame_cond_injection", "sum_to_time"))
        if self.use_audio and self.audio_dim <= 0:
            raise ValueError(f"Audio ONNX metadata requires positive audio_dim, got {self.audio_dim}")
        if self.use_human_motion and self.human_motion_dim <= 0:
            raise ValueError(f"Human-reference ONNX metadata requires positive human_motion_dim, got {self.human_motion_dim}")
        self.batch_size = int(self.metadata.get("batch_size", 1))
        self.cfg_scale = float(self.metadata.get("cfg_scale", 1.0))
        self.ddim_eta = float(self.metadata.get("ddim_eta", 0.0))
        self.sample_timestep_map = np.asarray(self.metadata["sample_timestep_map"], dtype=np.int64)
        self.sample_alphas_cumprod = np.asarray(self.metadata["sample_alphas_cumprod"], dtype=np.float64)
        self.sample_alphas_cumprod_prev = np.asarray(self.metadata["sample_alphas_cumprod_prev"], dtype=np.float64)
        if not (
            self.sample_timestep_map.shape == self.sample_alphas_cumprod.shape == self.sample_alphas_cumprod_prev.shape
        ):
            raise ValueError("Diffusion metadata timestep arrays must have identical shape")

        stats_path = self.metadata.get("stats_path")
        kinematics_path = self.metadata.get("kinematics_path")
        rotation_representation = self.metadata.get("rotation_representation")
        if stats_path is None or kinematics_path is None or rotation_representation is None:
            raise ValueError(
                "Diffusion metadata must include stats_path, kinematics_path, and rotation_representation. "
                "Re-export the ONNX model with the current exporter."
            )
        self.representation = G1MotionRepresentation(
            stats_path=stats_path,
            kinematics_path=kinematics_path,
            num_prev_states=self.num_prev_states,
            canonical_frame_idx=int(self.metadata["canonical_frame_idx"]),
            feat_dim=self.feat_dim,
            sequence_length=self.sequence_length,
            rotation_representation=str(rotation_representation),
        ).to(self.torch_device)
        if compile_history_encoder is None:
            compile_history_encoder = self.torch_device.type == "cuda" and hasattr(torch, "compile")
        self.compile_history_encoder = bool(compile_history_encoder)
        self._compiled_history_encoder: (
            Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
        ) = None
        self._history_encoder_compile_failed = False
        self._history_encoder_compile_error: str | None = None
        encoder_model = text_encoder_model or self.metadata.get("text_encoder_model")
        self.text_encoder = None
        if encoder_model is not None and str(encoder_model) != "":
            self.text_encoder = FrozenT5TextEncoder(
                model_name=str(encoder_model),
                max_length=self.text_max_length,
                output_dim=self.text_dim,
            ).to(self.torch_device).eval()
        self._text_condition_cache: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}

    def _encode_text(self, text: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if self.text_encoder is None:
            context = np.zeros((1, self.text_max_length, self.text_dim), dtype=np.float32)
            mask = np.ones((1, self.text_max_length), dtype=bool)
            null_context = context.copy()
            null_mask = mask.copy()
            return (
                {"text_context": context, "text_mask": mask},
                {"text_context": null_context, "text_mask": null_mask},
            )
        has_text = torch.tensor([text != ""], dtype=torch.bool, device=self.torch_device)
        with torch.no_grad():
            cond = self.text_encoder([text], has_text=has_text, force_null_text=False, device=self.torch_device)
            null = self.text_encoder(
                [text],
                # Match MotionGenerator._conditions: CFG null uses an empty caption
                # while preserving whether this sample has a text condition.
                has_text=has_text,
                force_null_text=True,
                device=self.torch_device,
            )
        return (
            {
                "text_context": cond["context"].detach().cpu().numpy().astype(np.float32, copy=False),
                "text_mask": cond["mask"].detach().cpu().numpy().astype(bool, copy=False),
            },
            {
                "text_context": null["context"].detach().cpu().numpy().astype(np.float32, copy=False),
                "text_mask": null["mask"].detach().cpu().numpy().astype(bool, copy=False),
            },
        )

    def cache_text_conditions(self, text: str) -> None:
        text_key = str(text)
        self._text_condition_cache[text_key] = self._encode_text(text_key)

    def _text_conditions(self, text: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        text_key = str(text)
        cached = self._text_condition_cache.get(text_key)
        if cached is not None:
            return cached
        return self._encode_text(text_key)

    def _history_features_from_qpos_fast(
        self,
        prev_qpos_36: torch.Tensor,
        fps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        body_pos = self.representation.kinematics.forward_body_positions(prev_qpos_36)
        return self.representation.codec.prev_state_features_from_history(
            prev_qpos_36,
            body_pos,
            None,
            fps=fps,
        )

    def _encode_history_features(
        self,
        prev_qpos_36: torch.Tensor,
        fps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.compile_history_encoder
            and self._compiled_history_encoder is None
            and not self._history_encoder_compile_failed
        ):
            compile_fn = getattr(torch, "compile", None)
            if compile_fn is None:
                self._history_encoder_compile_failed = True
                self._history_encoder_compile_error = "torch.compile is unavailable"
            else:
                self._compiled_history_encoder = compile_fn(
                    self._history_features_from_qpos_fast,
                    mode="reduce-overhead",
                )

        if self._compiled_history_encoder is not None:
            try:
                return self._compiled_history_encoder(prev_qpos_36, fps)
            except Exception as exc:  # pragma: no cover - backend dependent fallback
                self._history_encoder_compile_failed = True
                self._history_encoder_compile_error = f"{type(exc).__name__}: {exc}"
                self._compiled_history_encoder = None
        return self._history_features_from_qpos_fast(prev_qpos_36, fps)

    def _history_from_qpos(
        self,
        seed_qpos_36: np.ndarray,
        fps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        if seed_qpos_36.shape[0] < self.num_prev_states:
            raise ValueError(
                f"Seed motion requires at least {self.num_prev_states} frames, got {seed_qpos_36.shape[0]}"
            )
        timing: dict[str, float] = {}

        _sync_torch_device(self.torch_device)
        started = time.perf_counter()
        prev = torch.from_numpy(seed_qpos_36[-self.num_prev_states :]).to(self.torch_device).unsqueeze(0)
        _sync_torch_device(self.torch_device)
        timing["history_qpos_to_device_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        body_pos = self.representation.kinematics.forward_body_positions(prev)
        _sync_torch_device(self.torch_device)
        timing["history_fk_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        fps_tensor = torch.tensor([float(fps)], dtype=prev.dtype, device=prev.device)
        history, canon_root_pos, canon_root_quat = self.representation.codec.prev_state_features_from_history(
            prev,
            body_pos,
            None,
            fps=fps_tensor,
        )
        _sync_torch_device(self.torch_device)
        timing["history_features_seconds"] = time.perf_counter() - started
        timing["history_total_seconds"] = sum(timing.values())
        return history, canon_root_pos, canon_root_quat, timing

    def _onnx_pred(
        self,
        x: np.ndarray,
        model_timestep: int | np.ndarray,
        valid_mask: np.ndarray,
        history_features: np.ndarray,
        text_inputs: dict[str, np.ndarray],
        audio_inputs: dict[str, np.ndarray] | None = None,
        human_motion_inputs: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if np.isscalar(model_timestep):
            timesteps = np.full((x.shape[0], x.shape[1]), int(model_timestep), dtype=np.int64)
        else:
            timesteps = np.asarray(model_timestep, dtype=np.int64)
            if timesteps.shape != (x.shape[0], x.shape[1]):
                raise ValueError(f"timesteps expected shape {(x.shape[0], x.shape[1])}, got {timesteps.shape}")
        feeds = {
            "x": x.astype(np.float32, copy=False),
            "timesteps": timesteps,
            "valid_mask": valid_mask.astype(bool, copy=False),
            "history_features": history_features.astype(np.float32, copy=False),
            "text_context": text_inputs["text_context"].astype(np.float32, copy=False),
            "text_mask": text_inputs["text_mask"].astype(bool, copy=False),
        }
        if self.use_audio:
            if audio_inputs is None:
                features = np.zeros((x.shape[0], x.shape[1], self.audio_dim), dtype=np.float32)
                mask = np.zeros((x.shape[0], x.shape[1]), dtype=bool)
            else:
                features = audio_inputs["audio_features"].astype(np.float32, copy=False)
                mask = audio_inputs["audio_mask"].astype(bool, copy=False)
            if "audio_features" in self.input_names:
                feeds["audio_features"] = features
            elif "audio_cond" in self.input_names:
                feeds["audio_cond"] = features
            else:
                raise ValueError("This audio diffusion ONNX does not expose audio_features/audio_cond input")
            if "audio_mask" in self.input_names:
                feeds["audio_mask"] = mask
            elif "has_audio" in self.input_names:
                feeds["has_audio"] = mask
        elif audio_inputs is not None:
            raise ValueError("audio_features were provided to a non-audio ONNX planner")
        if self.use_human_motion:
            if human_motion_inputs is None:
                features = np.zeros((x.shape[0], x.shape[1], self.human_motion_dim), dtype=np.float32)
                mask = np.zeros((x.shape[0], x.shape[1]), dtype=bool)
            else:
                features = human_motion_inputs["human_motion"].astype(np.float32, copy=False)
                mask = human_motion_inputs["human_motion_mask"].astype(bool, copy=False)
            if "human_motion" in self.input_names:
                feeds["human_motion"] = features
            elif "human_motion_cond" in self.input_names:
                feeds["human_motion_cond"] = features
            else:
                raise ValueError("This human-reference diffusion ONNX does not expose human_motion/human_motion_cond input")
            if "human_motion_mask" in self.input_names:
                feeds["human_motion_mask"] = mask
            elif "has_human_motion" in self.input_names:
                feeds["has_human_motion"] = mask
        elif human_motion_inputs is not None:
            raise ValueError("human_motion was provided to a non-human-reference ONNX planner")
        started = time.perf_counter()
        pred = self.session.run(["pred"], feeds)[0]
        elapsed = time.perf_counter() - started
        self._onnx_infer_seconds += elapsed
        self._onnx_run_seconds.append(elapsed)
        return pred.astype(np.float32, copy=False)

    def _onnx_single_pred(
        self,
        x: np.ndarray,
        model_timestep: int,
        valid_mask: np.ndarray,
        history_features: np.ndarray,
        text_inputs: dict[str, np.ndarray],
        audio_inputs: dict[str, np.ndarray] | None = None,
        human_motion_inputs: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        repeats = int(self.batch_size)
        if repeats <= 1:
            return self._onnx_pred(
                x,
                model_timestep,
                valid_mask,
                history_features,
                text_inputs,
                audio_inputs,
                human_motion_inputs,
            )
        batch_text = {
            "text_context": np.repeat(text_inputs["text_context"], repeats, axis=0),
            "text_mask": np.repeat(text_inputs["text_mask"], repeats, axis=0),
        }
        batch_audio = None
        if audio_inputs is not None:
            batch_audio = {
                "audio_features": np.repeat(audio_inputs["audio_features"], repeats, axis=0),
                "audio_mask": np.repeat(audio_inputs["audio_mask"], repeats, axis=0),
            }
        batch_human_motion = None
        if human_motion_inputs is not None:
            batch_human_motion = {
                "human_motion": np.repeat(human_motion_inputs["human_motion"], repeats, axis=0),
                "human_motion_mask": np.repeat(human_motion_inputs["human_motion_mask"], repeats, axis=0),
            }
        return self._onnx_pred(
            np.repeat(x, repeats, axis=0),
            model_timestep,
            np.repeat(valid_mask, repeats, axis=0),
            np.repeat(history_features, repeats, axis=0),
            batch_text,
            batch_audio,
            batch_human_motion,
        )[:1]

    def _onnx_cfg_pred(
        self,
        x: np.ndarray,
        model_timestep: int,
        valid_mask: np.ndarray,
        history_features: np.ndarray,
        cond_text: dict[str, np.ndarray],
        null_text: dict[str, np.ndarray],
        cfg_scale: float,
        cfg_text_scale: float | None = None,
        cfg_audio_scale: float | None = None,
        cfg_human_scale: float | None = None,
        cond_audio: dict[str, np.ndarray] | None = None,
        null_audio: dict[str, np.ndarray] | None = None,
        cond_human_motion: dict[str, np.ndarray] | None = None,
        null_human_motion: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        separate_cfg = any(
            value is not None
            for value in (cfg_text_scale, cfg_audio_scale, cfg_human_scale)
        )
        if not separate_cfg:
            return self._onnx_joint_cfg_pred(
                x,
                model_timestep,
                valid_mask,
                history_features,
                cond_text,
                null_text,
                cfg_scale,
                cond_audio=cond_audio,
                null_audio=null_audio,
                cond_human_motion=cond_human_motion,
                null_human_motion=null_human_motion,
            )

        raw_null = self._onnx_single_pred(
            x,
            model_timestep,
            valid_mask,
            history_features,
            null_text,
            null_audio,
            null_human_motion,
        )
        pred = raw_null.copy()
        if cfg_text_scale is not None and float(cfg_text_scale) != 0.0:
            raw_text = self._onnx_single_pred(
                x,
                model_timestep,
                valid_mask,
                history_features,
                cond_text,
                null_audio,
                null_human_motion,
            )
            pred = pred + float(cfg_text_scale) * (raw_text - raw_null)
        if cfg_audio_scale is not None and float(cfg_audio_scale) != 0.0 and _has_condition_signal(cond_audio, "audio_mask"):
            raw_audio = self._onnx_single_pred(
                x,
                model_timestep,
                valid_mask,
                history_features,
                null_text,
                cond_audio,
                null_human_motion,
            )
            pred = pred + float(cfg_audio_scale) * (raw_audio - raw_null)
        if (
            cfg_human_scale is not None
            and float(cfg_human_scale) != 0.0
            and _has_condition_signal(cond_human_motion, "human_motion_mask")
        ):
            raw_human = self._onnx_single_pred(
                x,
                model_timestep,
                valid_mask,
                history_features,
                null_text,
                null_audio,
                cond_human_motion,
            )
            pred = pred + float(cfg_human_scale) * (raw_human - raw_null)
        return pred.astype(np.float32, copy=False)

    def _onnx_joint_cfg_pred(
        self,
        x: np.ndarray,
        model_timestep: int,
        valid_mask: np.ndarray,
        history_features: np.ndarray,
        cond_text: dict[str, np.ndarray],
        null_text: dict[str, np.ndarray],
        cfg_scale: float,
        cond_audio: dict[str, np.ndarray] | None = None,
        null_audio: dict[str, np.ndarray] | None = None,
        cond_human_motion: dict[str, np.ndarray] | None = None,
        null_human_motion: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if float(cfg_scale) == 1.0 and self.batch_size == 1:
            return self._onnx_pred(
                x, model_timestep, valid_mask, history_features, cond_text, cond_audio, cond_human_motion
            )
        if self.batch_size < 2:
            raw = self._onnx_pred(
                x, model_timestep, valid_mask, history_features, cond_text, cond_audio, cond_human_motion
            )
            raw_null = self._onnx_pred(
                x, model_timestep, valid_mask, history_features, null_text, null_audio, null_human_motion
            )
            return raw_null + float(cfg_scale) * (raw - raw_null)

        repeats = int(self.batch_size)
        batch_x = np.repeat(x, repeats, axis=0)
        batch_valid = np.repeat(valid_mask, repeats, axis=0)
        batch_history = np.repeat(history_features, repeats, axis=0)
        batch_audio = None
        if self.use_audio:
            if cond_audio is None or null_audio is None:
                raise ValueError("Audio CFG requires cond_audio and null_audio")
            batch_audio = {
                "audio_features": np.concatenate(
                    [
                        cond_audio["audio_features"],
                        np.repeat(null_audio["audio_features"], repeats - 1, axis=0),
                    ],
                    axis=0,
                ),
                "audio_mask": np.concatenate(
                    [
                        cond_audio["audio_mask"],
                        np.repeat(null_audio["audio_mask"], repeats - 1, axis=0),
                    ],
                    axis=0,
                ),
            }
        batch_human_motion = None
        if self.use_human_motion:
            if cond_human_motion is None or null_human_motion is None:
                raise ValueError("Human-reference CFG requires cond_human_motion and null_human_motion")
            batch_human_motion = {
                "human_motion": np.concatenate(
                    [
                        cond_human_motion["human_motion"],
                        np.repeat(null_human_motion["human_motion"], repeats - 1, axis=0),
                    ],
                    axis=0,
                ),
                "human_motion_mask": np.concatenate(
                    [
                        cond_human_motion["human_motion_mask"],
                        np.repeat(null_human_motion["human_motion_mask"], repeats - 1, axis=0),
                    ],
                    axis=0,
                ),
            }
        if float(cfg_scale) == 1.0:
            batch_text = {
                "text_context": np.repeat(cond_text["text_context"], repeats, axis=0),
                "text_mask": np.repeat(cond_text["text_mask"], repeats, axis=0),
            }
            if self.use_audio and cond_audio is not None:
                batch_audio = {
                    "audio_features": np.repeat(cond_audio["audio_features"], repeats, axis=0),
                    "audio_mask": np.repeat(cond_audio["audio_mask"], repeats, axis=0),
                }
            if self.use_human_motion and cond_human_motion is not None:
                batch_human_motion = {
                    "human_motion": np.repeat(cond_human_motion["human_motion"], repeats, axis=0),
                    "human_motion_mask": np.repeat(cond_human_motion["human_motion_mask"], repeats, axis=0),
                }
            return self._onnx_pred(
                batch_x, model_timestep, batch_valid, batch_history, batch_text, batch_audio, batch_human_motion
            )[:1]
        batch_text = {
            "text_context": np.concatenate(
                [
                    cond_text["text_context"],
                    np.repeat(null_text["text_context"], repeats - 1, axis=0),
                ],
                axis=0,
            ),
            "text_mask": np.concatenate(
                [
                    cond_text["text_mask"],
                    np.repeat(null_text["text_mask"], repeats - 1, axis=0),
                ],
                axis=0,
            ),
        }
        raw_all = self._onnx_pred(
            batch_x, model_timestep, batch_valid, batch_history, batch_text, batch_audio, batch_human_motion
        )
        raw = raw_all[:1]
        raw_null = raw_all[1:2]
        return raw_null + float(cfg_scale) * (raw - raw_null)

    def _sample_chunk(
        self,
        history_features: torch.Tensor,
        canonical_root_pos: torch.Tensor,
        canonical_root_quat: torch.Tensor,
        cond_text: dict[str, np.ndarray],
        null_text: dict[str, np.ndarray],
        cfg_scale: float,
        cfg_text_scale: float | None = None,
        cfg_audio_scale: float | None = None,
        cfg_human_scale: float | None = None,
        cond_audio: dict[str, np.ndarray] | None = None,
        null_audio: dict[str, np.ndarray] | None = None,
        cond_human_motion: dict[str, np.ndarray] | None = None,
        null_human_motion: dict[str, np.ndarray] | None = None,
        continuation_init: _ContinuationInit | None = None,
    ) -> tuple[np.ndarray, DiffusionContinuationState, _DitCacheStats]:
        x = self.rng.standard_normal((1, self.sequence_length, self.feat_dim)).astype(np.float32)
        max_step = self.sample_timestep_map.shape[0] - 1
        if continuation_init is not None and not (0 <= int(continuation_init.start_step) <= max_step):
            raise ValueError(f"Continuation start step {continuation_init.start_step} is outside the sampling schedule")
        valid_mask = np.ones((1, self.sequence_length), dtype=bool)
        history_np = history_features.detach().cpu().numpy().astype(np.float32, copy=False)
        latent_states = np.zeros(
            (self.sample_timestep_map.shape[0], self.sequence_length, self.feat_dim),
            dtype=np.float32,
        )
        valid_steps = np.zeros((self.sample_timestep_map.shape[0],), dtype=bool)
        dit_cache_enabled = self.dit_cache and continuation_init is None
        previous_pred_x0: np.ndarray | None = None
        last_similarity = 0.0
        consecutive_skips = 0
        executed_steps = 0
        skipped_steps = 0
        cache_cosines: list[float] = []

        for step in range(max_step, -1, -1):
            if continuation_init is not None and step <= continuation_init.start_step:
                x = self._apply_continuation_prefix(x, step, continuation_init)
            latent_states[step] = x[0]
            valid_steps[step] = True
            model_t = int(self.sample_timestep_map[step])
            should_skip = (
                dit_cache_enabled
                and previous_pred_x0 is not None
                and executed_steps >= self.dit_cache_warmup_steps
                and consecutive_skips < self.dit_cache_max_consecutive
                and last_similarity >= self.dit_cache_threshold
            )
            if should_skip:
                pred_x0 = previous_pred_x0
                consecutive_skips += 1
                skipped_steps += 1
            else:
                pred_x0 = self._onnx_cfg_pred(
                    x,
                    model_t,
                    valid_mask,
                    history_np,
                    cond_text,
                    null_text,
                    cfg_scale,
                    cfg_text_scale=cfg_text_scale,
                    cfg_audio_scale=cfg_audio_scale,
                    cfg_human_scale=cfg_human_scale,
                    cond_audio=cond_audio,
                    null_audio=null_audio,
                    cond_human_motion=cond_human_motion,
                    null_human_motion=null_human_motion,
                )
                if dit_cache_enabled and previous_pred_x0 is not None:
                    similarity_started = time.perf_counter()
                    last_similarity = _cosine_similarity_flat(pred_x0, previous_pred_x0)
                    self._dit_cache_similarity_seconds += time.perf_counter() - similarity_started
                    cache_cosines.append(last_similarity)
                previous_pred_x0 = pred_x0
                consecutive_skips = 0
                executed_steps += 1

            update_started = time.perf_counter()
            alpha_bar = float(self.sample_alphas_cumprod[step])
            alpha_bar_prev = float(self.sample_alphas_cumprod_prev[step])
            eps = (x - np.sqrt(alpha_bar) * pred_x0) / max(np.sqrt(1.0 - alpha_bar), 1e-8)
            sigma = (
                self.ddim_eta
                * np.sqrt(max((1.0 - alpha_bar_prev) / max(1.0 - alpha_bar, 1e-8), 0.0))
                * np.sqrt(max(1.0 - alpha_bar / max(alpha_bar_prev, 1e-8), 0.0))
            )
            mean_pred = (
                np.sqrt(alpha_bar_prev) * pred_x0
                + np.sqrt(max(1.0 - alpha_bar_prev - sigma * sigma, 0.0)) * eps
            )
            if step > 0:
                x = mean_pred + sigma * self.rng.standard_normal(x.shape).astype(np.float32)
            else:
                x = mean_pred
            if continuation_init is not None and step <= continuation_init.start_step:
                if step > 0:
                    x = self._apply_continuation_prefix(x, step - 1, continuation_init)
                else:
                    x[:, : continuation_init.overlap_frames] = continuation_init.x0
            x = np.where(valid_mask[..., None], x, 0.0).astype(np.float32, copy=False)
            self._diffusion_update_seconds += time.perf_counter() - update_started
        continuation_state = DiffusionContinuationState(
            latent_states=latent_states,
            valid_steps=valid_steps,
            sample_timestep_map=self.sample_timestep_map.astype(np.int64, copy=True),
            canonical_root_pos=canonical_root_pos.detach().cpu().numpy().astype(np.float32, copy=True),
            canonical_root_quat=canonical_root_quat.detach().cpu().numpy().astype(np.float32, copy=True),
        )
        cache_stats = _dit_cache_stats(
            enabled=dit_cache_enabled,
            threshold=self.dit_cache_threshold if dit_cache_enabled else None,
            warmup_steps=self.dit_cache_warmup_steps,
            max_consecutive=self.dit_cache_max_consecutive,
            executed_steps=executed_steps,
            skipped_steps=skipped_steps,
            cosines=cache_cosines,
        )
        return x.astype(np.float32, copy=False), continuation_state, cache_stats

    def _apply_continuation_prefix(
        self,
        x: np.ndarray,
        step: int,
        continuation_init: _ContinuationInit,
    ) -> np.ndarray:
        if continuation_init.x0.shape != continuation_init.noise.shape:
            raise ValueError("Continuation x0/noise shapes must match")
        if continuation_init.x0.shape != (1, continuation_init.overlap_frames, self.feat_dim):
            raise ValueError(
                "Continuation prefix expected shape "
                f"{(1, continuation_init.overlap_frames, self.feat_dim)}, got {continuation_init.x0.shape}"
            )
        alpha_bar = float(self.sample_alphas_cumprod[int(step)])
        known = (
            np.sqrt(alpha_bar) * continuation_init.x0
            + np.sqrt(max(1.0 - alpha_bar, 0.0)) * continuation_init.noise
        )
        x[:, : continuation_init.overlap_frames] = known.astype(np.float32, copy=False)
        return x

    def _build_continuation_init(
        self,
        *,
        previous_plan: MotionPlan,
        previous_plan_cursor_frames: int,
        continuation_steps: int,
        canonical_root_pos: torch.Tensor,
        canonical_root_quat: torch.Tensor,
    ) -> _ContinuationInit:
        start_step = _continuation_start_step(self.sample_timestep_map.shape[0], continuation_steps)
        cursor = int(previous_plan_cursor_frames)
        previous_qpos = _coerce_seed_qpos(previous_plan.qpos_36)
        if cursor < 0 or cursor >= previous_qpos.shape[0]:
            raise ValueError(
                f"previous_plan_cursor_frames must be in [0, {previous_qpos.shape[0] - 1}], got {cursor}"
            )
        overlap_frames = min(self.sequence_length, previous_qpos.shape[0] - cursor)
        previous_world_qpos = torch.from_numpy(previous_qpos[cursor : cursor + overlap_frames]).to(
            self.torch_device
        ).unsqueeze(0)
        previous_body_pos = self.representation.kinematics.forward_body_positions(previous_world_qpos)
        recanonicalized = self.representation.codec.canonicalize(
            previous_world_qpos,
            previous_body_pos,
            None,
            anchor_root_pos=canonical_root_pos,
            anchor_root_quat=canonical_root_quat,
            fps=torch.tensor([float(previous_plan.fps)], dtype=previous_world_qpos.dtype, device=self.torch_device),
        )
        continuation_features = self.representation.codec.assemble_features(recanonicalized)
        continuation_latent = self.representation.normalize_features(continuation_features)
        x0 = continuation_latent.detach().cpu().numpy().astype(np.float32, copy=False)
        noise = self.rng.standard_normal(x0.shape).astype(np.float32)
        return _ContinuationInit(
            x0=x0,
            noise=noise,
            start_step=start_step,
            overlap_frames=overlap_frames,
        )

    def plan(
        self,
        *,
        seed_qpos_36: np.ndarray,
        text: str,
        fps: float,
        num_frames: int,
        cfg_scale: float | None = None,
        cfg_text_scale: float | None = None,
        cfg_audio_scale: float | None = None,
        cfg_human_scale: float | None = None,
        previous_plan: MotionPlan | None = None,
        previous_plan_cursor_frames: int = 0,
        continuation_steps: int = 0,
        audio_features: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        human_motion: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
    ) -> MotionPlan:
        if int(num_frames) <= 0:
            raise ValueError("num_frames must be positive")
        if int(num_frames) % self.sequence_length != 0:
            raise ValueError(
                f"ONNX diffusion planner currently requires num_frames to be a multiple of sequence_length={self.sequence_length}"
            )
        continuation_steps_int = int(continuation_steps)
        if continuation_steps_int < 0:
            raise ValueError(f"continuation_steps must be non-negative, got {continuation_steps_int}")
        if continuation_steps_int > self.sample_timestep_map.shape[0]:
            raise ValueError(
                f"continuation_steps must be at most {self.sample_timestep_map.shape[0]}, got {continuation_steps_int}"
            )
        qpos_seed = _coerce_seed_qpos(seed_qpos_36)
        if not np.isfinite(float(fps)) or float(fps) <= 0.0:
            raise ValueError(f"fps must be positive and finite, got {fps}")
        scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        separate_cfg = any(
            value is not None
            for value in (cfg_text_scale, cfg_audio_scale, cfg_human_scale)
        )
        resolved_cfg_text_scale = None if not separate_cfg else float(cfg_text_scale or 0.0)
        resolved_cfg_audio_scale = None if not separate_cfg else float(cfg_audio_scale or 0.0)
        resolved_cfg_human_scale = None if not separate_cfg else float(cfg_human_scale or 0.0)
        plan_started = time.perf_counter()
        audio_prepare_seconds = 0.0
        audio_chunks = audio_masks = None
        if self.use_audio:
            if audio_features is None:
                audio_chunks = np.zeros(
                    (int(num_frames) // self.sequence_length, self.sequence_length, self.audio_dim),
                    dtype=np.float32,
                )
                audio_masks = np.zeros((int(num_frames) // self.sequence_length, self.sequence_length), dtype=bool)
            else:
                audio_started = time.perf_counter()
                audio_chunks, audio_masks = _coerce_audio_feature_chunks(
                    audio_features,
                    sequence_length=self.sequence_length,
                    audio_dim=self.audio_dim,
                    num_frames=int(num_frames),
                )
                audio_prepare_seconds = time.perf_counter() - audio_started
        elif audio_features is not None:
            raise ValueError("audio_features were provided to a non-audio ONNX planner")
        human_motion_prepare_seconds = 0.0
        human_motion_chunks = human_motion_masks = None
        if self.use_human_motion:
            if human_motion is None:
                human_motion_chunks = np.zeros(
                    (int(num_frames) // self.sequence_length, self.sequence_length, self.human_motion_dim),
                    dtype=np.float32,
                )
                human_motion_masks = np.zeros((int(num_frames) // self.sequence_length, self.sequence_length), dtype=bool)
            else:
                human_motion_started = time.perf_counter()
                human_motion_chunks, human_motion_masks = _coerce_human_motion_chunks(
                    human_motion,
                    sequence_length=self.sequence_length,
                    human_motion_dim=self.human_motion_dim,
                    num_frames=int(num_frames),
                )
                human_motion_prepare_seconds = time.perf_counter() - human_motion_started
        elif human_motion is not None:
            raise ValueError("human_motion was provided to a non-human-reference ONNX planner")
        text_started = time.perf_counter()
        cond_text, null_text = self._text_conditions(str(text))
        text_encode_seconds = time.perf_counter() - text_started
        self._onnx_infer_seconds = 0.0
        self._onnx_run_seconds = []
        self._dit_cache_similarity_seconds = 0.0
        self._diffusion_update_seconds = 0.0
        decode_seconds = 0.0
        decode_to_device_seconds = 0.0
        decode_representation_seconds = 0.0
        decode_compose_qpos_seconds = 0.0
        decode_denormalize_seconds = 0.0
        decode_cpu_copy_seconds = 0.0
        sampling_seconds = 0.0
        rollout_history_seconds = 0.0
        rollout_history_fk_seconds = 0.0
        rollout_history_features_seconds = 0.0
        continuation_seconds = 0.0
        continuation_applied = False
        continuation_start_step = None
        continuation_overlap_frames = 0

        with torch.no_grad():
            history, canon_root_pos, canon_root_quat, history_timing = self._history_from_qpos(qpos_seed, float(fps))
            canonicalization_seconds = history_timing["history_total_seconds"]
            continuation_init = None
            if previous_plan is not None and continuation_steps_int > 0:
                continuation_started = time.perf_counter()
                continuation_init = self._build_continuation_init(
                    previous_plan=previous_plan,
                    previous_plan_cursor_frames=int(previous_plan_cursor_frames),
                    continuation_steps=continuation_steps_int,
                    canonical_root_pos=canon_root_pos,
                    canonical_root_quat=canon_root_quat,
                )
                continuation_seconds = time.perf_counter() - continuation_started
                continuation_applied = True
                continuation_start_step = int(continuation_init.start_step)
                continuation_overlap_frames = int(continuation_init.overlap_frames)
            frames_left = int(num_frames)
            qpos_chunks: list[torch.Tensor] = []
            feature_chunks: list[torch.Tensor] = []
            fps_tensor = torch.tensor([float(fps)], dtype=torch.float32, device=self.torch_device)
            plan_continuation_state: DiffusionContinuationState | None = None
            dit_cache_chunk_stats: list[_DitCacheStats] = []
            chunk_index = 0
            while frames_left > 0:
                sampling_started = time.perf_counter()
                cond_audio = null_audio = None
                if self.use_audio:
                    assert audio_chunks is not None and audio_masks is not None
                    chunk_audio = audio_chunks[chunk_index : chunk_index + 1]
                    chunk_audio_mask = audio_masks[chunk_index : chunk_index + 1]
                    cond_audio = {"audio_features": chunk_audio, "audio_mask": chunk_audio_mask}
                    null_audio = {
                        "audio_features": np.zeros_like(chunk_audio),
                        "audio_mask": np.zeros_like(chunk_audio_mask),
                    }
                cond_human_motion = null_human_motion = None
                if self.use_human_motion:
                    assert human_motion_chunks is not None and human_motion_masks is not None
                    chunk_human_motion = human_motion_chunks[chunk_index : chunk_index + 1]
                    chunk_human_motion_mask = human_motion_masks[chunk_index : chunk_index + 1]
                    cond_human_motion = {"human_motion": chunk_human_motion, "human_motion_mask": chunk_human_motion_mask}
                    null_human_motion = {
                        "human_motion": np.zeros_like(chunk_human_motion),
                        "human_motion_mask": np.zeros_like(chunk_human_motion_mask),
                    }
                future_norm_np, chunk_continuation_state, chunk_dit_cache_stats = self._sample_chunk(
                    history,
                    canon_root_pos,
                    canon_root_quat,
                    cond_text,
                    null_text,
                    scale,
                    cfg_text_scale=resolved_cfg_text_scale,
                    cfg_audio_scale=resolved_cfg_audio_scale,
                    cfg_human_scale=resolved_cfg_human_scale,
                    cond_audio=cond_audio,
                    null_audio=null_audio,
                    cond_human_motion=cond_human_motion,
                    null_human_motion=null_human_motion,
                    continuation_init=continuation_init if chunk_index == 0 else None,
                )
                sampling_seconds += time.perf_counter() - sampling_started
                dit_cache_chunk_stats.append(chunk_dit_cache_stats)
                if chunk_index == 0:
                    plan_continuation_state = chunk_continuation_state
                decode_started = time.perf_counter()
                part_started = time.perf_counter()
                future_norm = torch.from_numpy(future_norm_np).to(self.torch_device)
                _sync_torch_device(self.torch_device)
                decode_to_device_seconds += time.perf_counter() - part_started

                part_started = time.perf_counter()
                decoded = self.representation.decode(future_norm)
                _sync_torch_device(self.torch_device)
                decode_representation_seconds += time.perf_counter() - part_started

                part_started = time.perf_counter()
                qpos = self.representation.compose_qpos_36(decoded, canon_root_pos, canon_root_quat)
                _sync_torch_device(self.torch_device)
                decode_compose_qpos_seconds += time.perf_counter() - part_started

                part_started = time.perf_counter()
                features = self.representation.denormalize_features(future_norm)
                _sync_torch_device(self.torch_device)
                decode_denormalize_seconds += time.perf_counter() - part_started
                decode_seconds += time.perf_counter() - decode_started

                part_started = time.perf_counter()
                qpos_chunks.append(qpos.detach().cpu())
                feature_chunks.append(features.detach().cpu())
                decode_cpu_copy_seconds += time.perf_counter() - part_started

                history_update_started = time.perf_counter()
                prev_qpos = qpos[:, -self.num_prev_states :]
                history_fk_started = time.perf_counter()
                prev_fk = self.representation.kinematics.forward_kinematics(prev_qpos)
                _sync_torch_device(self.torch_device)
                rollout_history_fk_seconds += time.perf_counter() - history_fk_started
                history_features_started = time.perf_counter()
                history, canon_root_pos, canon_root_quat = self.representation.codec.prev_state_features_from_history(
                    prev_qpos,
                    prev_fk["body_pos_w"],
                    prev_fk["body_quat_w"],
                    fps=fps_tensor,
                )
                _sync_torch_device(self.torch_device)
                rollout_history_features_seconds += time.perf_counter() - history_features_started
                rollout_history_seconds += time.perf_counter() - history_update_started
                frames_left -= self.sequence_length
                chunk_index += 1
                if frames_left > 0:
                    prev_qpos = qpos[:, -self.num_prev_states :]
                    history, canon_root_pos, canon_root_quat = self._encode_history_features(prev_qpos, fps_tensor)

        concat_started = time.perf_counter()
        qpos_out = torch.cat(qpos_chunks, dim=1)[0].numpy().astype(np.float32, copy=False)
        features_out = torch.cat(feature_chunks, dim=1)[0].numpy().astype(np.float32, copy=False)
        concat_seconds = time.perf_counter() - concat_started
        total_seconds = time.perf_counter() - plan_started
        dit_cache_metadata_chunks = [stats.as_metadata() for stats in dit_cache_chunk_stats]
        dit_cache_executed_steps = sum(stats.executed_steps for stats in dit_cache_chunk_stats)
        dit_cache_skipped_steps = sum(stats.skipped_steps for stats in dit_cache_chunk_stats)
        onnx_run_ms = np.asarray(self._onnx_run_seconds, dtype=np.float64) * 1000.0
        onnx_run_count = int(onnx_run_ms.shape[0])
        sampling_non_onnx_seconds = max(
            sampling_seconds
            - self._onnx_infer_seconds
            - self._diffusion_update_seconds
            - self._dit_cache_similarity_seconds,
            0.0,
        )
        timing_ms = {
            "text_encode_ms": text_encode_seconds * 1000.0,
            "audio_prepare_ms": audio_prepare_seconds * 1000.0,
            "human_motion_prepare_ms": human_motion_prepare_seconds * 1000.0,
            "downsample_ms": 0.0,
            "canonicalization_ms": canonicalization_seconds * 1000.0,
            "history_qpos_to_device_ms": history_timing["history_qpos_to_device_seconds"] * 1000.0,
            "history_fk_ms": history_timing["history_fk_seconds"] * 1000.0,
            "history_features_ms": history_timing["history_features_seconds"] * 1000.0,
            "continuation_ms": continuation_seconds * 1000.0,
            "diffusion_infer_ms": self._onnx_infer_seconds * 1000.0,
            "diffusion_sampling_total_ms": sampling_seconds * 1000.0,
            "diffusion_sampling_non_onnx_ms": sampling_non_onnx_seconds * 1000.0,
            "diffusion_update_ms": self._diffusion_update_seconds * 1000.0,
            "dit_cache_similarity_ms": self._dit_cache_similarity_seconds * 1000.0,
            "onnx_run_count": onnx_run_count,
            "onnx_run_mean_ms": float(onnx_run_ms.mean()) if onnx_run_count > 0 else 0.0,
            "onnx_run_min_ms": float(onnx_run_ms.min()) if onnx_run_count > 0 else 0.0,
            "onnx_run_max_ms": float(onnx_run_ms.max()) if onnx_run_count > 0 else 0.0,
            "ik_ms": 0.0,
            "interpolation_ms": decode_seconds * 1000.0,
            "decode_to_device_ms": decode_to_device_seconds * 1000.0,
            "decode_representation_ms": decode_representation_seconds * 1000.0,
            "decode_compose_qpos_ms": decode_compose_qpos_seconds * 1000.0,
            "decode_denormalize_ms": decode_denormalize_seconds * 1000.0,
            "decode_cpu_copy_ms": decode_cpu_copy_seconds * 1000.0,
            "rollout_history_ms": rollout_history_seconds * 1000.0,
            "rollout_history_fk_ms": rollout_history_fk_seconds * 1000.0,
            "rollout_history_features_ms": rollout_history_features_seconds * 1000.0,
            "concat_output_ms": concat_seconds * 1000.0,
            "total_ms": total_seconds * 1000.0,
        }
        metadata = {
            "diffusion_onnx": str(self.onnx_path),
            "providers": self.providers,
            "active_providers": list(self.session.get_providers()),
            "batch_size": self.batch_size,
            "tensorrt_fp16": self.tensorrt_fp16,
            "tensorrt_engine_cache_path": None if self.tensorrt_engine_cache_path is None else str(self.tensorrt_engine_cache_path),
            "dit_cache": self.dit_cache,
            "dit_cache_threshold": self.dit_cache_threshold,
            "dit_cache_warmup_steps": self.dit_cache_warmup_steps,
            "dit_cache_max_consecutive": self.dit_cache_max_consecutive,
            "dit_cache_executed_steps": int(dit_cache_executed_steps),
            "dit_cache_skipped_steps": int(dit_cache_skipped_steps),
            "dit_cache_chunks": dit_cache_metadata_chunks,
            "num_frames": int(num_frames),
            "fps": float(fps),
            "cfg_scale": scale,
            "cfg_text_scale": resolved_cfg_text_scale,
            "cfg_audio_scale": resolved_cfg_audio_scale,
            "cfg_human_scale": resolved_cfg_human_scale,
            "sequence_length": self.sequence_length,
            "use_audio": self.use_audio,
            "audio_dim": self.audio_dim if self.use_audio else None,
            "audio_conditioned": bool(audio_features is not None),
            "audio_chunks": None if audio_chunks is None else int(audio_chunks.shape[0]),
            "use_human_motion": self.use_human_motion,
            "human_motion_dim": self.human_motion_dim if self.use_human_motion else None,
            "human_motion_conditioned": bool(human_motion is not None),
            "human_motion_chunks": None if human_motion_chunks is None else int(human_motion_chunks.shape[0]),
            "continuation_steps": continuation_steps_int,
            "continuation_applied": continuation_applied,
            "continuation_start_step": continuation_start_step,
            "continuation_overlap_frames": continuation_overlap_frames,
            "previous_plan_cursor_frames": int(previous_plan_cursor_frames) if previous_plan is not None else None,
            "history_encoder": "compiled_pos_only"
            if self._compiled_history_encoder is not None
            else "pos_only",
            "history_encoder_compile_failed": bool(self._history_encoder_compile_failed),
            "history_encoder_compile_error": self._history_encoder_compile_error,
            "timing_ms": timing_ms,
        }
        return MotionPlan(
            qpos_36=qpos_out,
            motion_features=features_out,
            fps=float(fps),
            metadata=metadata,
            continuation_state=plan_continuation_state,
        )


def save_motion_plan(plan: MotionPlan, output_dir: str | Path, *, extra_metadata: dict[str, Any] | None = None) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "qpos_36.npy", plan.qpos_36.astype(np.float32, copy=False))
    np.savez_compressed(
        out / "reference_motion.npz",
        qpos_36=plan.qpos_36.astype(np.float32, copy=False),
        motion_features=plan.motion_features.astype(np.float32, copy=False),
        fps=np.asarray([float(plan.fps)], dtype=np.float32),
    )
    metadata = dict(plan.metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
