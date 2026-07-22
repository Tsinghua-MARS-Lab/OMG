from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from omg.generation.architecture import MODEL_ARCHITECTURE_KEY, build_model_architecture_contract


FRAME_COND_INJECTION_MODES = {
    "sum_to_time",
    "separate_to_h",
    "per_layer_film",
    "control_local_attn",
}
SEPARATE_FRAME_COND_MODES = {"separate_to_h", "per_layer_film", "control_local_attn"}
MASKED_FRAME_COND_MODES = {"per_layer_film", "control_local_attn"}


def _is_config(value: Any) -> bool:
    return isinstance(value, (dict, DictConfig))


def _instantiate_if_config(value: Any) -> Any:
    return instantiate(value) if _is_config(value) else value


class MotionGenerator(pl.LightningModule):
    def __init__(
        self,
        representation,
        denoiser,
        diffusion,
        loss,
        text_encoder: dict | None = None,
        diffusion_target: str = "future",
        condition_dim: int | None = None,
        text_mask_prob: float = 0.1,
        history_mask_prob: float = 0.1,
        use_audio: bool = False,
        audio_dim: int = 35,
        audio_mask_prob: float = 0.1,
        use_human_motion: bool = False,
        human_motion_dim: int = 66,
        human_motion_mask_prob: float = 0.1,
        frame_cond_injection: str = "per_layer_film",
        optimizer: dict | None = None,
        scheduler: dict | None = None,
        log_loss_term_grad_norms: bool = False,
        loss_term_grad_norm_every_n_steps: int = 100,
        loss_term_grad_norm_norm_type: float = 2.0,
    ):
        super().__init__()
        self.frame_cond_injection = str(frame_cond_injection)
        if self.frame_cond_injection not in FRAME_COND_INJECTION_MODES:
            raise ValueError(f"Unsupported frame_cond_injection: {self.frame_cond_injection}")
        self.representation = _instantiate_if_config(representation)
        if _is_config(denoiser):
            denoiser_cfg = dict(denoiser)
            denoiser_cfg["frame_cond_injection"] = self.frame_cond_injection
            self.denoiser = instantiate(denoiser_cfg)
        else:
            self.denoiser = denoiser
            actual_frame_cond = getattr(self.denoiser, "frame_cond_injection", None)
            if (
                actual_frame_cond is not None
                and str(actual_frame_cond) != self.frame_cond_injection
                and self.frame_cond_injection == "sum_to_time"
                and hasattr(self.denoiser, "set_frame_cond_injection")
            ):
                self.denoiser.set_frame_cond_injection(self.frame_cond_injection)
        self._validate_denoiser_frame_cond_injection()
        self.diffusion = _instantiate_if_config(diffusion)
        self.motion_loss = _instantiate_if_config(loss)
        self.diffusion_target = str(diffusion_target)
        if self.diffusion_target not in {"future", "history_future"}:
            raise ValueError(f"Unsupported diffusion_target: {self.diffusion_target}")

        self.condition_dim = int(condition_dim or getattr(self.denoiser, "hidden_dim", 768))
        self.text_encoder = None
        if text_encoder is not None:
            if isinstance(text_encoder, nn.Module):
                self.text_encoder = text_encoder
            elif _is_config(text_encoder):
                text_cfg = dict(text_encoder)
                if "_target_" in text_cfg:
                    self.text_encoder = instantiate(text_cfg)
                else:
                    from omg.generation.conditions.t5 import FrozenT5TextEncoder

                    self.text_encoder = FrozenT5TextEncoder(**text_cfg)
            else:
                from omg.generation.conditions.t5 import FrozenT5TextEncoder

                self.text_encoder = FrozenT5TextEncoder(**dict(text_encoder))

        self.history_projector = nn.Sequential(
            nn.LayerNorm(self.representation.feat_dim),
            nn.Linear(self.representation.feat_dim, self.condition_dim),
            nn.SiLU(),
            nn.Linear(self.condition_dim, self.condition_dim),
        )
        self.use_audio = bool(use_audio)
        self.audio_dim = int(audio_dim)
        self.audio_mask_prob = float(audio_mask_prob)
        self.audio_embedder = None
        self.use_human_motion = bool(use_human_motion)
        if self.use_audio or self.use_human_motion:
            from omg.generation.denoisers.transformer import MotionTransformerDenoiser

            if not isinstance(self.denoiser, MotionTransformerDenoiser):
                raise NotImplementedError(
                    "audio/human_motion frame-level conditioning is currently implemented only for "
                    "guided diffusion with MotionTransformerDenoiser"
                )
        if self.use_audio:
            self.audio_embedder = nn.Sequential(
                nn.LayerNorm(self.audio_dim),
                nn.Linear(self.audio_dim, self.condition_dim * 2),
                nn.SiLU(),
                nn.Dropout(float(getattr(self.denoiser, "dropout", 0.0))),
                nn.Linear(self.condition_dim * 2, self.condition_dim),
            )
        self.human_motion_dim = int(human_motion_dim)
        self.human_motion_mask_prob = float(human_motion_mask_prob)
        self.human_motion_embedder = None
        if self.use_human_motion:
            self.human_motion_embedder = nn.Sequential(
                nn.LayerNorm(self.human_motion_dim),
                nn.Linear(self.human_motion_dim, self.condition_dim * 2),
                nn.SiLU(),
                nn.Dropout(float(getattr(self.denoiser, "dropout", 0.0))),
                nn.Linear(self.condition_dim * 2, self.condition_dim),
            )
        self.text_mask_prob = float(text_mask_prob)
        self.history_mask_prob = float(history_mask_prob)
        self.optimizer_cfg = optimizer or {"lr": 2e-4, "weight_decay": 0.01}
        self.scheduler_cfg = scheduler
        self.log_loss_term_grad_norms = bool(log_loss_term_grad_norms)
        self.loss_term_grad_norm_every_n_steps = int(loss_term_grad_norm_every_n_steps)
        self.loss_term_grad_norm_norm_type = float(loss_term_grad_norm_norm_type)
        self._logged_condition_shapes = False
        self._configure_denoiser_frame_condition_grads()

    def _validate_denoiser_frame_cond_injection(self) -> None:
        actual = getattr(self.denoiser, "frame_cond_injection", None)
        if actual is None:
            return
        actual = str(actual)
        if actual != self.frame_cond_injection:
            raise ValueError(
                "MotionGenerator frame_cond_injection must match the already-instantiated denoiser: "
                f"model={self.frame_cond_injection!r}, denoiser={actual!r}. "
                "Instantiate the denoiser with the same mode instead of changing the mode after construction."
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint[MODEL_ARCHITECTURE_KEY] = build_model_architecture_contract(self)

    def _configure_denoiser_frame_condition_grads(self) -> None:
        separate_mode = self.frame_cond_injection == "separate_to_h"
        modality_flags = {
            "audio": separate_mode and self.use_audio,
            "human_motion": separate_mode and self.use_human_motion,
        }
        for name, enabled in modality_flags.items():
            for suffix in ("adapter", "gate"):
                module_or_param = getattr(self.denoiser, f"{name}_{suffix}", None)
                if module_or_param is None:
                    continue
                if isinstance(module_or_param, nn.Parameter):
                    module_or_param.requires_grad = enabled
                elif isinstance(module_or_param, nn.Module):
                    for param in module_or_param.parameters():
                        param.requires_grad = enabled
        film_mode = self.frame_cond_injection == "per_layer_film"
        control_local_attn_mode = self.frame_cond_injection == "control_local_attn"
        layers = getattr(self.denoiser, "layers", [])
        for layer in layers:
            for module_name, enabled in (
                ("audio_film", film_mode and self.use_audio),
                ("human_motion_film", film_mode and self.use_human_motion),
                ("audio_local_attn", control_local_attn_mode and self.use_audio),
                ("human_motion_control", control_local_attn_mode and self.use_human_motion),
                ("human_motion_local_attn", control_local_attn_mode and self.use_human_motion),
            ):
                module = getattr(layer, module_name, None)
                if module is None:
                    continue
                for param in module.parameters():
                    param.requires_grad = enabled
            for param_name, enabled in (
                ("audio_local_attn_gate", control_local_attn_mode and self.use_audio),
                ("human_motion_control_gate", control_local_attn_mode and self.use_human_motion),
                ("human_motion_local_attn_gate", control_local_attn_mode and self.use_human_motion),
            ):
                param = getattr(layer, param_name, None)
                if isinstance(param, nn.Parameter):
                    param.requires_grad = enabled

    def _history_features(self, batch: dict) -> torch.Tensor:
        history = batch.get("history_features", batch.get("prev_state_features"))
        if history is None:
            raise KeyError("history_features is required")
        return history

    def _captions(self, batch: dict) -> list[str]:
        captions = batch.get("caption")
        if captions is None:
            return ["" for _ in range(int(batch.get("B", 1)))]
        if isinstance(captions, str):
            return [captions]
        return [str(item) for item in captions]

    def _text_context(
        self,
        batch: dict,
        force_null_text: bool,
        is_training: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captions = self._captions(batch)
        if self.text_encoder is None:
            text_proj = getattr(self.denoiser, "text_proj", None)
            text_dim = int(getattr(text_proj, "in_features", self.condition_dim))
            context = torch.zeros(len(captions), 1, text_dim, device=device)
            mask = torch.ones(len(captions), 1, dtype=torch.bool, device=device)
            return context, mask
        has_text = batch.get("has_text")
        if has_text is None:
            has_text = torch.tensor([caption != "" for caption in captions], dtype=torch.bool, device=device)
        else:
            has_text = has_text.to(device=device, dtype=torch.bool)
        if is_training and self.text_mask_prob > 0.0 and not force_null_text:
            keep = torch.rand(has_text.shape, device=device) >= self.text_mask_prob
            has_text = has_text & keep
        encoded = self.text_encoder(
            captions,
            has_text=has_text,
            force_null_text=force_null_text,
            device=device,
        )
        return encoded["context"], encoded["mask"]

    def _require_frame_feature(
        self,
        batch: dict,
        key: str,
        option_name: str,
        expected_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = batch.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"{option_name}=True requires batch['{key}'] to be a Tensor, got {type(value).__name__}")
        value = value.to(device=device, dtype=dtype)
        valid = batch["mask"]["valid"].to(device=device, dtype=torch.bool)
        if value.ndim != 3:
            raise ValueError(f"Expected batch['{key}'] shape (B, L, {expected_dim}), got {tuple(value.shape)}")
        if value.shape[:2] != valid.shape:
            raise ValueError(
                f"batch['{key}'] length must match mask.valid: feature={tuple(value.shape[:2])}, "
                f"valid={tuple(valid.shape)}"
            )
        if value.shape[-1] != expected_dim:
            raise ValueError(f"Expected batch['{key}'] feature dim {expected_dim}, got {value.shape[-1]}")
        return value

    def _frame_mask(self, batch: dict, key: str, device: torch.device) -> torch.Tensor:
        valid = batch["mask"]["valid"].to(device=device, dtype=torch.bool)
        mask = batch["mask"].get(key)
        if mask is None:
            raise ValueError(
                f"mask['{key}'] is required when the corresponding frame-level condition is enabled. "
                "Use False for missing frames instead of relying on zero features."
            )
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.shape != valid.shape:
            raise ValueError(f"Expected mask['{key}'] shape {tuple(valid.shape)}, got {tuple(mask.shape)}")
        return mask & valid

    @staticmethod
    def _has_frame_condition_input(batch: dict, feature_key: str, mask_key: str) -> bool:
        if not torch.is_tensor(batch.get(feature_key)):
            return False
        mask = batch.get("mask", {}).get(mask_key)
        return not torch.is_tensor(mask) or bool(mask.any().item())

    def _log_condition_shapes_once(self, **items: torch.Tensor | None) -> None:
        if self._logged_condition_shapes:
            return
        pieces = []
        for name, value in items.items():
            if torch.is_tensor(value):
                pieces.append(f"{name}={tuple(value.shape)}")
            elif value is not None:
                pieces.append(f"{name}={type(value).__name__}")
        if pieces:
            print("[INFO] MotionGenerator condition shapes: " + " ".join(pieces))
        self._logged_condition_shapes = True

    def _conditions(
        self,
        batch: dict,
        force_null_text: bool = False,
        force_null_audio: bool = False,
        force_null_human_motion: bool = False,
    ) -> dict[str, torch.Tensor]:
        device = next(self.denoiser.parameters()).device
        history = self._history_features(batch).to(device=device, dtype=self.representation.mean.dtype)
        history_norm = self.representation.normalize_features(history)
        history_tokens = self.history_projector(history_norm)
        if self.training and not force_null_text:
            if self.history_mask_prob > 0.0:
                keep_history = (torch.rand(history.shape[0], 1, 1, device=device) >= self.history_mask_prob).to(history_tokens.dtype)
                history_tokens = history_tokens * keep_history
        extra_tokens = [history_tokens]
        text_context, text_mask = self._text_context(
            batch,
            force_null_text=force_null_text,
            is_training=self.training,
            device=device,
        )
        conditions = {
            "text_context": text_context,
            "text_mask": text_mask,
            "extra_tokens": torch.cat(extra_tokens, dim=1),
        }
        frame_cond = None
        audio = audio_cond = audio_mask = human_motion = human_motion_cond = human_motion_mask = None
        has_audio_input = self._has_frame_condition_input(batch, "audio_features", "has_audio")
        has_human_motion_input = self._has_frame_condition_input(batch, "human_motion", "has_human_motion")
        if self.use_audio and self.audio_embedder is not None and has_audio_input:
            audio = self._require_frame_feature(batch, "audio_features", "use_audio", self.audio_dim, device, history.dtype)
            audio_mask = self._frame_mask(batch, "has_audio", device)
            audio_cond = self.audio_embedder(audio)
            if force_null_audio:
                audio_mask = torch.zeros_like(audio_mask)
                audio_cond = audio_cond.new_zeros(audio_cond.shape)
            elif self.training and self.audio_mask_prob > 0.0:
                keep_audio = torch.rand(audio.shape[0], device=device) >= self.audio_mask_prob
                audio_mask = audio_mask & keep_audio[:, None]
                audio_cond = audio_cond * keep_audio[:, None, None].to(audio_cond.dtype)
            audio_cond = audio_cond * audio_mask.unsqueeze(-1).to(audio_cond.dtype)
            if self.frame_cond_injection == "sum_to_time":
                frame_cond = audio_cond if frame_cond is None else frame_cond + audio_cond
        if self.use_human_motion and self.human_motion_embedder is not None and has_human_motion_input:
            human_motion = self._require_frame_feature(
                batch,
                "human_motion",
                "use_human_motion",
                self.human_motion_dim,
                device,
                history.dtype,
            )
            human_motion_mask = self._frame_mask(batch, "has_human_motion", device)
            human_motion_cond = self.human_motion_embedder(human_motion)
            if force_null_human_motion:
                human_motion_mask = torch.zeros_like(human_motion_mask)
                human_motion_cond = human_motion_cond.new_zeros(human_motion_cond.shape)
            elif self.training and self.human_motion_mask_prob > 0.0:
                keep_human_motion = torch.rand(human_motion.shape[0], device=device) >= self.human_motion_mask_prob
                human_motion_mask = human_motion_mask & keep_human_motion[:, None]
                human_motion_cond = human_motion_cond * keep_human_motion[:, None, None].to(human_motion_cond.dtype)
            human_motion_cond = human_motion_cond * human_motion_mask.unsqueeze(-1).to(human_motion_cond.dtype)
            if self.frame_cond_injection == "sum_to_time":
                frame_cond = human_motion_cond if frame_cond is None else frame_cond + human_motion_cond
        if self.frame_cond_injection == "sum_to_time" and frame_cond is not None:
            if self.diffusion_target == "history_future":
                zeros = frame_cond.new_zeros(frame_cond.shape[0], history.shape[1], frame_cond.shape[2])
                frame_cond = torch.cat([zeros, frame_cond], dim=1)
            conditions["frame_cond"] = frame_cond
        if self.frame_cond_injection in SEPARATE_FRAME_COND_MODES:
            if audio_cond is not None and self.frame_cond_injection in SEPARATE_FRAME_COND_MODES:
                if self.diffusion_target == "history_future":
                    zeros = audio_cond.new_zeros(audio_cond.shape[0], history.shape[1], audio_cond.shape[2])
                    audio_cond = torch.cat([zeros, audio_cond], dim=1)
                    audio_mask = torch.cat(
                        [
                            torch.zeros(audio_mask.shape[0], history.shape[1], dtype=torch.bool, device=device),
                            audio_mask,
                        ],
                        dim=1,
                    )
                conditions["audio_cond"] = audio_cond
                if self.frame_cond_injection in MASKED_FRAME_COND_MODES:
                    conditions["audio_mask"] = audio_mask
            if human_motion_cond is not None and self.frame_cond_injection in SEPARATE_FRAME_COND_MODES:
                if self.diffusion_target == "history_future":
                    zeros = human_motion_cond.new_zeros(
                        human_motion_cond.shape[0],
                        history.shape[1],
                        human_motion_cond.shape[2],
                    )
                    human_motion_cond = torch.cat([zeros, human_motion_cond], dim=1)
                    human_motion_mask = torch.cat(
                        [
                            torch.zeros(human_motion_mask.shape[0], history.shape[1], dtype=torch.bool, device=device),
                            human_motion_mask,
                        ],
                        dim=1,
                    )
                conditions["human_motion_cond"] = human_motion_cond
                if self.frame_cond_injection in MASKED_FRAME_COND_MODES:
                    conditions["human_motion_mask"] = human_motion_mask
        self._log_condition_shapes_once(
            audio_features=audio,
            audio_cond=audio_cond,
            audio_mask=audio_mask,
            human_motion=human_motion,
            human_motion_cond=human_motion_cond,
            human_motion_mask=human_motion_mask,
            frame_cond=frame_cond,
        )
        return conditions

    @staticmethod
    def _clone_conditions(conditions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.clone() if torch.is_tensor(value) else value for key, value in conditions.items()}

    def _cfg_condition_branches(
        self,
        batch: dict,
        *,
        cfg_text_scale: float,
        cfg_audio_scale: float,
        cfg_human_scale: float,
    ) -> tuple[dict[str, torch.Tensor], list[tuple[dict[str, torch.Tensor], float]] | None]:
        null_conditions = self._conditions(
            batch,
            force_null_text=True,
            force_null_audio=True,
            force_null_human_motion=True,
        )
        branches: list[tuple[dict[str, torch.Tensor], float]] = []
        has_audio_input = self._has_frame_condition_input(batch, "audio_features", "has_audio")
        has_human_motion_input = self._has_frame_condition_input(batch, "human_motion", "has_human_motion")
        if float(cfg_text_scale) != 0.0:
            text_only = self._conditions(
                batch,
                force_null_text=False,
                force_null_audio=True,
                force_null_human_motion=True,
            )
            branches.append((text_only, float(cfg_text_scale)))
        if self.use_audio and has_audio_input and float(cfg_audio_scale) != 0.0:
            audio_only = self._conditions(
                batch,
                force_null_text=True,
                force_null_audio=False,
                force_null_human_motion=True,
            )
            branches.append((audio_only, float(cfg_audio_scale)))
        if self.use_human_motion and has_human_motion_input and float(cfg_human_scale) != 0.0:
            human_only = self._conditions(
                batch,
                force_null_text=True,
                force_null_audio=True,
                force_null_human_motion=False,
            )
            branches.append((human_only, float(cfg_human_scale)))
        return null_conditions, branches or None

    def _target_sequence(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
        future = self.representation.encode(batch)
        valid_future = batch["mask"]["valid"].to(future.device).bool()
        if self.diffusion_target == "future":
            return future, valid_future, 0
        history = self._history_features(batch).to(future.device)
        history_norm = self.representation.normalize_features(history)
        valid_history = torch.ones(history_norm.shape[:2], dtype=torch.bool, device=future.device)
        return torch.cat([history_norm, future], dim=1), torch.cat([valid_history, valid_future], dim=1), history_norm.shape[1]

    def _motion_loss_disabled(self) -> bool:
        weights = getattr(self.motion_loss, "weights", None)
        if not isinstance(weights, dict) or not weights:
            return False
        return all(float(weight) == 0.0 for weight in weights.values())

    def _zero_motion_terms(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        return {
            "motion_loss": zero,
            "motion_loss_unclipped": zero,
            "loss_term_clip_active_count": zero,
        }

    def _shared_step(self, batch: dict, split: str, batch_idx: int | None = None) -> torch.Tensor:
        target, valid, history_len = self._target_sequence(batch)
        conditions = self._conditions(batch)
        diff = self.diffusion.training_losses(
            self.denoiser,
            target,
            conditions,
            valid,
            history_len=history_len,
        )
        pred_norm = diff["pred_x0"][:, history_len:]
        pred_features = self.representation.denormalize_features(pred_norm)
        target_features = batch["motion_features"].to(pred_features.device)
        if self._motion_loss_disabled():
            motion_terms = self._zero_motion_terms(diff["diffusion_loss"])
        else:
            motion_terms = self.motion_loss(pred_features, target_features, batch, self.representation)
        total = diff["diffusion_loss"] + motion_terms["motion_loss"]
        batch_size = target.shape[0]
        loss_term_grad_norms = None
        should_log_loss_term_grad_norms = self._should_log_loss_term_grad_norms(split)
        if should_log_loss_term_grad_norms or self._should_capture_guard_loss_term_grad_norms(split):
            loss_term_grad_norms = self._compute_loss_term_grad_norms(diff["diffusion_loss"], motion_terms)
        if should_log_loss_term_grad_norms and loss_term_grad_norms is not None:
            self.log_dict(
                {f"{split}/loss_term_grad_norm/{name}": value for name, value in loss_term_grad_norms.items()},
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
                sync_dist=True,
            )
        self.log(f"{split}/loss", total, prog_bar=(split == "train"), batch_size=batch_size, sync_dist=True)
        self.log(f"{split}/diffusion_loss", diff["diffusion_loss"], batch_size=batch_size, sync_dist=True)
        for name, value in motion_terms.items():
            if torch.is_tensor(value) and value.ndim == 0:
                self.log(f"{split}/{name}", value, batch_size=batch_size, sync_dist=True)
        if split == "train":
            self._last_train_diagnostics = self._build_train_diagnostics(
                batch=batch,
                batch_idx=batch_idx,
                total=total,
                diffusion_loss=diff["diffusion_loss"],
                motion_terms=motion_terms,
                loss_term_grad_norms=loss_term_grad_norms,
                diffusion_info=diff,
                pred_features=pred_features,
                target_features=target_features,
            )
        return total

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train", batch_idx=batch_idx)

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self._shared_step(batch, "val", batch_idx=batch_idx)

    def _should_log_loss_term_grad_norms(self, split: str) -> bool:
        if split != "train" or not self.log_loss_term_grad_norms:
            return False
        every = self.loss_term_grad_norm_every_n_steps
        if every <= 0:
            return False
        step = int(getattr(self, "global_step", 0))
        return every == 1 or step % every == 0

    def _should_capture_guard_loss_term_grad_norms(self, split: str) -> bool:
        return split == "train" and bool(getattr(self, "_capture_loss_term_grad_norms_for_divergence_guard", False))

    def _compute_loss_term_grad_norms(
        self,
        diffusion_loss: torch.Tensor,
        motion_terms: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        params = [param for param in self.parameters() if param.requires_grad]
        if not params:
            zero = diffusion_loss.new_zeros(())
            return {name: zero for name in self._loss_terms_for_grad_norm(diffusion_loss, motion_terms)}
        out: dict[str, torch.Tensor] = {}
        for name, term in self._loss_terms_for_grad_norm(diffusion_loss, motion_terms).items():
            out[name] = self._loss_term_grad_norm(term, params)
        return out

    def _loss_terms_for_grad_norm(
        self,
        diffusion_loss: torch.Tensor,
        motion_terms: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        terms = {"diffusion_loss": diffusion_loss}
        weights = getattr(self.motion_loss, "weights", {})
        for raw_name, weight in weights.items():
            key = f"{raw_name}_loss"
            value = motion_terms.get(key)
            if torch.is_tensor(value) and value.ndim == 0 and float(weight) != 0.0:
                terms[key] = value * float(weight)
        motion_loss = motion_terms.get("motion_loss")
        if torch.is_tensor(motion_loss) and motion_loss.ndim == 0:
            terms["motion_loss"] = motion_loss
        return terms

    def _loss_term_grad_norm(self, term: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
        if not term.requires_grad:
            return term.detach().new_zeros(())
        grads = torch.autograd.grad(
            term,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        norms = [
            torch.linalg.vector_norm(grad.detach().float(), ord=self.loss_term_grad_norm_norm_type).to(term.device)
            for grad in grads
            if grad is not None
        ]
        if not norms:
            return term.detach().new_zeros(())
        return torch.linalg.vector_norm(torch.stack(norms), ord=self.loss_term_grad_norm_norm_type)

    def _build_train_diagnostics(
        self,
        *,
        batch: dict,
        batch_idx: int | None,
        total: torch.Tensor,
        diffusion_loss: torch.Tensor,
        motion_terms: dict[str, torch.Tensor],
        loss_term_grad_norms: dict[str, torch.Tensor] | None,
        diffusion_info: dict[str, torch.Tensor],
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> dict[str, Any]:
        codec = self.representation.codec
        root_rot_start, root_rot_end = codec.feature_slices["root_rot_local"]
        pred_root_rot = pred_features[..., root_rot_start:root_rot_end].detach().float()
        target_root_rot = target_features[..., root_rot_start:root_rot_end].detach().float()
        pred_root_quat = codec.rotation_features_to_quat(pred_root_rot).detach().float()
        target_root_quat = codec.rotation_features_to_quat(target_root_rot).detach().float()
        valid = batch["mask"]["valid"].detach().bool()
        caption = batch.get("caption", [])
        meta = batch.get("meta", [])
        diffusion_timesteps = self._diffusion_timestep_diagnostics(diffusion_info)
        return {
            "batch_idx": None if batch_idx is None else int(batch_idx),
            "batch_size": int(pred_features.shape[0]),
            "loss": self._scalar(total),
            "diffusion_loss": self._scalar(diffusion_loss),
            "motion_terms": {
                name: self._scalar(value)
                for name, value in motion_terms.items()
                if torch.is_tensor(value) and value.ndim == 0
            },
            "loss_term_grad_norms": None
            if loss_term_grad_norms is None
            else {name: self._scalar(value) for name, value in loss_term_grad_norms.items()},
            "numeric": {
                "rotation_representation": getattr(codec, "rotation_representation", "quat"),
                "pred_features_abs_max": self._scalar(pred_features.detach().float().abs().max()),
                "target_features_abs_max": self._scalar(target_features.detach().float().abs().max()),
                "pred_root_rot_feature_norm_min": self._scalar(torch.linalg.vector_norm(pred_root_rot, dim=-1).min()),
                "pred_root_rot_feature_norm_max": self._scalar(torch.linalg.vector_norm(pred_root_rot, dim=-1).max()),
                "target_root_rot_feature_norm_min": self._scalar(torch.linalg.vector_norm(target_root_rot, dim=-1).min()),
                "target_root_rot_feature_norm_max": self._scalar(torch.linalg.vector_norm(target_root_rot, dim=-1).max()),
                "pred_root_quat_norm_min": self._scalar(torch.linalg.vector_norm(pred_root_quat, dim=-1).min()),
                "pred_root_quat_norm_max": self._scalar(torch.linalg.vector_norm(pred_root_quat, dim=-1).max()),
                "target_root_quat_norm_min": self._scalar(torch.linalg.vector_norm(target_root_quat, dim=-1).min()),
                "target_root_quat_norm_max": self._scalar(torch.linalg.vector_norm(target_root_quat, dim=-1).max()),
                "valid_frames": int(valid.sum().detach().cpu().item()),
                "finite_pred_features": bool(torch.isfinite(pred_features.detach()).all().cpu().item()),
                "finite_target_features": bool(torch.isfinite(target_features.detach()).all().cpu().item()),
            },
            "diffusion_timesteps": diffusion_timesteps,
            "batch_meta": self._batch_meta(meta, caption),
        }

    def _diffusion_timestep_diagnostics(self, diffusion_info: dict[str, torch.Tensor]) -> dict[str, Any] | None:
        timesteps = diffusion_info.get("timesteps")
        if not torch.is_tensor(timesteps):
            return None
        t = timesteps.detach().long().cpu()
        if t.ndim == 0:
            t = t.reshape(1)
        flat = t.reshape(-1)
        out: dict[str, Any] = {
            "timesteps": t.tolist(),
            "min": int(flat.min().item()),
            "max": int(flat.max().item()),
            "mean": float(flat.float().mean().item()),
        }
        train_map = getattr(self.diffusion, "train_timestep_map", None)
        if torch.is_tensor(train_map):
            train_map_cpu = train_map.detach().long().cpu()
            index_by_timestep = {int(value.item()): idx for idx, value in enumerate(train_map_cpu)}
            out["train_timestep_indices"] = [index_by_timestep.get(int(value.item())) for value in flat]
        sqrt_alpha = getattr(self.diffusion, "base_sqrt_alphas_cumprod", None)
        sqrt_one_minus = getattr(self.diffusion, "base_sqrt_one_minus_alphas_cumprod", None)
        if torch.is_tensor(sqrt_alpha) and torch.is_tensor(sqrt_one_minus):
            safe = flat.clamp(min=0, max=int(sqrt_alpha.numel()) - 1)
            alpha_values = sqrt_alpha.detach().float().cpu().index_select(0, safe)
            noise_values = sqrt_one_minus.detach().float().cpu().index_select(0, safe)
            out["sqrt_alpha_cumprod"] = alpha_values.reshape(t.shape).tolist()
            out["sqrt_one_minus_alpha_cumprod"] = noise_values.reshape(t.shape).tolist()
            out["sqrt_alpha_cumprod_min"] = float(alpha_values.min().item())
            out["sqrt_alpha_cumprod_max"] = float(alpha_values.max().item())
            out["sqrt_one_minus_alpha_cumprod_min"] = float(noise_values.min().item())
            out["sqrt_one_minus_alpha_cumprod_max"] = float(noise_values.max().item())
        return out

    @staticmethod
    def _scalar(value: torch.Tensor) -> float | None:
        out = float(value.detach().float().cpu().item())
        return out if math.isfinite(out) else None

    @staticmethod
    def _batch_meta(meta: Any, caption: Any) -> list[dict[str, Any]]:
        captions = list(caption) if isinstance(caption, (list, tuple)) else []
        items = meta if isinstance(meta, list) else []
        out = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "sample_idx": idx,
                    "dataset": str(item.get("source_dataset", "")),
                    "source_file": str(item.get("source_file", "")),
                    "sequence_name": item.get("sequence_name", ""),
                    "window_start": item.get("window_start"),
                    "window_end": item.get("window_end"),
                    "segment_index": item.get("segment_index"),
                    "segment_frame_start": item.get("segment_frame_start"),
                    "segment_frame_end": item.get("segment_frame_end"),
                    "caption": captions[idx] if idx < len(captions) else item.get("video_summary", ""),
                    "label_path": item.get("label_path"),
                }
            )
        return out

    def _build_lr_scheduler(self, opt: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        cfg = self.scheduler_cfg or {}
        scheduler_type = str(cfg.get("type", "cosine"))
        t_max = int(cfg.get("t_max", 100000))
        eta_min = float(cfg.get("eta_min", 1e-6))
        base_lrs = [float(group["lr"]) for group in opt.param_groups]
        if t_max <= 0:
            raise ValueError(f"scheduler.t_max must be positive, got {t_max}")
        if eta_min < 0.0:
            raise ValueError(f"scheduler.eta_min must be non-negative, got {eta_min}")
        if eta_min > min(base_lrs):
            raise ValueError(f"scheduler.eta_min={eta_min} exceeds the minimum optimizer lr={min(base_lrs)}")

        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min)

        if scheduler_type == "linear_warmup_cosine":
            warmup_steps = int(cfg.get("warmup_steps", 0))
            if warmup_steps <= 0:
                raise ValueError(f"scheduler.warmup_steps must be positive for linear_warmup_cosine, got {warmup_steps}")
            if warmup_steps >= t_max:
                raise ValueError(
                    f"scheduler.warmup_steps must be smaller than scheduler.t_max, "
                    f"got warmup_steps={warmup_steps}, t_max={t_max}"
                )
            warmup_start_factor = float(cfg.get("warmup_start_factor", 1.0e-8))
            if not 0.0 < warmup_start_factor <= 1.0:
                raise ValueError(
                    f"scheduler.warmup_start_factor must be in (0, 1], got {warmup_start_factor}"
                )
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=t_max - warmup_steps,
                eta_min=eta_min,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )

        raise ValueError(f"Unsupported scheduler.type: {scheduler_type}")

    def configure_optimizers(self) -> Any:
        params = [param for param in self.parameters() if param.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=float(self.optimizer_cfg.get("lr", 2e-4)),
            weight_decay=float(self.optimizer_cfg.get("weight_decay", 0.01)),
        )
        if not self.scheduler_cfg:
            return opt
        scheduler = self._build_lr_scheduler(opt)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _sample_future_norm(
        self,
        batch: dict,
        future_len: int,
        cfg_scale: float | None = None,
        cfg_text_scale: float | None = None,
        cfg_audio_scale: float | None = None,
        cfg_human_scale: float | None = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        valid_future = torch.ones(batch["history_features"].shape[0], future_len, dtype=torch.bool, device=device)
        if cfg_scale is None:
            text_scale = float(self.diffusion.cfg_scale) if cfg_text_scale is None else float(cfg_text_scale)
            audio_scale = 1.0 if cfg_audio_scale is None else float(cfg_audio_scale)
            human_scale = 1.0 if cfg_human_scale is None else float(cfg_human_scale)
            null_conditions, cfg_branches = self._cfg_condition_branches(
                batch,
                cfg_text_scale=text_scale,
                cfg_audio_scale=audio_scale,
                cfg_human_scale=human_scale,
            )
            conditions = null_conditions
            sample_cfg_scale = 1.0
        else:
            null_conditions = self._conditions(
                batch,
                force_null_text=True,
                force_null_audio=True,
                force_null_human_motion=True,
            )
            conditions = self._conditions(
                batch,
                force_null_text=False,
                force_null_audio=False,
                force_null_human_motion=False,
            )
            cfg_branches = None
            sample_cfg_scale = float(cfg_scale)
        if self.diffusion_target == "future":
            return self.diffusion.sample(
                self.denoiser,
                (batch["history_features"].shape[0], future_len, self.representation.feat_dim),
                conditions,
                valid_mask=valid_future,
                null_conditions=null_conditions,
                cfg_scale=sample_cfg_scale,
                cfg_branches=cfg_branches,
            )

        history = batch["history_features"].to(device=device)
        history_norm = self.representation.normalize_features(history)
        valid = torch.cat(
            [
                torch.ones(history_norm.shape[:2], dtype=torch.bool, device=device),
                valid_future,
            ],
            dim=1,
        )
        sample = self.diffusion.sample(
            self.denoiser,
            (history.shape[0], history.shape[1] + future_len, self.representation.feat_dim),
            conditions,
            history=history_norm,
            valid_mask=valid,
            null_conditions=null_conditions,
            cfg_scale=sample_cfg_scale,
            cfg_branches=cfg_branches,
        )
        return sample[:, history.shape[1] :]

    @torch.no_grad()
    def generate(
        self,
        batch: dict,
        num_frames: int,
        cfg_scale: float | None = None,
        cfg_text_scale: float | None = None,
        cfg_audio_scale: float | None = None,
        cfg_human_scale: float | None = None,
    ) -> dict[str, torch.Tensor | None]:
        device = next(self.parameters()).device
        chunk_len = int(self.representation.sequence_length)
        frames_left = int(num_frames)
        if frames_left <= 0:
            raise ValueError("num_frames must be positive")

        history = self._history_features(batch).to(device=device)
        canon_root_pos = batch["canon_root_pos"].to(device=device)
        canon_root_quat = batch["canon_root_quat"].to(device=device)
        fps = batch["fps"].to(device=device)
        valid_condition = None
        has_audio_input = self.use_audio and self._has_frame_condition_input(batch, "audio_features", "has_audio")
        has_human_motion_input = self.use_human_motion and self._has_frame_condition_input(batch, "human_motion", "has_human_motion")
        if has_audio_input or has_human_motion_input:
            valid_condition = batch["mask"]["valid"].to(device=device, dtype=torch.bool)
            if valid_condition.shape[1] < frames_left:
                raise ValueError(
                    f"generate(num_frames={frames_left}) requires at least {frames_left} valid condition frames, "
                    f"got {valid_condition.shape[1]}"
                )
            if not valid_condition[:, :frames_left].all():
                raise ValueError("generate with frame-level conditions requires all requested condition frames to be valid")
        audio_features = None
        has_audio = None
        if has_audio_input:
            audio_features = self._require_frame_feature(
                batch,
                "audio_features",
                "use_audio",
                self.audio_dim,
                device,
                history.dtype,
            )
            if audio_features.shape[1] < frames_left:
                raise ValueError(
                    f"generate(num_frames={frames_left}) requires at least {frames_left} audio frames, "
                    f"got {audio_features.shape[1]}"
                )
            has_audio = batch["mask"].get("has_audio")
            if has_audio is None:
                raise ValueError("generate with audio requires mask['has_audio']")
            has_audio = has_audio.to(device=device, dtype=torch.bool)
        human_motion = None
        has_human_motion = None
        if has_human_motion_input:
            human_motion = self._require_frame_feature(
                batch,
                "human_motion",
                "use_human_motion",
                self.human_motion_dim,
                device,
                history.dtype,
            )
            if human_motion.shape[1] < frames_left:
                raise ValueError(
                    f"generate(num_frames={frames_left}) requires at least {frames_left} human reference frames, "
                    f"got {human_motion.shape[1]}"
                )
            has_human_motion = batch["mask"].get("has_human_motion")
            if has_human_motion is None:
                raise ValueError("generate with human reference condition requires mask['has_human_motion']")
            has_human_motion = has_human_motion.to(device=device, dtype=torch.bool)
        qpos_chunks = []
        feature_chunks = []
        frame_offset = 0
        while frames_left > 0:
            curr_len = min(chunk_len, frames_left)
            sample_batch = {
                "B": history.shape[0],
                "history_features": history,
                "prev_state_features": history,
                "canon_root_pos": canon_root_pos,
                "canon_root_quat": canon_root_quat,
                "fps": fps,
                "caption": batch.get("caption", ["" for _ in range(history.shape[0])]),
                "has_text": batch.get("has_text", torch.ones(history.shape[0], dtype=torch.bool, device=device)),
                "mask": {
                    "valid": torch.ones(history.shape[0], curr_len, dtype=torch.bool, device=device),
                },
            }
            if audio_features is not None:
                sample_batch["audio_features"] = audio_features[:, frame_offset : frame_offset + curr_len]
                sample_batch["mask"]["has_audio"] = has_audio[:, frame_offset : frame_offset + curr_len]
            if human_motion is not None:
                sample_batch["human_motion"] = human_motion[:, frame_offset : frame_offset + curr_len]
                sample_batch["mask"]["has_human_motion"] = has_human_motion[:, frame_offset : frame_offset + curr_len]
            future_norm = self._sample_future_norm(
                sample_batch,
                curr_len,
                cfg_scale=cfg_scale,
                cfg_text_scale=cfg_text_scale,
                cfg_audio_scale=cfg_audio_scale,
                cfg_human_scale=cfg_human_scale,
            )
            decoded = self.representation.decode(future_norm)
            qpos = self.representation.compose_qpos_36(decoded, canon_root_pos, canon_root_quat)
            features = self.representation.denormalize_features(future_norm)
            qpos_chunks.append(qpos)
            feature_chunks.append(features)

            prev_qpos = qpos[:, -self.representation.num_prev_states :]
            prev_fk = self.representation.kinematics.forward_kinematics(prev_qpos)
            history, canon_root_pos, canon_root_quat = self.representation.codec.prev_state_features_from_history(
                prev_qpos,
                prev_fk["body_pos_w"],
                prev_fk["body_quat_w"],
                fps=fps,
            )
            frames_left -= curr_len
            frame_offset += curr_len

        return {
            "motion_features": torch.cat(feature_chunks, dim=1),
            "qpos_36": torch.cat(qpos_chunks, dim=1),
        }
