from __future__ import annotations

import torch
import torch.nn as nn

from omg.generation.diffusion.schedules import make_beta_schedule


def _condition_tensor(conditions: dict) -> torch.Tensor:
    for value in conditions.values():
        if torch.is_tensor(value):
            return value
    raise ValueError("Condition dictionary does not contain any tensors")


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target_ndim: int) -> torch.Tensor:
    out = values.to(timesteps.device)[timesteps.long()]
    while out.ndim < target_ndim:
        out = out.unsqueeze(-1)
    return out


class DiffusionForcingProcess(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        sampling_steps: int = 50,
        objective: str = "pred_x0",
        beta_schedule: str = "cosine",
        noise_level: str = "random_independent",
        loss_weighting: str = "min_snr",
        snr_clip: float = 5.0,
        ddim_eta: float = 0.0,
        clip_noise: float = 3.0,
        cfg_scale: float = 2.5,
    ):
        super().__init__()
        if objective != "pred_x0":
            raise ValueError("DiffusionForcingProcess is configured for x0 prediction")
        if noise_level not in {"random_independent", "random_uniform"}:
            raise ValueError(f"Unsupported noise_level: {noise_level}")
        self.timesteps = int(timesteps)
        self.sampling_steps = int(sampling_steps)
        self.objective = str(objective)
        self.noise_level = str(noise_level)
        self.loss_weighting = str(loss_weighting)
        self.snr_clip = float(snr_clip)
        self.ddim_eta = float(ddim_eta)
        self.clip_noise = float(clip_noise)
        self.cfg_scale = float(cfg_scale)

        betas = make_beta_schedule(beta_schedule, self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).clamp(min=1e-10, max=1.0)
        one_minus = (1.0 - alphas_cumprod).clamp(min=1e-10)
        snr = alphas_cumprod / one_minus
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(one_minus), persistent=False)
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0), persistent=False)
        self.register_buffer("snr", snr.clamp(max=1e6), persistent=False)

    def _make_noise_levels(
        self,
        batch_size: int,
        seq_len: int,
        history_len: int,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        device = valid_mask.device
        if self.noise_level == "random_uniform":
            levels = torch.randint(0, self.timesteps, (batch_size, 1), device=device).expand(batch_size, seq_len).clone()
        else:
            levels = torch.randint(0, self.timesteps, (batch_size, seq_len), device=device)
        if history_len > 0:
            levels[:, :history_len] = -1
        levels = torch.where(valid_mask.bool(), levels, torch.full_like(levels, self.timesteps - 1))
        return levels

    def _q_sample(self, x_start: torch.Tensor, levels: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        safe = levels.clamp(min=0)
        x_t = _extract(self.sqrt_alphas_cumprod, safe, x_start.ndim).to(x_start.dtype) * x_start
        x_t = x_t + _extract(self.sqrt_one_minus_alphas_cumprod, safe, x_start.ndim).to(x_start.dtype) * noise
        clean_mask = levels < 0
        if clean_mask.any():
            x_t = torch.where(clean_mask.unsqueeze(-1), x_start, x_t)
        return x_t

    def _predict_noise_from_start(self, x_t: torch.Tensor, levels: torch.Tensor, pred_x0: torch.Tensor) -> torch.Tensor:
        safe = levels.clamp(min=0)
        return (
            _extract(self.sqrt_recip_alphas_cumprod, safe, x_t.ndim).to(x_t.dtype) * x_t
            - pred_x0
        ) / _extract(self.sqrt_recipm1_alphas_cumprod, safe, x_t.ndim).to(x_t.dtype).clamp_min(1e-8)

    def _loss_weights(self, levels: torch.Tensor, ndim: int, dtype: torch.dtype) -> torch.Tensor:
        safe = levels.clamp(min=0)
        if self.loss_weighting == "uniform":
            weights = torch.ones_like(safe, dtype=torch.float32)
        elif self.loss_weighting == "min_snr":
            snr = self.snr.to(safe.device)[safe]
            weights = snr.clamp(max=self.snr_clip)
        else:
            raise ValueError(f"Unsupported loss_weighting: {self.loss_weighting}")
        weights = weights.to(dtype=dtype)
        while weights.ndim < ndim:
            weights = weights.unsqueeze(-1)
        return weights

    def training_losses(
        self,
        denoiser: nn.Module,
        x_start: torch.Tensor,
        conditions: dict,
        valid_mask: torch.Tensor,
        history_len: int = 0,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x_start.shape
        levels = self._make_noise_levels(batch_size, seq_len, history_len, valid_mask)
        noise = torch.randn_like(x_start).clamp(-self.clip_noise, self.clip_noise)
        x_t = self._q_sample(x_start, levels, noise)
        raw = denoiser(x_t, levels.clamp(min=0), conditions, valid_mask=valid_mask)
        pred_x0 = raw.clamp(min=-10.0, max=10.0)

        loss = (raw - x_start.detach()).pow(2)
        loss = loss * self._loss_weights(levels, loss.ndim, loss.dtype)
        loss_mask = valid_mask.bool()
        if history_len > 0:
            steps = torch.arange(seq_len, device=x_start.device)
            loss_mask = loss_mask & (steps.unsqueeze(0) >= int(history_len))
        per_frame = loss.mean(dim=-1)
        mask = loss_mask.to(per_frame.dtype)
        per_sample = (per_frame * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return {"diffusion_loss": per_sample.mean(), "pred_x0": pred_x0, "raw_prediction": raw}

    def _model_predictions(
        self,
        denoiser: nn.Module,
        x: torch.Tensor,
        levels: torch.Tensor,
        conditions: dict,
        valid_mask: torch.Tensor,
        null_conditions: dict | None,
        cfg_scale: float,
        cfg_branches: list[tuple[dict, float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe = levels.clamp(min=0)
        if cfg_branches is not None:
            if null_conditions is None:
                raise ValueError("cfg_branches requires null_conditions")
            null_out = denoiser(x, safe, null_conditions, valid_mask=valid_mask)
            model_out = null_out
            for branch_conditions, branch_scale in cfg_branches:
                if float(branch_scale) == 0.0:
                    continue
                branch_out = denoiser(x, safe, branch_conditions, valid_mask=valid_mask)
                model_out = model_out + float(branch_scale) * (branch_out - null_out)
        else:
            model_out = denoiser(x, safe, conditions, valid_mask=valid_mask)
            if null_conditions is not None and cfg_scale != 1.0:
                null_out = denoiser(x, safe, null_conditions, valid_mask=valid_mask)
                model_out = null_out + float(cfg_scale) * (model_out - null_out)
        x_start = model_out.clamp(min=-10.0, max=10.0)
        pred_noise = self._predict_noise_from_start(x, safe, x_start)
        return x_start, pred_noise

    def _ddim_step(
        self,
        denoiser: nn.Module,
        x: torch.Tensor,
        curr_levels: torch.Tensor,
        next_levels: torch.Tensor,
        conditions: dict,
        valid_mask: torch.Tensor,
        null_conditions: dict | None,
        cfg_scale: float,
        cfg_branches: list[tuple[dict, float]] | None = None,
    ) -> torch.Tensor:
        is_curr_clean = curr_levels < 0
        is_next_clean = next_levels < 0
        safe_curr = curr_levels.clamp(min=0)
        safe_next = next_levels.clamp(min=0)
        x_start, pred_noise = self._model_predictions(
            denoiser,
            x,
            safe_curr,
            conditions,
            valid_mask,
            null_conditions,
            cfg_scale,
            cfg_branches=cfg_branches,
        )
        alpha = self.alphas_cumprod.to(x.device)[safe_curr]
        alpha = torch.where(is_curr_clean, torch.ones_like(alpha), alpha)
        alpha_next = self.alphas_cumprod.to(x.device)[safe_next]
        alpha_next = torch.where(is_next_clean, torch.ones_like(alpha_next), alpha_next)
        alpha = alpha.unsqueeze(-1).to(x.dtype)
        alpha_next = alpha_next.unsqueeze(-1).to(x.dtype)
        sigma = self.ddim_eta * torch.sqrt(
            ((1.0 - alpha / alpha_next.clamp_min(1e-8)) * (1.0 - alpha_next) / (1.0 - alpha).clamp_min(1e-8)).clamp_min(0.0)
        )
        sigma = torch.where(is_next_clean.unsqueeze(-1), torch.zeros_like(sigma), sigma)
        coeff = torch.sqrt((1.0 - alpha_next - sigma.pow(2)).clamp_min(0.0))
        noise = torch.randn_like(x).clamp(-self.clip_noise, self.clip_noise)
        x_pred = x_start * torch.sqrt(alpha_next) + coeff * pred_noise + sigma * noise
        x_pred = x_pred.clamp(min=-10.0, max=10.0)
        skip = (curr_levels == next_levels).unsqueeze(-1)
        x_pred = torch.where(skip, x, x_pred)
        return x_pred.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        shape: tuple[int, int, int],
        conditions: dict,
        history: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        null_conditions: dict | None = None,
        cfg_scale: float | None = None,
        cfg_branches: list[tuple[dict, float]] | None = None,
    ) -> torch.Tensor:
        ref = _condition_tensor(conditions)
        batch_size, seq_len, _ = shape
        x = torch.randn(shape, device=ref.device, dtype=ref.dtype)
        history_len = 0 if history is None else int(history.shape[1])
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, seq_len, device=ref.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=ref.device, dtype=torch.bool)
        if history is not None:
            x[:, :history_len] = history.to(device=ref.device, dtype=ref.dtype)
        scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        indices = torch.linspace(self.timesteps - 1, 0, self.sampling_steps, device=ref.device).long()
        for idx, curr in enumerate(indices):
            next_level = indices[idx + 1] if idx < len(indices) - 1 else torch.tensor(-1, device=ref.device)
            curr_levels = torch.full((batch_size, seq_len), int(curr.item()), device=ref.device, dtype=torch.long)
            next_levels = torch.full((batch_size, seq_len), int(next_level.item()), device=ref.device, dtype=torch.long)
            if history_len > 0:
                curr_levels[:, :history_len] = -1
                next_levels[:, :history_len] = -1
                x[:, :history_len] = history.to(device=ref.device, dtype=ref.dtype)
            x = self._ddim_step(
                denoiser,
                x,
                curr_levels,
                next_levels,
                conditions,
                valid_mask,
                null_conditions,
                scale,
                cfg_branches=cfg_branches,
            )
            if history_len > 0:
                x[:, :history_len] = history.to(device=ref.device, dtype=ref.dtype)
        return x
