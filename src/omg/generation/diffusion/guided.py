from __future__ import annotations

import torch
import torch.nn as nn

from omg.generation.diffusion.objectives import objective_target, prediction_to_x0
from omg.generation.diffusion.schedules import make_beta_schedule, sample_timesteps


def _space_timesteps(num_timesteps: int, section_counts: str | int | list[int]) -> list[int]:
    if isinstance(section_counts, int):
        section_counts = str(section_counts)
    if isinstance(section_counts, str):
        if section_counts == "":
            return list(range(num_timesteps))
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for stride in range(1, num_timesteps):
                steps = list(range(0, num_timesteps, stride))
                if len(steps) == desired_count:
                    return steps
            raise ValueError(f"cannot create exactly {desired_count} DDIM steps from {num_timesteps}")
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps: list[int] = []
    for idx, section_count in enumerate(section_counts):
        size = size_per + (1 if idx < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        for _ in range(section_count):
            all_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        start_idx += size
    return sorted(set(all_steps))


def _spaced_betas(base_betas: torch.Tensor, use_timesteps: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    base_betas64 = base_betas.double()
    base_alphas_cumprod = torch.cumprod(1.0 - base_betas64, dim=0)
    last_alpha_cumprod = torch.tensor(1.0, dtype=torch.float64)
    new_betas = []
    timestep_map = []
    use_set = set(use_timesteps)
    for idx, alpha_cumprod in enumerate(base_alphas_cumprod):
        if idx in use_set:
            new_betas.append(1.0 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(idx)
    return torch.stack(new_betas).float(), torch.tensor(timestep_map, dtype=torch.long)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target_ndim: int) -> torch.Tensor:
    out = values.to(timesteps.device).index_select(0, timesteps.long())
    while out.ndim < target_ndim:
        out = out.unsqueeze(-1)
    return out


def _condition_tensor(conditions: dict) -> torch.Tensor:
    for value in conditions.values():
        if torch.is_tensor(value):
            return value
    raise ValueError("Condition dictionary does not contain any tensors")


class GuidedDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        objective: str = "pred_x0",
        beta_schedule: str = "cosine",
        train_timestep_respacing: str = "",
        test_timestep_respacing: str = "50",
        sampler: str = "ddim",
        ddim_eta: float = 0.0,
        cfg_scale: float = 2.5,
        zero_terminal_snr: bool = False,
        loss_weighting: str = "uniform",
        snr_clip: float = 5.0,
        snr_gamma: float | None = None,
        timestep_sampling: str = "uniform",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
    ):
        super().__init__()
        if sampler != "ddim":
            raise ValueError(f"Unsupported sampler: {sampler}")
        if objective != "pred_x0":
            raise ValueError("GuidedDiffusion is configured for x0 prediction")
        self.timesteps = int(timesteps)
        self.objective = str(objective)
        self.train_timestep_respacing = str(train_timestep_respacing)
        self.test_timestep_respacing = str(test_timestep_respacing)
        self.ddim_eta = float(ddim_eta)
        self.cfg_scale = float(cfg_scale)
        self.zero_terminal_snr = bool(zero_terminal_snr)
        self.loss_weighting = str(loss_weighting)
        self.snr_clip = float(snr_clip if snr_gamma is None else snr_gamma)
        self.timestep_sampling = str(timestep_sampling)
        self.logit_normal_mean = float(logit_normal_mean)
        self.logit_normal_std = float(logit_normal_std)

        base_betas = make_beta_schedule(beta_schedule, self.timesteps, zero_terminal_snr=self.zero_terminal_snr)
        base_alphas_cumprod = torch.cumprod(1.0 - base_betas, dim=0)
        one_minus = (1.0 - base_alphas_cumprod).clamp_min(1e-8)
        base_snr = base_alphas_cumprod / one_minus
        self.register_buffer("base_alphas_cumprod", base_alphas_cumprod, persistent=False)
        self.register_buffer("base_sqrt_alphas_cumprod", torch.sqrt(base_alphas_cumprod), persistent=False)
        self.register_buffer(
            "base_sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - base_alphas_cumprod),
            persistent=False,
        )
        self.register_buffer("base_snr", base_snr.clamp(max=1e6), persistent=False)

        train_steps = _space_timesteps(self.timesteps, self.train_timestep_respacing)
        sample_steps = _space_timesteps(self.timesteps, self.test_timestep_respacing)
        _, train_map = _spaced_betas(base_betas, train_steps)
        sample_betas, sample_map = _spaced_betas(base_betas, sample_steps)
        sample_alphas_cumprod = torch.cumprod(1.0 - sample_betas, dim=0)
        self.register_buffer("train_timestep_map", train_map, persistent=False)
        self.register_buffer("sample_timestep_map", sample_map, persistent=False)
        self.register_buffer("sample_alphas_cumprod", sample_alphas_cumprod, persistent=False)
        self.register_buffer(
            "sample_alphas_cumprod_prev",
            torch.cat([torch.ones(1, dtype=sample_alphas_cumprod.dtype), sample_alphas_cumprod[:-1]]),
            persistent=False,
        )

    def _sample_train_indices(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return sample_timesteps(
            (batch_size,),
            self.train_timestep_map.numel(),
            strategy=self.timestep_sampling,
            logit_normal_mean=self.logit_normal_mean,
            logit_normal_std=self.logit_normal_std,
            device=device,
        )

    def _loss_weights(self, timesteps: torch.Tensor, target_ndim: int, dtype: torch.dtype) -> torch.Tensor:
        if self.loss_weighting == "uniform":
            weights = torch.ones_like(timesteps, dtype=torch.float32)
        elif self.loss_weighting == "min_snr":
            snr = self.base_snr.to(timesteps.device).index_select(0, timesteps.long())
            weights = snr.clamp(max=self.snr_clip)
        elif self.loss_weighting == "min_snr_gamma":
            snr = self.base_snr.to(timesteps.device).index_select(0, timesteps.long())
            weights = snr.clamp(min=0.0, max=self.snr_clip)
        else:
            raise ValueError(f"Unsupported loss_weighting: {self.loss_weighting}")
        weights = weights.to(dtype=dtype)
        while weights.ndim < target_ndim:
            weights = weights.unsqueeze(-1)
        return weights

    def _q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_a = _extract(self.base_sqrt_alphas_cumprod, t, x_start.ndim).to(x_start.dtype)
        sqrt_om = _extract(self.base_sqrt_one_minus_alphas_cumprod, t, x_start.ndim).to(x_start.dtype)
        return sqrt_a * x_start + sqrt_om * noise

    def training_losses(
        self,
        denoiser: nn.Module,
        x_start: torch.Tensor,
        conditions: dict,
        valid_mask: torch.Tensor,
        history_len: int = 0,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x_start.shape
        device = x_start.device
        train_index = self._sample_train_indices(batch_size, device)
        timesteps = self.train_timestep_map.to(device).index_select(0, train_index)
        noise = torch.randn_like(x_start)
        x_t = self._q_sample(x_start, timesteps, noise)
        t_seq = timesteps[:, None].expand(-1, seq_len)
        raw = denoiser(x_t, t_seq, conditions, valid_mask=valid_mask)
        sqrt_a = _extract(self.base_sqrt_alphas_cumprod, timesteps, x_start.ndim).to(x_start.dtype)
        sqrt_om = _extract(self.base_sqrt_one_minus_alphas_cumprod, timesteps, x_start.ndim).to(x_start.dtype)
        pred_x0 = prediction_to_x0(self.objective, raw, x_t, noise, sqrt_a, sqrt_om)
        target = objective_target(self.objective, x_start, noise)

        loss_mask = valid_mask.bool()
        if history_len > 0:
            steps = torch.arange(seq_len, device=device)
            loss_mask = loss_mask & (steps.unsqueeze(0) >= int(history_len))
        loss = (raw - target).pow(2)
        loss = loss * self._loss_weights(timesteps, loss.ndim, loss.dtype)
        per_frame = loss.mean(dim=-1)
        mask = loss_mask.to(per_frame.dtype)
        per_sample = (per_frame * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return {
            "diffusion_loss": per_sample.mean(),
            "pred_x0": pred_x0,
            "raw_prediction": raw,
            "target": target,
            "x_t": x_t,
            "timesteps": timesteps,
        }

    def _predict_x0_with_cfg(
        self,
        denoiser: nn.Module,
        x: torch.Tensor,
        t_index: torch.Tensor,
        conditions: dict,
        valid_mask: torch.Tensor,
        null_conditions: dict | None,
        cfg_scale: float,
        cfg_branches: list[tuple[dict, float]] | None = None,
    ) -> torch.Tensor:
        model_t = self.sample_timestep_map.to(t_index.device).index_select(0, t_index)
        t_seq = model_t[:, None].expand(-1, x.shape[1])
        if cfg_branches is not None:
            if null_conditions is None:
                raise ValueError("cfg_branches requires null_conditions")
            raw_null = denoiser(x, t_seq, null_conditions, valid_mask=valid_mask)
            raw = raw_null
            for branch_conditions, branch_scale in cfg_branches:
                if float(branch_scale) == 0.0:
                    continue
                raw_branch = denoiser(x, t_seq, branch_conditions, valid_mask=valid_mask)
                raw = raw + float(branch_scale) * (raw_branch - raw_null)
        else:
            raw = denoiser(x, t_seq, conditions, valid_mask=valid_mask)
            if null_conditions is not None and cfg_scale != 1.0:
                raw_null = denoiser(x, t_seq, null_conditions, valid_mask=valid_mask)
                raw = raw_null + float(cfg_scale) * (raw - raw_null)
        alpha_bar = _extract(self.sample_alphas_cumprod, t_index, x.ndim).to(x.dtype)
        sqrt_a = torch.sqrt(alpha_bar)
        sqrt_om = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-8))
        return prediction_to_x0(self.objective, raw, x, torch.zeros_like(x), sqrt_a, sqrt_om)

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        shape: tuple[int, int, int],
        conditions: dict,
        valid_mask: torch.Tensor | None = None,
        null_conditions: dict | None = None,
        cfg_scale: float | None = None,
        cfg_branches: list[tuple[dict, float]] | None = None,
    ) -> torch.Tensor:
        ref = _condition_tensor(conditions)
        batch_size, seq_len, _ = shape
        x = torch.randn(shape, device=ref.device, dtype=ref.dtype)
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=ref.device)
        else:
            valid_mask = valid_mask.to(device=ref.device, dtype=torch.bool)
        scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)

        for step in range(self.sample_alphas_cumprod.numel() - 1, -1, -1):
            t_index = torch.full((batch_size,), step, device=ref.device, dtype=torch.long)
            pred_x0 = self._predict_x0_with_cfg(
                denoiser,
                x,
                t_index,
                conditions,
                valid_mask,
                null_conditions,
                scale,
                cfg_branches=cfg_branches,
            )
            alpha_bar = _extract(self.sample_alphas_cumprod, t_index, x.ndim).to(x.dtype)
            alpha_bar_prev = _extract(self.sample_alphas_cumprod_prev, t_index, x.ndim).to(x.dtype)
            eps = (x - alpha_bar.sqrt() * pred_x0) / (1.0 - alpha_bar).sqrt().clamp_min(1e-8)
            sigma = (
                self.ddim_eta
                * torch.sqrt(((1.0 - alpha_bar_prev) / (1.0 - alpha_bar)).clamp_min(0.0))
                * torch.sqrt((1.0 - alpha_bar / alpha_bar_prev.clamp_min(1e-8)).clamp_min(0.0))
            )
            mean_pred = (
                pred_x0 * alpha_bar_prev.sqrt()
                + torch.sqrt((1.0 - alpha_bar_prev - sigma.pow(2)).clamp_min(0.0)) * eps
            )
            x = mean_pred + sigma * torch.randn_like(x) if step > 0 else mean_pred
            x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return x
