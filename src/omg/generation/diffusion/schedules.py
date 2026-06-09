from __future__ import annotations

import math

import torch


def enforce_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    betas64 = betas.to(dtype=torch.float64)
    alphas = 1.0 - betas64
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    first = sqrt_alphas_cumprod[0].clone()
    last = sqrt_alphas_cumprod[-1].clone()
    denom = (first - last).clamp_min(1e-12)
    sqrt_alphas_cumprod = (sqrt_alphas_cumprod - last) * first / denom
    alphas_cumprod = sqrt_alphas_cumprod.square()
    alphas = torch.cat([alphas_cumprod[:1], alphas_cumprod[1:] / alphas_cumprod[:-1].clamp_min(1e-12)])
    return (1.0 - alphas).clamp(0.0, 1.0).to(dtype=betas.dtype)


def make_beta_schedule(
    name: str,
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    zero_terminal_snr: bool = False,
) -> torch.Tensor:
    if name == "linear":
        betas = torch.linspace(beta_start, beta_end, int(timesteps), dtype=torch.float32)
    elif name == "cosine":
        steps = torch.arange(int(timesteps) + 1, dtype=torch.float64)
        alphas = torch.cos(((steps / int(timesteps)) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - (alphas[1:] / alphas[:-1])
        betas = betas.clamp(1e-5, 0.999).float()
    else:
        raise ValueError(f"Unknown beta schedule: {name}")
    if zero_terminal_snr:
        betas = enforce_zero_terminal_snr(betas)
    return betas


def sample_timesteps(
    shape: tuple[int, ...],
    num_timesteps: int,
    *,
    strategy: str = "uniform",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if strategy == "uniform":
        return torch.randint(0, int(num_timesteps), shape, device=device)
    if strategy == "logit_normal":
        values = torch.randn(shape, device=device) * float(logit_normal_std) + float(logit_normal_mean)
        timesteps = (torch.sigmoid(values) * int(num_timesteps)).long()
        return timesteps.clamp(max=int(num_timesteps) - 1)
    raise ValueError(f"Unsupported timestep_sampling strategy: {strategy}")
