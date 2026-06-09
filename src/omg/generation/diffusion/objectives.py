from __future__ import annotations

import torch


def prediction_to_x0(objective: str, raw: torch.Tensor, x_t: torch.Tensor, noise: torch.Tensor, sqrt_alpha: torch.Tensor, sqrt_one_minus_alpha: torch.Tensor) -> torch.Tensor:
    if objective == "pred_x0":
        return raw
    if objective == "pred_noise":
        return (x_t - sqrt_one_minus_alpha * raw) / sqrt_alpha.clamp_min(1e-8)
    raise ValueError(f"Unsupported objective: {objective}")


def objective_target(objective: str, x_start: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    if objective == "pred_x0":
        return x_start
    if objective == "pred_noise":
        return noise
    raise ValueError(f"Unsupported objective: {objective}")
