from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    """Build a fixed absolute position encoding for a token sequence."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError(f"Position-encoding dimension must be positive, got {self.dim}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape (B,T,D), got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.dim:
            raise ValueError(
                f"Token dimension {tokens.shape[-1]} does not match position-encoding dimension {self.dim}"
            )

        length = int(tokens.shape[1])
        if length == 0:
            return tokens

        position = torch.arange(length, device=tokens.device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=tokens.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.dim)
        )
        encoding = torch.zeros(length, self.dim, device=tokens.device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        if self.dim > 1:
            encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        return encoding.to(dtype=tokens.dtype).unsqueeze(0).expand(tokens.shape[0], -1, -1)
