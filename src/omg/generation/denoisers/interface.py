from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseMotionDenoiser(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond_tokens: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
