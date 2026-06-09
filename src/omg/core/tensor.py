from __future__ import annotations

import torch


def get_valid_mask(max_len: int, valid_len: int, device="cpu") -> torch.Tensor:
    mask = torch.zeros(int(max_len), dtype=torch.bool, device=device)
    mask[: int(valid_len)] = True
    return mask


def repeat_to_max_len(x: torch.Tensor, max_len: int, dim: int = 0) -> torch.Tensor:
    if x.shape[dim] == max_len:
        return x
    if x.shape[dim] > max_len:
        raise ValueError(f"Unexpected length {x.shape[dim]} > {max_len}")
    moved = x.transpose(0, dim)
    pad = moved[-1:].expand(max_len - moved.shape[0], *moved.shape[1:])
    return torch.cat([moved, pad], dim=0).transpose(0, dim)
