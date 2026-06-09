from __future__ import annotations

import torch


def batch_r_precision(
    logits: torch.Tensor,
    top_k: int = 3,
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    nearest = logits.argsort(dim=1, descending=True)[:, :top_k]
    if positive_mask is None:
        positive_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    hits_at_rank = torch.gather(positive_mask, dim=1, index=nearest)
    hits = torch.cummax(hits_at_rank.float(), dim=1).values
    return hits.mean(dim=0)
