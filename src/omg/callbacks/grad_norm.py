from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from pytorch_lightning import Callback


class GradNormMonitor(Callback):
    def __init__(
        self,
        norm_type: float = 2.0,
        log_every_n_steps: int = 1,
        module_prefixes: Iterable[str] | None = None,
    ) -> None:
        self.norm_type = float(norm_type)
        self.log_every_n_steps = int(log_every_n_steps)
        self.module_prefixes = tuple(module_prefixes or ())

    def on_before_optimizer_step(self, trainer: Any, pl_module: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        step = int(getattr(trainer, "global_step", 0))
        if self.log_every_n_steps > 1 and step % self.log_every_n_steps != 0:
            return
        params = [(name, param) for name, param in pl_module.named_parameters() if param.grad is not None]
        metrics = {"train/grad_norm/global": self._grad_norm([param for _, param in params])}
        for prefix in self.module_prefixes:
            group = [param for name, param in params if name == prefix or name.startswith(f"{prefix}.")]
            if group:
                metrics[f"train/grad_norm/{prefix.replace('.', '/')}"] = self._grad_norm(group)
        pl_module.log_dict(metrics, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)

    def _grad_norm(self, params: list[torch.nn.Parameter]) -> torch.Tensor:
        if not params:
            return torch.tensor(0.0)
        device = params[0].grad.device
        norms = [torch.linalg.vector_norm(param.grad.detach(), ord=self.norm_type).to(device) for param in params]
        return torch.linalg.vector_norm(torch.stack(norms), ord=self.norm_type)
