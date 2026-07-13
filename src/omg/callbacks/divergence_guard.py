from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback


class DivergenceGuard(Callback):
    """Fail fast when training leaves a numerically healthy regime."""

    def __init__(
        self,
        dirpath: str,
        warmup_steps: int = 2000,
        max_loss: float | None = None,
        loss_monitor: str = "loss",
        loss_patience: int = 1,
        max_grad_norm: float | None = None,
        capture_loss_term_grad_norms: bool = False,
        norm_type: float = 2.0,
    ) -> None:
        self.dirpath = Path(dirpath)
        self.warmup_steps = int(warmup_steps)
        self.max_loss = None if max_loss is None else float(max_loss)
        self.loss_monitor = str(loss_monitor)
        self.loss_patience = max(1, int(loss_patience))
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.norm_type = float(norm_type)
        self.capture_loss_term_grad_norms = bool(capture_loss_term_grad_norms)
        self._loss_strikes = 0

    def setup(self, trainer: Any, pl_module: torch.nn.Module, stage: str | None = None) -> None:
        if stage in {None, "fit"}:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            if self.capture_loss_term_grad_norms and self.max_grad_norm is not None:
                setattr(pl_module, "_capture_loss_term_grad_norms_for_divergence_guard", True)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: torch.nn.Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = int(getattr(trainer, "global_step", 0))
        if step < self.warmup_steps or self.max_loss is None:
            return
        diagnostics = getattr(pl_module, "_last_train_diagnostics", None) or {}
        loss = self._extract_loss(outputs, diagnostics)
        if loss is None:
            return
        if loss > self.max_loss:
            self._loss_strikes += 1
            event = self._event(trainer, "loss_threshold", batch_idx=batch_idx, loss=loss, diagnostics=diagnostics)
            self._write_event(event)
            if self._loss_strikes >= self.loss_patience:
                raise RuntimeError(
                    f"DivergenceGuard stopped training at step={step}: "
                    f"{self.loss_monitor}={loss:.6g} exceeded max_loss={self.max_loss:.6g} "
                    f"for {self._loss_strikes} checked train batches"
                )
        else:
            self._loss_strikes = 0

    def on_before_optimizer_step(self, trainer: Any, pl_module: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        step = int(getattr(trainer, "global_step", 0))
        if step < self.warmup_steps or self.max_grad_norm is None:
            return
        params = [param for param in pl_module.parameters() if param.grad is not None]
        grad_norm = self._grad_norm(params)
        if grad_norm is None or grad_norm <= self.max_grad_norm:
            return
        diagnostics = getattr(pl_module, "_last_train_diagnostics", None) or {}
        event = self._event(trainer, "grad_norm_threshold", grad_norm=grad_norm, diagnostics=diagnostics)
        self._write_event(event)
        raise RuntimeError(
            f"DivergenceGuard stopped training at step={step}: "
            f"grad_norm={grad_norm:.6g} exceeded max_grad_norm={self.max_grad_norm:.6g}"
        )

    def _extract_loss(self, outputs: Any, diagnostics: dict[str, Any]) -> float | None:
        value = diagnostics.get(self.loss_monitor)
        if value is None and self.loss_monitor == "loss" and isinstance(outputs, dict):
            value = outputs.get("loss")
        if torch.is_tensor(value):
            value = float(value.detach().float().cpu().item())
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else math.inf

    def _grad_norm(self, params: list[torch.nn.Parameter]) -> float | None:
        if not params:
            return 0.0
        device = params[0].grad.device
        norms = [torch.linalg.vector_norm(param.grad.detach(), ord=self.norm_type).to(device) for param in params]
        norm = torch.linalg.vector_norm(torch.stack(norms), ord=self.norm_type)
        value = float(norm.detach().float().cpu().item())
        return value if math.isfinite(value) else math.inf

    def _event(self, trainer: Any, reason: str, **payload: Any) -> dict[str, Any]:
        return {
            "reason": reason,
            "global_step": int(getattr(trainer, "global_step", 0)),
            "current_epoch": int(getattr(trainer, "current_epoch", 0)),
            "global_rank": int(getattr(trainer, "global_rank", 0)),
            **self._sanitize(payload),
        }

    def _sanitize(self, value: Any) -> Any:
        if torch.is_tensor(value):
            if value.ndim == 0:
                item = float(value.detach().float().cpu().item())
                return item if math.isfinite(item) else None
            return self._sanitize(value.detach().cpu().tolist())
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(key): self._sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize(item) for item in value]
        return str(value)

    def _write_event(self, event: dict[str, Any]) -> None:
        rank = int(event.get("global_rank", 0))
        path = self.dirpath / f"divergence_guard_rank{rank}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
