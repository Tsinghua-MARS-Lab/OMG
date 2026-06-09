from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback


class TrainingDiagnosticsLogger(Callback):
    def __init__(
        self,
        dirpath: str,
        train_log_every_n_steps: int = 10,
        grad_log_every_n_steps: int = 10,
        max_batch_meta: int = 128,
        norm_type: float = 2.0,
        module_prefixes: Iterable[str] | None = None,
    ) -> None:
        self.dirpath = Path(dirpath)
        self.train_log_every_n_steps = int(train_log_every_n_steps)
        self.grad_log_every_n_steps = int(grad_log_every_n_steps)
        self.max_batch_meta = int(max_batch_meta)
        self.norm_type = float(norm_type)
        self.module_prefixes = tuple(module_prefixes or ())
        self._file = None

    def setup(self, trainer: Any, pl_module: torch.nn.Module, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        rank = int(getattr(trainer, "global_rank", 0))
        path = self.dirpath / f"training_diagnostics_rank{rank}.jsonl"
        self._file = path.open("a", encoding="utf-8")

    def teardown(self, trainer: Any, pl_module: torch.nn.Module, stage: str | None = None) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: torch.nn.Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = int(getattr(trainer, "global_step", 0))
        if self.train_log_every_n_steps > 1 and step % self.train_log_every_n_steps != 0:
            return
        diagnostics = getattr(pl_module, "_last_train_diagnostics", None)
        if diagnostics is None:
            return
        event = self._base_event(trainer, pl_module, "train_batch")
        event["batch_idx"] = int(batch_idx)
        event.update(self._sanitize(diagnostics))
        self._write(event)

    def on_before_optimizer_step(self, trainer: Any, pl_module: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        step = int(getattr(trainer, "global_step", 0))
        if self.grad_log_every_n_steps > 1 and step % self.grad_log_every_n_steps != 0:
            return
        params = [(name, param) for name, param in pl_module.named_parameters() if param.grad is not None]
        grad_norms = {"global": self._to_float(self._grad_norm([param for _, param in params]))}
        for prefix in self.module_prefixes:
            group = [param for name, param in params if name == prefix or name.startswith(f"{prefix}.")]
            if group:
                grad_norms[prefix.replace(".", "/")] = self._to_float(self._grad_norm(group))
        event = self._base_event(trainer, pl_module, "grad_norm")
        event["grad_norm"] = grad_norms
        event["optimizer_lrs"] = self._optimizer_lrs(optimizer)
        latest = getattr(pl_module, "_last_train_diagnostics", None)
        if latest is not None:
            event["latest_train"] = self._sanitize(latest)
        self._write(event)

    def _base_event(self, trainer: Any, pl_module: torch.nn.Module, event_type: str) -> dict[str, Any]:
        return {
            "event": event_type,
            "global_step": int(getattr(trainer, "global_step", 0)),
            "current_epoch": int(getattr(trainer, "current_epoch", 0)),
            "global_rank": int(getattr(trainer, "global_rank", 0)),
            "local_rank": int(getattr(trainer, "local_rank", 0)),
            "world_size": int(getattr(trainer, "world_size", 1)),
            "training": bool(getattr(pl_module, "training", False)),
        }

    def _optimizer_lrs(self, optimizer: torch.optim.Optimizer) -> list[float | None]:
        return [self._finite_or_none(float(group.get("lr", 0.0))) for group in optimizer.param_groups]

    def _grad_norm(self, params: list[torch.nn.Parameter]) -> torch.Tensor:
        if not params:
            return torch.tensor(0.0)
        device = params[0].grad.device
        norms = [torch.linalg.vector_norm(param.grad.detach(), ord=self.norm_type).to(device) for param in params]
        return torch.linalg.vector_norm(torch.stack(norms), ord=self.norm_type)

    def _to_float(self, value: torch.Tensor | float | int) -> float | None:
        if torch.is_tensor(value):
            value = float(value.detach().float().cpu().item())
        return self._finite_or_none(float(value))

    def _finite_or_none(self, value: float) -> float | None:
        return value if math.isfinite(value) else None

    def _sanitize(self, value: Any) -> Any:
        if torch.is_tensor(value):
            if value.ndim == 0:
                return self._to_float(value)
            return self._sanitize(value.detach().cpu().tolist())
        if isinstance(value, float):
            return self._finite_or_none(value)
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(key): self._sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize(item) for item in value[: self.max_batch_meta]]
        return str(value)

    def _write(self, event: dict[str, Any]) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        self._file.flush()
