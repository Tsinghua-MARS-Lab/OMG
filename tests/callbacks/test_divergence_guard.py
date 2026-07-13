from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from omg.callbacks.divergence_guard import DivergenceGuard


class TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self._last_train_diagnostics = {
            "loss": 0.25,
            "diffusion_loss": 1.5,
            "loss_term_grad_norms": {
                "diffusion_loss": 1.5,
                "fk_body_pos_loss": 12.0,
                "motion_loss": 13.0,
            },
        }


def test_grad_norm_guard_event_records_latest_loss_term_grad_norms(tmp_path):
    guard = DivergenceGuard(
        dirpath=str(tmp_path),
        warmup_steps=0,
        max_grad_norm=1.0,
        capture_loss_term_grad_norms=True,
    )
    module = TinyModule()
    trainer = SimpleNamespace(global_step=7, current_epoch=2, global_rank=0)

    guard.setup(trainer, module, stage="fit")
    assert module._capture_loss_term_grad_norms_for_divergence_guard is True

    for param in module.parameters():
        param.grad = torch.full_like(param, 10.0)

    with pytest.raises(RuntimeError, match="grad_norm"):
        guard.on_before_optimizer_step(trainer, module, torch.optim.SGD(module.parameters(), lr=0.1))

    path = tmp_path / "divergence_guard_rank0.jsonl"
    event = json.loads(path.read_text(encoding="utf-8").strip())

    assert event["reason"] == "grad_norm_threshold"
    assert event["global_step"] == 7
    assert event["diagnostics"]["loss_term_grad_norms"] == {
        "diffusion_loss": 1.5,
        "fk_body_pos_loss": 12.0,
        "motion_loss": 13.0,
    }


def test_grad_norm_guard_leaves_loss_term_capture_disabled_by_default(tmp_path):
    guard = DivergenceGuard(
        dirpath=str(tmp_path),
        warmup_steps=0,
        max_grad_norm=1.0,
    )
    module = TinyModule()

    guard.setup(SimpleNamespace(global_step=0, current_epoch=0, global_rank=0), module, stage="fit")

    assert not hasattr(module, "_capture_loss_term_grad_norms_for_divergence_guard")


def test_grad_norm_guard_can_leave_loss_term_capture_disabled(tmp_path):
    guard = DivergenceGuard(
        dirpath=str(tmp_path),
        warmup_steps=0,
        max_grad_norm=1.0,
        capture_loss_term_grad_norms=False,
    )
    module = TinyModule()

    guard.setup(SimpleNamespace(global_step=0, current_epoch=0, global_rank=0), module, stage="fit")

    assert not hasattr(module, "_capture_loss_term_grad_norms_for_divergence_guard")


def test_loss_guard_can_monitor_scale_invariant_component(tmp_path):
    guard = DivergenceGuard(
        dirpath=str(tmp_path),
        warmup_steps=0,
        max_loss=1.0,
        loss_monitor="diffusion_loss",
    )
    module = TinyModule()
    trainer = SimpleNamespace(global_step=7, current_epoch=2, global_rank=0)

    with pytest.raises(RuntimeError, match="diffusion_loss=1.5"):
        guard.on_train_batch_end(trainer, module, {"loss": torch.tensor(0.25)}, None, 3)

    event = json.loads((tmp_path / "divergence_guard_rank0.jsonl").read_text(encoding="utf-8").strip())
    assert event["reason"] == "loss_threshold"
    assert event["loss"] == 1.5
