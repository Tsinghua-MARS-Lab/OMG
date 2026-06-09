from __future__ import annotations

import time

from pytorch_lightning import Callback


class SpeedTimer(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapsed = time.time() - getattr(self, "_start", time.time())
        batch_size = batch.get("B", 1) if isinstance(batch, dict) else 1
        pl_module.log("train/step_seconds", elapsed, prog_bar=False, batch_size=batch_size)
