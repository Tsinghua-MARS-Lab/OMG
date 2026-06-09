from __future__ import annotations

from typing import Any

from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar, convert_inf


class GlobalStepProgressBar(TQDMProgressBar):
    """Show optimizer/global steps instead of raw dataloader batches."""

    def on_train_start(self, trainer, pl_module) -> None:
        super().on_train_start(trainer, pl_module)
        self._sync_train_bar_to_global_step(trainer, pl_module)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if self._leave:
            self.train_progress_bar = self.init_train_tqdm()
        self._sync_train_bar_to_global_step(trainer, pl_module)
        self.train_progress_bar.set_description(f"Global step")

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.train_progress_bar is None:
            return
        global_step = int(trainer.global_step)
        total = self.train_progress_bar.total
        if self._should_update(global_step, total):
            self._set_train_bar_n(global_step)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_validation_batch_start(
        self,
        trainer,
        pl_module,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return
        if self.val_progress_bar is None:
            return

        total = convert_inf(self.total_val_batches_current_dataloader)
        self.val_progress_bar.total = total
        self.val_progress_bar.initial = 0
        self.val_progress_bar.n = 0
        desc = self.sanity_check_description if trainer.sanity_checking else self.validation_description
        self.val_progress_bar.set_description(f"{desc} DataLoader {dataloader_idx}")
        self.val_progress_bar.refresh()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.train_progress_bar is not None and not self.train_progress_bar.disable:
            self._set_train_bar_n(int(trainer.global_step))
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
        if self._leave:
            self.train_progress_bar.close()

    def _sync_train_bar_to_global_step(self, trainer, pl_module) -> None:
        if self.train_progress_bar is None:
            return
        max_steps = trainer.max_steps
        total = None if max_steps is None or int(max_steps) < 0 else int(max_steps)
        # Lightning can expose the tqdm instance before tqdm.__init__ has fully
        # populated private fields used by reset(); set the public counters directly.
        self.train_progress_bar.total = total
        self.train_progress_bar.initial = int(trainer.global_step)
        self._set_train_bar_n(int(trainer.global_step))
        self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def _set_train_bar_n(self, n: int) -> None:
        if self.train_progress_bar is None:
            return
        self.train_progress_bar.n = n
        self.train_progress_bar.refresh()
