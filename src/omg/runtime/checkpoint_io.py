from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO


class DirectTorchCheckpointIO(CheckpointIO):
    """Checkpoint IO that lets torch stream directly to the target file.

    Lightning default checkpoint IO serializes the full checkpoint into a
    BytesIO buffer and then writes that buffer in one large call. Some network
    filesystems reject multi-GB single writes even when chunked writes succeed.
    This implementation preserves Lightning checkpoint semantics while using
    torch.save(path), which streams the archive directly to the filesystem.
    """

    def save_checkpoint(
        self,
        checkpoint: dict[str, Any],
        path: str | Path,
        storage_options: Any | None = None,
    ) -> None:
        if storage_options is not None:
            raise TypeError("DirectTorchCheckpointIO does not support storage_options")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(path)

    def load_checkpoint(
        self,
        path: str | Path,
        map_location: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return torch.load(path, map_location=map_location, **kwargs)

    def remove_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        if path.exists():
            path.unlink()
