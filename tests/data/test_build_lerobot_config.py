from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml

from omg.cli.data.build_lerobot_config import build_config


def test_build_config_preserves_available_splits(tmp_path: Path) -> None:
    dataset = tmp_path / "source" / "toy"
    (dataset / "g1").mkdir(parents=True)
    (dataset / "labels").mkdir()
    (dataset / "info.yaml").write_text(
        yaml.safe_dump({"train": {"clip_a": 1}, "val": {"clip_b": 1}, "test": {"clip_c": 1}}),
        encoding="utf-8",
    )
    config = build_config(
        Namespace(
            source_root=dataset.parent,
            exclude_names=[],
            exclude_prefixes=[],
            default_excludes=True,
            require_labels=True,
            sequence_duration=2.0,
            fps=30.0,
            skip_missing_labels=True,
        )
    )

    assert config["metadata"]["split_datasets"] == {"train": 1, "val": 1, "test": 1}
    for split in ("train", "val", "test"):
        item = config["dataset_opts"][split][f"toy_{split}"]
        assert item["split"] == split
        assert item["labels_root"] == str(dataset / "labels")
