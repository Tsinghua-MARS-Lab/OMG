from __future__ import annotations

from pathlib import Path

import yaml

from omg.data.split_yaml import (
    dataset_motion_npz_path,
    flatten_dataset_split_paths,
    resolve_dataset_caption_txt_path,
)


def test_flatten_flat_path_dict_keys() -> None:
    data = yaml.safe_load(
        """
train:
  folder4/279303_retarget: 1
  folder5/304659_retarget: 1
"""
    )["train"]
    assert flatten_dataset_split_paths(data) == ["folder4/279303_retarget", "folder5/304659_retarget"]


def test_flatten_nested_dict_legacy() -> None:
    data = {"folder4": {"279303_retarget": 1}, "folder5": {"304659_retarget": 1}}
    assert flatten_dataset_split_paths(data) == ["folder4/279303_retarget", "folder5/304659_retarget"]


def test_dataset_motion_npz_path_joins_relative_entry(tmp_path: Path) -> None:
    g1 = tmp_path / "g1"
    g1.mkdir()
    rel = Path("fit3d/train/s01/x_retarget.npz")
    (g1 / rel.parent).mkdir(parents=True)
    (g1 / rel).write_bytes(b"")
    p = dataset_motion_npz_path(g1, "fit3d/train/s01/x_retarget")
    assert p == g1 / rel


def test_resolve_caption_txt_parallel_tree(tmp_path: Path) -> None:
    texts = tmp_path / "texts"
    (texts / "folder4").mkdir(parents=True)
    (texts / "folder4" / "279303.txt").write_text("cap\n", encoding="utf-8")
    got = resolve_dataset_caption_txt_path("folder4/279303_retarget", texts)
    assert got == texts / "folder4" / "279303.txt"


def test_resolve_caption_txt_missing_returns_none_without_tree_scan(tmp_path: Path) -> None:
    texts = tmp_path / "texts"
    nested = texts / "other" / "deep"
    nested.mkdir(parents=True)
    (nested / "279303.txt").write_text("cap\n", encoding="utf-8")
    assert resolve_dataset_caption_txt_path("folder4/279303_retarget", texts) is None
