from __future__ import annotations

import argparse
import json
from pathlib import Path

from omg.data.lerobot_export import export_lerobot


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export unified OMG source datasets as LeRobotDataset v3.0.")
    parser.add_argument("--data-config", type=Path, default=Path("configs/generation/data/omg_data.yaml"))
    parser.add_argument("--representation-config", type=Path, default=Path("configs/generation/representation/125d.yaml"))
    parser.add_argument("--paths-config", type=Path, default=Path("configs/generation/paths/default.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="THU-MARS/OMG-Data")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional source dataset config keys to export.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames-per-file", type=int, default=2_000_000)
    parser.add_argument("--episodes-per-file", type=int, default=100_000)
    parser.add_argument("--read-workers", type=int, default=16)
    parser.add_argument("--max-entries-per-dataset", type=int, default=None)
    parser.add_argument("--max-episodes-per-dataset", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000, help="Print export progress every N episodes per dataset; set 0 to disable.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    summary = export_lerobot(build_argparser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
