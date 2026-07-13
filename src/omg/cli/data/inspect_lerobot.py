from __future__ import annotations

import argparse
import json
from pathlib import Path

from omg.data.lerobot_inspect import inspect_lerobot


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect an official OMG LeRobotDataset v3 export.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = inspect_lerobot(args.root)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
