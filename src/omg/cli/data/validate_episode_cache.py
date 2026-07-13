from __future__ import annotations

import argparse
import json
from pathlib import Path

from omg.data.episode_cache_inspect import inspect_episode_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an OMG frame-level episode cache.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = {split: inspect_episode_cache(args.root, split) for split in args.splits}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not all(item["valid"] for item in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
