#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an OMG-Data export with the official LeRobot loader.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--repo-id", default="THU-MARS/OMG-Data")
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--expected-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, download_videos=False)
    first = dataset[0]
    last = dataset[len(dataset) - 1]
    summary = {
        "repo_id": args.repo_id,
        "root": str(args.root.resolve()),
        "frames": len(dataset),
        "episodes": dataset.num_episodes,
        "features": sorted(dataset.features),
        "first_episode_index": int(first["episode_index"]),
        "last_episode_index": int(last["episode_index"]),
        "first_task": first["task"],
        "last_task": last["task"],
    }
    if args.expected_episodes is not None and summary["episodes"] != args.expected_episodes:
        raise ValueError(f"Expected {args.expected_episodes} episodes, got {summary['episodes']}")
    if args.expected_frames is not None and summary["frames"] != args.expected_frames:
        raise ValueError(f"Expected {args.expected_frames} frames, got {summary['frames']}")
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
