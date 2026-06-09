from __future__ import annotations

import argparse
from pathlib import Path

from omg.runtime.onnx_providers import DEFAULT_TRACKER_ONNX_PROVIDERS_CSV
from omg.tracking.holomotion.runner import TrackerRunConfig, run_holomotion_tracker


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HoloMotion tracker-only rollout for a G1 qpos_36 reference.")
    parser.add_argument("--mode", choices=("tracker-only",), default="tracker-only")
    parser.add_argument("--reference", required=True, help=".npy/.npz/.pt reference containing qpos_36")
    parser.add_argument("--reference-fps", type=float, default=None)
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--holomotion-onnx", required=True)
    parser.add_argument("--robot-xml", default=None)
    parser.add_argument("--providers", default=DEFAULT_TRACKER_ONNX_PROVIDERS_CSV, help="ONNX Runtime providers. Defaults to TensorRT, then CUDA.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--control-substeps", type=int, default=10)
    parser.add_argument("--action-clip", type=float, default=10.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--camera-view", choices=["iso", "side", "front", "back"], default="side")
    parser.add_argument("--follow-mode", choices=["none", "xy", "xyz", "heading"], default="xy")
    parser.add_argument("--camera-distance", type=float, default=4.0)
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    parser.add_argument("--overlay-text", default="")
    return parser


def run_tracker_only(args: argparse.Namespace) -> Path:
    result = run_holomotion_tracker(
        TrackerRunConfig(
            reference=args.reference,
            reference_fps=args.reference_fps,
            target_fps=float(args.target_fps),
            holomotion_onnx=args.holomotion_onnx,
            robot_xml=args.robot_xml,
            providers=args.providers,
            steps=args.steps,
            control_substeps=int(args.control_substeps),
            action_clip=float(args.action_clip),
            output=args.output,
            video=bool(args.video),
            video_path=args.video_path,
            video_width=int(args.video_width),
            video_height=int(args.video_height),
            camera_view=args.camera_view,
            follow_mode=args.follow_mode,
            camera_distance=float(args.camera_distance),
            camera_elevation=float(args.camera_elevation),
            overlay_text=args.overlay_text,
            mode="tracker-only",
            planner_mode=None,
        )
    )
    return result.rollout_path


def main() -> None:
    args = build_argparser().parse_args()
    output = run_tracker_only(args)
    print(output)


if __name__ == "__main__":
    main()
