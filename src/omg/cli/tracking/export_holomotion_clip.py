from __future__ import annotations

import argparse
import json

from omg.tracking.holomotion.export_clip import (
    HoloMotionClipExportConfig,
    export_holomotion_deployment_clip,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an OMG qpos_36 reference to HoloMotion deployment motion_data NPZ format."
    )
    parser.add_argument("--reference", required=True, help="Input .npy/.npz/.pt reference containing qpos_36")
    parser.add_argument("--reference-fps", type=float, default=None, help="Override input FPS when the reference file has no fps")
    parser.add_argument("--target-fps", type=float, default=50.0, help="Deployment clip FPS; HoloMotion deployment runs at 50Hz")
    parser.add_argument("--robot-xml", default=None, help="MuJoCo scene XML. Defaults to assets/holomotion/g1_29dof/scene_29dof.xml")
    parser.add_argument("--output", required=True, help="Output .npz path under the HoloMotion deployment motion_data directory")
    parser.add_argument("--metadata-json", default=None, help="Optional JSON object merged into output metadata")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    metadata = None
    if args.metadata_json is not None:
        metadata = json.loads(args.metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError("--metadata-json must decode to a JSON object")
    result = export_holomotion_deployment_clip(
        HoloMotionClipExportConfig(
            reference=args.reference,
            reference_fps=args.reference_fps,
            target_fps=args.target_fps,
            robot_xml=args.robot_xml,
            output=args.output,
            metadata=metadata,
        )
    )
    print(result.output_path)
    print(f"frames={result.frames} fps={result.fps:.3f}")


if __name__ == "__main__":
    main()
