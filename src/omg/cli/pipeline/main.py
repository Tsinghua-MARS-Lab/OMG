from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from omg.pipeline import DEFAULT_TRACKER_ONNX_PROVIDERS_CSV, OnnxDiffusionPlanner
from omg.tracking.holomotion.io import load_reference_motion
from omg.tracking.holomotion.reference import resample_qpos

from omg.cli.pipeline.modes import (
    _run_async,
    _run_diffusion_only,
    _run_offline_track,
    _run_sync,
    _run_tracker_only,
)
from omg.pipeline.audio import load_pipeline_audio_features, load_pipeline_realtime_audio_wav
from omg.pipeline.condition_sequence import (
    condition_sequence_text,
    load_pipeline_condition_sequence,
)
from omg.pipeline.human import load_pipeline_human_motion
from omg.cli.pipeline.utils import HOLOMOTION_TRACKER_FPS, _output_name


def _diffusion_tensorrt_fp16(args: argparse.Namespace) -> bool:
    if args.tensorrt_fp16 is not None:
        return bool(args.tensorrt_fp16)
    return args.mode == "async"


def _diffusion_tensorrt_cache_path(args: argparse.Namespace, output_dir: Path) -> Path | None:
    if args.tensorrt_engine_cache_path is not None:
        return Path(args.tensorrt_engine_cache_path).expanduser()
    if args.mode == "async":
        return output_dir / "tensorrt_engine_cache"
    return None


def _diffusion_dit_cache(args: argparse.Namespace) -> bool:
    if args.dit_cache is not None:
        return bool(args.dit_cache)
    return args.mode == "async"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OMG diffusion planning and optional HoloMotion tracking."
    )
    parser.add_argument("--mode", choices=["diffusion-only", "tracker-only", "sync", "async", "offline-track"], default="diffusion-only")
    parser.add_argument("--diffusion-onnx", default=None, help="Exported OMG denoiser step ONNX. Required unless --mode tracker-only.")
    parser.add_argument("--seed-motion", required=True, help="Seed qpos_36 motion .npy/.npz/.pt used for history frames.")
    parser.add_argument("--seed-fps", type=float, default=None, help="FPS for seed .npy or override for seed file metadata.")
    parser.add_argument("--target-fps", type=float, default=None, help="Diffusion planning FPS. Defaults to seed FPS. Ignored by tracker-only.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=60,
        help=(
            "diffusion-only/offline-track: generated diffusion frames; "
            "tracker-only/sync/async: total tracker-executed frames."
        ),
    )
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--cfg-text-scale", type=float, default=None, help="Text CFG scale. If any per-modality CFG scale is set, omitted modality scales default to 0.")
    parser.add_argument("--cfg-audio-scale", type=float, default=None, help="Audio CFG scale. If any per-modality CFG scale is set, omitted modality scales default to 0.")
    parser.add_argument("--cfg-human-scale", type=float, default=None, help="Human-reference CFG scale. If any per-modality CFG scale is set, omitted modality scales default to 0.")
    audio_group = parser.add_mutually_exclusive_group(required=False)
    audio_group.add_argument("--audio-wav", default=None, help="Raw wav audio condition for audio-conditioned diffusion ONNX.")
    audio_group.add_argument(
        "--realtime-audio-wav",
        default=None,
        help="Raw wav audio condition for async mode, sliced by tracker execution time at each replan.",
    )
    audio_group.add_argument("--audio-features", default=None, help="Precomputed frame-level audio feature .npy, shape (T,35).")
    parser.add_argument(
        "--audio-feature-type",
        choices=["current35"],
        default="current35",
        help="Feature extractor used for --audio-wav. Must match the training/exported model convention.",
    )
    parser.add_argument(
        "--human-motion",
        default=None,
        help="Optional human reference condition path: .npy/.npz with shape (T,D), (T,J,3), or keys human_motion/human_joints/joints/poses.",
    )
    parser.add_argument(
        "--diffusion-continuation-steps",
        type=int,
        default=0,
        help="Final DDIM denoising steps that constrain the overlap prefix during replanning. Set 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--providers",
        default=None,
        help="ONNX Runtime providers for diffusion ONNX. Defaults to TensorRT then CUDA for TensorRT-compatible exports; otherwise CUDA.",
    )
    parser.add_argument(
        "--tensorrt-fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable TensorRT FP16 for diffusion ONNX. Defaults to enabled for --mode async and disabled otherwise.",
    )
    parser.add_argument(
        "--tensorrt-engine-cache-path",
        default=None,
        help="TensorRT diffusion engine/timing cache directory. Defaults under the async output directory.",
    )
    parser.add_argument(
        "--dit-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable DreamZero-style DiT cache step skipping. Defaults to enabled for --mode async and disabled otherwise.",
    )
    parser.add_argument("--dit-cache-threshold", type=float, default=0.995)
    parser.add_argument("--dit-cache-warmup-steps", type=int, default=4)
    parser.add_argument("--dit-cache-max-consecutive", type=int, default=2)
    parser.add_argument("--torch-device", default="auto", help="Device for T5 and G1 representation: auto, cpu, cuda.")
    parser.add_argument("--text-encoder-model", default=None, help="Override text encoder model/path from ONNX metadata.")
    parser.add_argument(
        "--compile-history-encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compile the planner history encoder with torch.compile. Defaults to enabled on CUDA.",
    )
    parser.add_argument("--text", default=None)
    parser.add_argument(
        "--condition-sequence",
        default=None,
        help=(
            "Per-diffusion-chunk modality sequence separated by '|'. "
            "Supported chunks: text: prompt, audio: /path/to.wav, humanref: /path/to.npy-or.npz, "
            "and compositions such as text+audio: prompt+/path/to.wav. Audio chunks without [N] auto-expand "
            "to the wav duration. Chunks without a prefix are treated as text."
        ),
    )
    parser.add_argument(
        "--audio-type",
        choices=["audio", "feature"],
        default="audio",
        help=(
            "Condition-sequence audio mode. audio slices wav at replan time; "
            "feature precomputes current35 features from wav at startup and slices the feature timeline."
        ),
    )

    parser.add_argument("--holomotion-onnx", default=None, help="Required for --mode tracker-only, sync, async, or offline-track.")
    parser.add_argument("--robot-xml", default=None)
    parser.add_argument(
        "--tracker-providers",
        default=DEFAULT_TRACKER_ONNX_PROVIDERS_CSV,
        help="ONNX Runtime providers for HoloMotion. Defaults to TensorRT, then CUDA.",
    )
    parser.add_argument("--control-substeps", type=int, default=10)
    parser.add_argument("--action-clip", type=float, default=10.0)
    parser.add_argument(
        "--async-latency-frames",
        type=int,
        default=None,
        help="Optional tracker-frame latency for deterministic async simulation. Defaults to measured diffusion planning time.",
    )
    parser.add_argument(
        "--async-replan-remaining-frames",
        type=int,
        default=40,
        help="Start async replanning when the reference buffer has at most this many unexecuted tracker frames.",
    )

    parser.add_argument("--output-root", default="outputs_pipeline")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--video", action="store_true", help="diffusion-only: render reference; tracker-only/sync/async/offline-track: render tracker execution.")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--camera-view", choices=["iso", "side", "front", "back", "both"], default="side")
    parser.add_argument("--follow-mode", choices=["none", "xy", "xyz", "heading"], default="xy")
    parser.add_argument("--camera-distance", type=float, default=4.0)
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    parser.add_argument("--scene-preset", choices=["minimal", "studio"], default="studio")
    parser.add_argument("--title", default="OMG Diffusion")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reference = load_reference_motion(args.seed_motion, fps=args.seed_fps)
    output_dir = Path(args.output_root) / _output_name(args)
    if args.mode == "tracker-only":
        _run_tracker_only(
            args,
            reference_qpos=reference.qpos_36,
            reference_fps=float(reference.fps),
            output_dir=output_dir,
        )
        return

    if args.diffusion_onnx is None:
        raise ValueError("--diffusion-onnx is required unless --mode tracker-only")
    target_fps = float(reference.fps if args.target_fps is None else args.target_fps)
    seed_qpos = reference.qpos_36
    if target_fps != float(reference.fps):
        seed_qpos = resample_qpos(seed_qpos, reference.fps, target_fps)
    if args.condition_sequence is not None:
        if args.text is not None:
            raise ValueError("--condition-sequence cannot be combined with --text")
        if args.audio_wav is not None or args.realtime_audio_wav is not None or args.audio_features is not None:
            raise ValueError("--condition-sequence cannot be combined with global audio condition arguments")
        if args.human_motion is not None:
            raise ValueError("--condition-sequence cannot be combined with --human-motion")

    planner = OnnxDiffusionPlanner(
        args.diffusion_onnx,
        providers=args.providers,
        text_encoder_model=args.text_encoder_model,
        torch_device=args.torch_device,
        seed=args.seed,
        tensorrt_fp16=_diffusion_tensorrt_fp16(args),
        tensorrt_engine_cache_path=_diffusion_tensorrt_cache_path(args, output_dir),
        dit_cache=_diffusion_dit_cache(args),
        dit_cache_threshold=args.dit_cache_threshold,
        dit_cache_warmup_steps=args.dit_cache_warmup_steps,
        dit_cache_max_consecutive=args.dit_cache_max_consecutive,
        compile_history_encoder=args.compile_history_encoder,
    )
    args.plan_text_prompts = ["" if args.text is None else str(args.text).strip()]
    text = args.plan_text_prompts[0]
    if args.condition_sequence is not None:
        condition_audio_step_frames = int(planner.sequence_length)
        if args.mode == "async":
            tracker_plan_horizon = int(
                round(float(planner.sequence_length) * float(HOLOMOTION_TRACKER_FPS) / float(target_fps))
            )
            replan_interval = max(1, tracker_plan_horizon - int(args.async_replan_remaining_frames))
            condition_audio_step_frames = max(1, int(replan_interval * float(target_fps) / float(HOLOMOTION_TRACKER_FPS)))
        condition_sequence = load_pipeline_condition_sequence(
            args.condition_sequence,
            planner=planner,
            target_fps=target_fps,
            audio_feature_type=args.audio_feature_type,
            num_frames_per_chunk=int(planner.sequence_length),
            audio_step_frames=condition_audio_step_frames,
            audio_type=args.audio_type,
        )
        args.condition_sequence_chunks = condition_sequence
        plan_texts = [
            condition_sequence_text(condition_sequence, "", plan_index)
            for plan_index in range(len(condition_sequence))
        ]
        args.plan_text_prompts = plan_texts
        text = plan_texts[0] if plan_texts else ""
    audio_features = None
    human_motion = None
    if args.realtime_audio_wav is not None and args.mode not in {"sync", "async"}:
        raise ValueError("--realtime-audio-wav currently requires --mode sync or --mode async")
    if args.audio_wav is not None or args.realtime_audio_wav is not None or args.audio_features is not None:
        if not planner.use_audio:
            raise ValueError("--audio-wav/--realtime-audio-wav/--audio-features requires an audio-conditioned diffusion ONNX")
        realtime_audio = args.realtime_audio_wav is not None
        if realtime_audio:
            audio_features = load_pipeline_realtime_audio_wav(
                args.realtime_audio_wav,
                fps=target_fps,
                audio_dim=planner.audio_dim,
            )
        else:
            audio_path = args.audio_wav if args.audio_wav is not None else args.audio_features
            audio_source_type = "wav" if args.audio_wav is not None else "feature"
            audio_num_frames = max(int(args.num_frames), int(planner.sequence_length))
            if args.mode == "async":
                tracker_plan_horizon = int(
                    round(float(planner.sequence_length) * float(HOLOMOTION_TRACKER_FPS) / float(target_fps))
                )
                replan_interval = max(1, int(tracker_plan_horizon) - int(args.async_replan_remaining_frames))
                required_plans = int(np.ceil(float(args.num_frames) / float(replan_interval))) + 2
                audio_num_frames = max(audio_num_frames, required_plans * int(planner.sequence_length))
            audio_features = load_pipeline_audio_features(
                audio_path,
                source_type=audio_source_type,
                fps=target_fps,
                audio_dim=planner.audio_dim,
                num_frames=audio_num_frames,
                feature_type=args.audio_feature_type,
            )
    if args.human_motion is not None:
        if not planner.use_human_motion:
            raise ValueError("--human-motion requires a human-reference-conditioned diffusion ONNX")
        human_motion = load_pipeline_human_motion(
            args.human_motion,
            fps=target_fps,
            human_motion_dim=planner.human_motion_dim,
            num_frames=max(int(args.num_frames), int(planner.sequence_length)),
        )
    reference_path = output_dir / "reference_motion.npz"
    if args.mode == "diffusion-only":
        _run_diffusion_only(
            args,
            planner=planner,
            seed_qpos=seed_qpos,
            target_fps=target_fps,
            text=text,
            output_dir=output_dir,
            reference_path=reference_path,
            audio_features=audio_features,
            human_motion=human_motion,
        )
    elif args.mode == "sync":
        _run_sync(
            args,
            planner=planner,
            seed_qpos=seed_qpos,
            target_fps=target_fps,
            text=text,
            output_dir=output_dir,
            reference_path=reference_path,
            audio_features=audio_features,
            human_motion=human_motion,
        )
    elif args.mode == "offline-track":
        _run_offline_track(
            args,
            planner=planner,
            seed_qpos=seed_qpos,
            target_fps=target_fps,
            text=text,
            output_dir=output_dir,
            reference_path=reference_path,
            audio_features=audio_features,
            human_motion=human_motion,
        )
    else:
        _run_async(
            args,
            planner=planner,
            seed_qpos=seed_qpos,
            target_fps=target_fps,
            text=text,
            output_dir=output_dir,
            reference_path=reference_path,
            audio_features=audio_features,
            human_motion=human_motion,
        )

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
