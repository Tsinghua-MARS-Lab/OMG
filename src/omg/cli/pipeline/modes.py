from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from omg.pipeline import OnnxDiffusionPlanner, save_motion_plan
from omg.render.mujoco import render_qpos_video
from omg.tracking.holomotion.reference import resample_qpos
from omg.tracking.holomotion.runner import HoloMotionRolloutRunner

from omg.pipeline.audio import (
    PipelineAudioCondition,
    audio_features_for_plan,
    audio_features_for_timeline_frame,
    describe_pipeline_audio_features,
)
from omg.pipeline.human import (
    PipelineHumanMotion,
    describe_pipeline_human_motion,
    human_motion_for_plan,
)
from omg.pipeline.condition_sequence import (
    PipelineConditionSequence,
    condition_sequence_audio,
    condition_sequence_for_plan,
    condition_sequence_human,
    condition_sequence_text,
    describe_condition_sequence,
)
from omg.cli.pipeline.utils import (
    HOLOMOTION_TRACKER_FPS,
    ReplanMeanStats,
    _append_executed_history,
    _async_elapsed_tracker_frames,
    _resampled_frame_count,
    _save_plan_chunks,
    _save_tracker_reference,
    _source_cursor_from_tracker_cursor,
    _tracker_frames_from_latency,
)


def _motion_seed_frame_overlay(seed_frames: int, generated_frames: int) -> list[list[str]]:
    return [
        [
            "caption: motion seed",
            f"seed frame: {frame_idx + 1}/{seed_frames}",
        ]
        for frame_idx in range(seed_frames)
    ] + [
        [
            "caption: generated motion",
            f"generated frame: {frame_idx + 1}/{generated_frames}",
        ]
        for frame_idx in range(generated_frames)
    ]


def _motion_seed_qpos(seed_qpos: np.ndarray, history_frames: int) -> np.ndarray:
    if seed_qpos.shape[0] < history_frames:
        raise ValueError(f"Seed motion requires at least {history_frames} frames, got {seed_qpos.shape[0]}")
    return np.asarray(seed_qpos[-history_frames:], dtype=np.float32)


def _plan_texts(args: argparse.Namespace, fallback_text: str) -> list[str]:
    condition_sequence = _condition_sequence(args)
    if condition_sequence is not None:
        return [
            condition_sequence_text(condition_sequence, fallback_text, index)
            for index in range(len(condition_sequence))
        ]
    prompts = getattr(args, "plan_text_prompts", None)
    if prompts is None:
        return [fallback_text]
    prompts = [str(item).strip() for item in prompts]
    if not prompts:
        raise ValueError("Plan text list is empty")
    return prompts


def _condition_sequence(args: argparse.Namespace) -> PipelineConditionSequence | None:
    return getattr(args, "condition_sequence_chunks", None)


def _condition_modality_for_plan(args: argparse.Namespace, plan_index: int) -> str | None:
    chunk = condition_sequence_for_plan(_condition_sequence(args), int(plan_index))
    return None if chunk is None else chunk.modality


def _uses_condition_chunks(args: argparse.Namespace, fallback_text: str) -> bool:
    return len(_plan_texts(args, fallback_text)) > 1


def _text_for_plan(args: argparse.Namespace, fallback_text: str, plan_index: int) -> str:
    condition_sequence = _condition_sequence(args)
    if condition_sequence is not None:
        return condition_sequence_text(condition_sequence, fallback_text, int(plan_index))
    prompts = _plan_texts(args, fallback_text)
    index = int(plan_index)
    if len(prompts) == 1:
        return prompts[0]
    if index < 0 or index >= len(prompts):
        raise ValueError(
            f"Condition sequence has {len(prompts)} chunks, but diffusion plan index {index} was requested"
        )
    return prompts[index]


def _text_metadata(args: argparse.Namespace, fallback_text: str) -> dict[str, object]:
    condition_sequence = _condition_sequence(args)
    if condition_sequence is not None:
        texts = [
            condition_sequence_text(condition_sequence, fallback_text, index)
            for index in range(len(condition_sequence))
        ]
        return {
            "text": texts[0] if texts else "",
            "plan_texts": texts,
            "condition_chunks": len(texts),
            "condition_sequence": describe_condition_sequence(condition_sequence),
        }
    prompts = _plan_texts(args, fallback_text)
    if len(prompts) == 1:
        return {"text": prompts[0]}
    return {
        "text": prompts[0],
        "plan_texts": prompts,
        "condition_chunks": len(prompts),
    }


def _overlay_text(args: argparse.Namespace, fallback_text: str) -> str:
    condition_sequence = _condition_sequence(args)
    if condition_sequence is not None:
        labels = []
        for index, chunk in enumerate(condition_sequence):
            value = chunk.text if chunk.modality == "text" else str(chunk.source_path)
            labels.append(f"{index + 1}: {chunk.modality}: {value}")
        return " | ".join(labels)
    prompts = _plan_texts(args, fallback_text)
    if len(prompts) == 1:
        return prompts[0]
    return " | ".join(f"{idx + 1}: {prompt}" for idx, prompt in enumerate(prompts))


def _static_video_overlay_text(args: argparse.Namespace, fallback_text: str) -> str:
    return fallback_text if not _uses_condition_chunks(args, fallback_text) and _condition_sequence(args) is None else ""


def _tracker_frame_prompts(
    args: argparse.Namespace,
    fallback_text: str,
    *,
    tracker_frames: int,
    source_fps: float,
    tracker_fps: float,
    sequence_length: int,
) -> list[str]:
    prompts = _plan_texts(args, fallback_text)
    frame_prompts = []
    for tracker_frame in range(int(tracker_frames)):
        source_frame = int(np.floor(float(tracker_frame) * float(source_fps) / float(tracker_fps) + 1e-9))
        plan_index = min(source_frame // int(sequence_length), len(prompts) - 1)
        modality = _condition_modality_for_plan(args, plan_index)
        if modality is None or modality == "text":
            frame_prompts.append(prompts[plan_index])
        else:
            frame_prompts.append(modality)
    return frame_prompts


def _validate_condition_chunk_frames(
    args: argparse.Namespace,
    fallback_text: str,
    *,
    num_frames: int,
    sequence_length: int,
) -> None:
    prompts = _plan_texts(args, fallback_text)
    if len(prompts) <= 1:
        return
    expected = len(prompts) * int(sequence_length)
    if int(num_frames) != expected:
        raise ValueError(
            "Condition-sequence diffusion-only/offline-track requires one diffusion chunk per condition: "
            f"--num-frames must be {expected} for {len(prompts)} prompts and sequence_length={sequence_length}, "
            f"got {num_frames}"
        )


def _audio_condition_for_plan(
    args: argparse.Namespace,
    audio_features: PipelineAudioCondition | None,
    plan_index: int,
    *,
    request_tracker_frame: int | None,
    target_fps: float,
    num_frames: int,
    sequence_length: int,
    allow_multi_chunk: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    condition_sequence = _condition_sequence(args)
    if condition_sequence is not None:
        timeline_starts = getattr(args, "condition_sequence_audio_timeline_starts", None)
        if timeline_starts is None:
            timeline_starts = {}
            setattr(args, "condition_sequence_audio_timeline_starts", timeline_starts)
        return condition_sequence_audio(
            condition_sequence,
            int(plan_index),
            request_tracker_frame=request_tracker_frame,
            target_fps=float(target_fps),
            tracker_fps=float(HOLOMOTION_TRACKER_FPS),
            num_frames=int(num_frames),
            timeline_starts=timeline_starts,
        )
    if request_tracker_frame is not None:
        return _audio_features_for_timeline_replan(
            audio_features,
            plan_index=int(plan_index),
            request_tracker_frame=int(request_tracker_frame),
            target_fps=float(target_fps),
            num_frames=int(num_frames),
            sequence_length=int(sequence_length),
        )
    return audio_features_for_plan(
        audio_features,
        int(plan_index),
        num_frames=int(num_frames),
        sequence_length=int(sequence_length),
        allow_multi_chunk=bool(allow_multi_chunk),
    )


def _human_condition_for_plan(
    args: argparse.Namespace,
    human_motion: PipelineHumanMotion | None,
    plan_index: int,
    *,
    num_frames: int,
    sequence_length: int,
    allow_multi_chunk: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    condition_sequence = _condition_sequence(args)
    if condition_sequence is not None:
        return condition_sequence_human(
            condition_sequence,
            int(plan_index),
            num_frames=int(num_frames),
        )
    return human_motion_for_plan(
        human_motion,
        int(plan_index),
        num_frames=int(num_frames),
        sequence_length=int(sequence_length),
        allow_multi_chunk=bool(allow_multi_chunk),
    )


def _append_reference_history(seed_qpos: np.ndarray, generated_qpos: np.ndarray, history_frames: int) -> np.ndarray:
    updated = np.concatenate(
        [
            np.asarray(seed_qpos, dtype=np.float32),
            np.asarray(generated_qpos, dtype=np.float32),
        ],
        axis=0,
    )
    if updated.shape[0] < history_frames:
        raise ValueError(f"Generated history requires {history_frames} frames, got {updated.shape[0]}")
    return updated[-history_frames:].astype(np.float32, copy=False)


def _plan_prompt_frame_overlay(
    *,
    seed_frames: int,
    prompt_by_generated_frame: list[str],
    chunk_frame_count: int,
) -> list[list[str]]:
    overlays = [
        [
            "caption: motion seed",
            f"seed frame: {frame_idx + 1}/{seed_frames}",
        ]
        for frame_idx in range(seed_frames)
    ]
    for frame_idx, prompt in enumerate(prompt_by_generated_frame):
        chunk_idx = frame_idx // int(chunk_frame_count)
        chunk_cursor = frame_idx % int(chunk_frame_count)
        overlays.append(
            [
                f"text: {prompt}",
                f"chunk: {chunk_idx + 1} frame: {chunk_cursor + 1}/{int(chunk_frame_count)}",
            ]
        )
    return overlays


def _run_diffusion_only(
    args: argparse.Namespace,
    *,
    planner: OnnxDiffusionPlanner,
    seed_qpos: np.ndarray,
    target_fps: float,
    text: str,
    output_dir: Path,
    reference_path: Path,
    audio_features: PipelineAudioCondition | None = None,
    human_motion: PipelineHumanMotion | None = None,
) -> None:
    if _uses_condition_chunks(args, text):
        _validate_condition_chunk_frames(
            args,
            text,
            num_frames=int(args.num_frames),
            sequence_length=planner.sequence_length,
        )
        plan_qpos_chunks: list[np.ndarray] = []
        plan_feature_chunks: list[np.ndarray] = []
        replan_events: list[dict[str, object]] = []
        replan_stats = ReplanMeanStats()
        current_seed_qpos = np.asarray(seed_qpos, dtype=np.float32)
        for plan_index in range(len(_plan_texts(args, text))):
            plan_text = _text_for_plan(args, text, plan_index)
            plan_audio_features = _audio_condition_for_plan(
                args,
                audio_features,
                plan_index,
                request_tracker_frame=None,
                target_fps=target_fps,
                num_frames=planner.sequence_length,
                sequence_length=planner.sequence_length,
            )
            plan_human_motion = _human_condition_for_plan(
                args,
                human_motion,
                plan_index,
                num_frames=planner.sequence_length,
                sequence_length=planner.sequence_length,
            )
            plan = planner.plan(
                seed_qpos_36=current_seed_qpos,
                text=plan_text,
                fps=target_fps,
                num_frames=planner.sequence_length,
                cfg_scale=args.cfg_scale,
                cfg_text_scale=args.cfg_text_scale,
                cfg_audio_scale=args.cfg_audio_scale,
                cfg_human_scale=args.cfg_human_scale,
                audio_features=plan_audio_features,
                human_motion=plan_human_motion,
            )
            replan_events.append(
                replan_stats.log(
                    plan_id=plan_index,
                    prompt=plan_text,
                    source="diffusion",
                    launch_step=plan_index * planner.sequence_length,
                    activate_step=plan_index * planner.sequence_length,
                    timing_ms=plan.metadata.get("timing_ms", {}),
                )
            )
            plan_qpos_chunks.append(plan.qpos_36)
            plan_feature_chunks.append(plan.motion_features)
            current_seed_qpos = _append_reference_history(
                current_seed_qpos,
                plan.qpos_36,
                planner.num_prev_states,
            )
        metadata = {
            "mode": args.mode,
            "seed_motion": str(args.seed_motion),
            "audio_features": describe_pipeline_audio_features(audio_features),
            "human_motion": describe_pipeline_human_motion(human_motion),
            "plan_frames": int(planner.sequence_length),
            "num_replans": len(plan_qpos_chunks),
            "replan_events": replan_events,
        }
        metadata.update(_text_metadata(args, text))
        _save_plan_chunks(
            qpos_chunks=plan_qpos_chunks,
            feature_chunks=plan_feature_chunks,
            fps=target_fps,
            output_dir=output_dir,
            metadata=metadata,
        )
        if args.video:
            video_path = Path(args.video_path) if args.video_path else output_dir / "qpos_36_mujoco.mp4"
            video_seed_qpos = _motion_seed_qpos(seed_qpos, planner.num_prev_states)
            video_qpos = np.concatenate([video_seed_qpos, *plan_qpos_chunks], axis=0)
            prompt_by_generated_frame = []
            for plan_index, chunk in enumerate(plan_qpos_chunks):
                prompt_by_generated_frame.extend([_text_for_plan(args, text, plan_index)] * int(chunk.shape[0]))
            rendered = render_qpos_video(
                video_qpos,
                video_path,
                fps=int(round(target_fps)),
                width=args.video_width,
                height=args.video_height,
                camera_view=args.camera_view,
                follow_mode=args.follow_mode,
                elevation=float(args.camera_elevation),
                scene_preset=args.scene_preset,
                title=args.title,
                overlay_lines=[],
                frame_overlay_lines=_plan_prompt_frame_overlay(
                    seed_frames=video_seed_qpos.shape[0],
                    prompt_by_generated_frame=prompt_by_generated_frame,
                    chunk_frame_count=planner.sequence_length,
                ),
            )
            print(f"video={Path(rendered).resolve()}")
        print(output_dir.resolve())
        return

    plan_text = _text_for_plan(args, text, 0)
    plan_audio_features = _audio_condition_for_plan(
        args,
        audio_features,
        0,
        request_tracker_frame=None,
        target_fps=target_fps,
        num_frames=args.num_frames,
        sequence_length=planner.sequence_length,
        allow_multi_chunk=True,
    )
    plan_human_motion = _human_condition_for_plan(
        args,
        human_motion,
        0,
        num_frames=args.num_frames,
        sequence_length=planner.sequence_length,
        allow_multi_chunk=True,
    )
    plan = planner.plan(
        seed_qpos_36=seed_qpos,
        text=plan_text,
        fps=target_fps,
        num_frames=args.num_frames,
        cfg_scale=args.cfg_scale,
        cfg_text_scale=args.cfg_text_scale,
        cfg_audio_scale=args.cfg_audio_scale,
        cfg_human_scale=args.cfg_human_scale,
        audio_features=plan_audio_features,
        human_motion=plan_human_motion,
    )
    save_motion_plan(
        plan,
        output_dir,
        extra_metadata={
            "mode": args.mode,
            "seed_motion": str(args.seed_motion),
            **_text_metadata(args, text),
            "audio_features": describe_pipeline_audio_features(audio_features),
            "human_motion": describe_pipeline_human_motion(human_motion),
        },
    )
    if args.video:
        video_path = Path(args.video_path) if args.video_path else output_dir / "qpos_36_mujoco.mp4"
        video_seed_qpos = _motion_seed_qpos(seed_qpos, planner.num_prev_states)
        video_qpos = np.concatenate(
            [video_seed_qpos, np.asarray(plan.qpos_36, dtype=np.float32)],
            axis=0,
        )
        rendered = render_qpos_video(
            video_qpos,
            video_path,
            fps=int(round(target_fps)),
            width=args.video_width,
            height=args.video_height,
            camera_view=args.camera_view,
            follow_mode=args.follow_mode,
            elevation=float(args.camera_elevation),
            scene_preset=args.scene_preset,
            title=args.title,
            overlay_lines=[
                f"text: {plan_text}",
            ],
            frame_overlay_lines=_motion_seed_frame_overlay(video_seed_qpos.shape[0], plan.qpos_36.shape[0]),
        )
        print(f"video={Path(rendered).resolve()}")
    print(output_dir.resolve())


def _run_tracker_only(
    args: argparse.Namespace,
    *,
    reference_qpos: np.ndarray,
    reference_fps: float,
    output_dir: Path,
) -> None:
    if args.holomotion_onnx is None:
        raise ValueError("--holomotion-onnx is required when --mode tracker-only")
    total_frames = int(args.num_frames)
    if total_frames <= 0:
        raise ValueError("--num-frames must be positive")

    reference_path = _save_tracker_reference(reference_qpos, reference_fps, output_dir)
    video_path = None
    if args.video:
        video_path = Path(args.video_path) if args.video_path else output_dir / "holomotion_rollout.mp4"
    runner = HoloMotionRolloutRunner(
        holomotion_onnx=args.holomotion_onnx,
        target_fps=HOLOMOTION_TRACKER_FPS,
        robot_xml=args.robot_xml,
        providers=args.tracker_providers,
        control_substeps=int(args.control_substeps),
        action_clip=float(args.action_clip),
        video=bool(args.video),
        video_path=video_path,
        video_width=int(args.video_width),
        video_height=int(args.video_height),
        camera_view=args.camera_view,
        follow_mode=args.follow_mode,
        camera_distance=float(args.camera_distance),
        camera_elevation=float(args.camera_elevation),
        overlay_text="",
    )
    try:
        available_tracker_frames = _resampled_frame_count(
            reference_qpos,
            source_fps=reference_fps,
            target_fps=HOLOMOTION_TRACKER_FPS,
        )
        steps = min(total_frames, available_tracker_frames)
        if steps <= 0:
            raise RuntimeError("Tracker-only reference produced no executable tracker frames")
        chunk = runner.run_reference_chunk(
            reference_qpos,
            reference_fps=reference_fps,
            steps=steps,
            plan_id=0,
            plan_start=0,
            plan_horizon=available_tracker_frames,
        )
        if chunk.frames <= 0:
            raise RuntimeError("HoloMotion tracker executed zero frames")
    finally:
        runner.close()

    metadata = {
        "mode": args.mode,
        "seed_motion": str(args.seed_motion),
        "holomotion_onnx": str(Path(args.holomotion_onnx).expanduser()),
        "reference_fps": float(reference_fps),
        "tracker_fps": HOLOMOTION_TRACKER_FPS,
        "total_tracker_frames": int(chunk.frames),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tracker_output = output_dir / "holomotion_rollout.npz"
    rollout_path = runner.save(
        output_path=tracker_output,
        reference_path=reference_path,
        mode="tracker-only",
        planner_mode=None,
    )
    print(f"tracker_output={rollout_path.resolve()}")
    if video_path is not None:
        print(f"video={Path(video_path).resolve()}")
    print(output_dir.resolve())


def _run_offline_track(
    args: argparse.Namespace,
    *,
    planner: OnnxDiffusionPlanner,
    seed_qpos: np.ndarray,
    target_fps: float,
    text: str,
    output_dir: Path,
    reference_path: Path,
    audio_features: PipelineAudioCondition | None = None,
    human_motion: PipelineHumanMotion | None = None,
) -> None:
    if args.holomotion_onnx is None:
        raise ValueError("--holomotion-onnx is required when --mode offline-track")
    diffusion_frames = int(args.num_frames)
    if diffusion_frames <= 0:
        raise ValueError("--num-frames must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_start = time.perf_counter()
    plan_qpos_chunks: list[np.ndarray] = []
    plan_feature_chunks: list[np.ndarray] = []
    replan_events: list[dict[str, object]] = []
    replan_stats = ReplanMeanStats()
    current_seed_qpos = np.asarray(seed_qpos, dtype=np.float32)
    if _uses_condition_chunks(args, text):
        _validate_condition_chunk_frames(
            args,
            text,
            num_frames=diffusion_frames,
            sequence_length=planner.sequence_length,
        )
        for plan_index in range(len(_plan_texts(args, text))):
            plan_text = _text_for_plan(args, text, plan_index)
            plan_audio_features = _audio_condition_for_plan(
                args,
                audio_features,
                plan_index,
                request_tracker_frame=None,
                target_fps=target_fps,
                num_frames=planner.sequence_length,
                sequence_length=planner.sequence_length,
            )
            plan_human_motion = _human_condition_for_plan(
                args,
                human_motion,
                plan_index,
                num_frames=planner.sequence_length,
                sequence_length=planner.sequence_length,
            )
            plan = planner.plan(
                seed_qpos_36=current_seed_qpos,
                text=plan_text,
                fps=target_fps,
                num_frames=planner.sequence_length,
                cfg_scale=args.cfg_scale,
                cfg_text_scale=args.cfg_text_scale,
                cfg_audio_scale=args.cfg_audio_scale,
                cfg_human_scale=args.cfg_human_scale,
                audio_features=plan_audio_features,
                human_motion=plan_human_motion,
            )
            replan_events.append(
                replan_stats.log(
                    plan_id=plan_index,
                    prompt=plan_text,
                    source="diffusion",
                    launch_step=plan_index * planner.sequence_length,
                    activate_step=plan_index * planner.sequence_length,
                    timing_ms=plan.metadata.get("timing_ms", {}),
                )
            )
            plan_qpos_chunks.append(plan.qpos_36)
            plan_feature_chunks.append(plan.motion_features)
            current_seed_qpos = _append_reference_history(
                current_seed_qpos,
                plan.qpos_36,
                planner.num_prev_states,
            )
    else:
        plan_text = _text_for_plan(args, text, 0)
        plan_audio_features = _audio_condition_for_plan(
            args,
            audio_features,
            0,
            request_tracker_frame=None,
            target_fps=target_fps,
            num_frames=diffusion_frames,
            sequence_length=planner.sequence_length,
            allow_multi_chunk=True,
        )
        plan_human_motion = _human_condition_for_plan(
            args,
            human_motion,
            0,
            num_frames=diffusion_frames,
            sequence_length=planner.sequence_length,
            allow_multi_chunk=True,
        )
        plan = planner.plan(
            seed_qpos_36=seed_qpos,
            text=plan_text,
            fps=target_fps,
            num_frames=diffusion_frames,
            cfg_scale=args.cfg_scale,
            cfg_text_scale=args.cfg_text_scale,
            cfg_audio_scale=args.cfg_audio_scale,
            cfg_human_scale=args.cfg_human_scale,
            audio_features=plan_audio_features,
            human_motion=plan_human_motion,
        )
        replan_events.append(
            replan_stats.log(
                plan_id=0,
                prompt=plan_text,
                source="diffusion",
                launch_step=0,
                activate_step=0,
                timing_ms=plan.metadata.get("timing_ms", {}),
            )
        )
        plan_qpos_chunks.append(plan.qpos_36)
        plan_feature_chunks.append(plan.motion_features)
    planning_seconds = time.perf_counter() - plan_start
    plan_qpos = np.concatenate(plan_qpos_chunks, axis=0).astype(np.float32, copy=False)
    metadata = {}
    metadata.update({
        "mode": args.mode,
        "seed_motion": str(args.seed_motion),
        "holomotion_onnx": str(Path(args.holomotion_onnx).expanduser()),
        "tracker_fps": HOLOMOTION_TRACKER_FPS,
        "diffusion_frames": int(plan_qpos.shape[0]),
        "diffusion_fps": float(target_fps),
        "planning_seconds": float(planning_seconds),
        "audio_features": describe_pipeline_audio_features(audio_features),
        "human_motion": describe_pipeline_human_motion(human_motion),
        "plan_frames": int(planner.sequence_length),
        "num_replans": len(plan_qpos_chunks),
        "replan_events": replan_events,
    })
    metadata.update(_text_metadata(args, text))
    _save_plan_chunks(
        qpos_chunks=plan_qpos_chunks,
        feature_chunks=plan_feature_chunks,
        fps=target_fps,
        output_dir=output_dir,
        metadata=metadata,
    )

    video_path = None
    if args.video:
        video_path = Path(args.video_path) if args.video_path else output_dir / "holomotion_rollout.mp4"
    runner = HoloMotionRolloutRunner(
        holomotion_onnx=args.holomotion_onnx,
        target_fps=HOLOMOTION_TRACKER_FPS,
        robot_xml=args.robot_xml,
        providers=args.tracker_providers,
        control_substeps=int(args.control_substeps),
        action_clip=float(args.action_clip),
        video=bool(args.video),
        video_path=video_path,
        video_width=int(args.video_width),
        video_height=int(args.video_height),
        camera_view=args.camera_view,
        follow_mode=args.follow_mode,
        camera_distance=float(args.camera_distance),
        camera_elevation=float(args.camera_elevation),
        overlay_text=_static_video_overlay_text(args, text),
    )
    tracker_start = time.perf_counter()
    try:
        if args.video:
            runner.append_seed_video(_motion_seed_qpos(seed_qpos, planner.num_prev_states), reference_fps=target_fps)
        available_tracker_frames = _resampled_frame_count(
            plan_qpos,
            source_fps=target_fps,
            target_fps=HOLOMOTION_TRACKER_FPS,
        )
        if available_tracker_frames <= 0:
            raise RuntimeError("Offline-track planning produced no executable tracker frames")
        chunk = runner.run_reference_chunk(
            plan_qpos,
            reference_fps=target_fps,
            steps=available_tracker_frames,
            plan_id=0,
            plan_start=0,
            plan_horizon=available_tracker_frames,
            overlay_text_by_frame=_tracker_frame_prompts(
                args,
                text,
                tracker_frames=available_tracker_frames,
                source_fps=target_fps,
                tracker_fps=HOLOMOTION_TRACKER_FPS,
                sequence_length=planner.sequence_length,
            ),
        )
        if chunk.frames <= 0:
            raise RuntimeError("HoloMotion tracker executed zero frames")
    finally:
        runner.close()
    tracker_seconds = time.perf_counter() - tracker_start

    metadata.update(
        {
            "available_tracker_frames": int(available_tracker_frames),
            "total_tracker_frames": int(chunk.frames),
            "tracker_seconds": float(tracker_seconds),
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tracker_output = output_dir / "holomotion_rollout.npz"
    rollout_path = runner.save(
        output_path=tracker_output,
        reference_path=reference_path,
        mode="offline-track",
        planner_mode=args.mode,
    )
    print(f"tracker_output={rollout_path.resolve()}")
    if video_path is not None:
        print(f"video={Path(video_path).resolve()}")
    print(output_dir.resolve())


def _run_sync(
    args: argparse.Namespace,
    *,
    planner: OnnxDiffusionPlanner,
    seed_qpos: np.ndarray,
    target_fps: float,
    text: str,
    output_dir: Path,
    reference_path: Path,
    audio_features: PipelineAudioCondition | None = None,
    human_motion: PipelineHumanMotion | None = None,
) -> None:
    if args.holomotion_onnx is None:
        raise ValueError("--holomotion-onnx is required when --mode sync")
    total_frames = int(args.num_frames)
    if total_frames <= 0:
        raise ValueError("--num-frames must be positive")
    plan_frames = int(planner.sequence_length)

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = None
    if args.video:
        video_path = Path(args.video_path) if args.video_path else output_dir / "holomotion_rollout.mp4"
    runner = HoloMotionRolloutRunner(
        holomotion_onnx=args.holomotion_onnx,
        target_fps=HOLOMOTION_TRACKER_FPS,
        robot_xml=args.robot_xml,
        providers=args.tracker_providers,
        control_substeps=int(args.control_substeps),
        action_clip=float(args.action_clip),
        video=bool(args.video),
        video_path=video_path,
        video_width=int(args.video_width),
        video_height=int(args.video_height),
        camera_view=args.camera_view,
        follow_mode=args.follow_mode,
        camera_distance=float(args.camera_distance),
        camera_elevation=float(args.camera_elevation),
        overlay_text=_static_video_overlay_text(args, text),
    )

    plan_qpos_chunks: list[np.ndarray] = []
    plan_feature_chunks: list[np.ndarray] = []
    replan_events: list[dict[str, object]] = []
    sync_plan_events: list[dict[str, int]] = []
    replan_stats = ReplanMeanStats()
    executed_frames = 0
    plan_index = 0
    previous_plan = None
    previous_plan_cursor_frames = 0
    try:
        if args.video:
            runner.append_seed_video(_motion_seed_qpos(seed_qpos, planner.num_prev_states), reference_fps=target_fps)
        while executed_frames < total_frames:
            plan_text = _text_for_plan(args, text, plan_index)
            plan_audio_features = _audio_condition_for_plan(
                args,
                audio_features,
                plan_index=plan_index,
                request_tracker_frame=executed_frames,
                target_fps=target_fps,
                num_frames=plan_frames,
                sequence_length=planner.sequence_length,
            )
            plan_human_motion = _human_condition_for_plan(
                args,
                human_motion,
                plan_index,
                num_frames=plan_frames,
                sequence_length=planner.sequence_length,
            )
            plan = planner.plan(
                seed_qpos_36=seed_qpos,
                text=plan_text,
                fps=target_fps,
                num_frames=plan_frames,
                cfg_scale=args.cfg_scale,
                cfg_text_scale=args.cfg_text_scale,
                cfg_audio_scale=args.cfg_audio_scale,
                cfg_human_scale=args.cfg_human_scale,
                previous_plan=previous_plan,
                previous_plan_cursor_frames=previous_plan_cursor_frames,
                continuation_steps=args.diffusion_continuation_steps,
                audio_features=plan_audio_features,
                human_motion=plan_human_motion,
            )
            replan_event = replan_stats.log(
                plan_id=plan_index,
                prompt=plan_text,
                source="diffusion",
                launch_step=executed_frames,
                activate_step=executed_frames,
                timing_ms=plan.metadata.get("timing_ms", {}),
            )
            replan_events.append(replan_event)
            plan_qpos_chunks.append(plan.qpos_36)
            plan_feature_chunks.append(plan.motion_features)
            available_tracker_frames = _resampled_frame_count(plan.qpos_36, source_fps=plan.fps, target_fps=HOLOMOTION_TRACKER_FPS)
            steps = min(available_tracker_frames, total_frames - executed_frames)
            if steps <= 0:
                raise RuntimeError("Sync planning produced no executable tracker frames")
            source_cursor_frames = _source_cursor_from_tracker_cursor(
                steps,
                source_fps=plan.fps,
                tracker_fps=HOLOMOTION_TRACKER_FPS,
            )
            sync_plan_events.append(
                {
                    "plan_id": int(plan_index),
                    "diffusion_frames": int(plan.qpos_36.shape[0]),
                    "diffusion_fps": int(round(float(plan.fps))),
                    "available_tracker_frames": int(available_tracker_frames),
                    "executed_tracker_frames": int(steps),
                    "source_cursor_frames": int(source_cursor_frames),
                    "launch_step": int(executed_frames),
                    "activate_step": int(executed_frames),
                    "text": plan_text,
                }
            )
            chunk = runner.run_reference_chunk(
                plan.qpos_36,
                reference_fps=plan.fps,
                steps=steps,
                plan_id=plan_index,
                plan_start=0,
                plan_horizon=available_tracker_frames,
                overlay_text=plan_text,
            )
            if chunk.frames <= 0:
                raise RuntimeError("HoloMotion tracker executed zero frames")
            executed_frames += chunk.frames
            seed_qpos = _append_executed_history(
                seed_qpos,
                chunk.qpos_36,
                executed_fps=HOLOMOTION_TRACKER_FPS,
                target_fps=target_fps,
                history_frames=planner.num_prev_states,
            )
            previous_plan = plan
            previous_plan_cursor_frames = source_cursor_frames
            plan_index += 1
    finally:
        runner.close()

    _save_plan_chunks(
        qpos_chunks=plan_qpos_chunks,
        feature_chunks=plan_feature_chunks,
        fps=target_fps,
        output_dir=output_dir,
        metadata={
            "mode": args.mode,
            "seed_motion": str(args.seed_motion),
            "holomotion_onnx": str(Path(args.holomotion_onnx).expanduser()),
            "tracker_fps": HOLOMOTION_TRACKER_FPS,
            "total_tracker_frames": executed_frames,
            "plan_frames": plan_frames,
            "execute_frames_per_plan": "resampled_horizon",
            "diffusion_continuation_steps": int(args.diffusion_continuation_steps),
            "audio_features": describe_pipeline_audio_features(audio_features),
            "human_motion": describe_pipeline_human_motion(human_motion),
            "num_replans": plan_index,
            "replan_events": replan_events,
            "sync_plan_events": sync_plan_events,
            **_text_metadata(args, text),
        },
    )
    tracker_output = output_dir / "holomotion_rollout.npz"
    rollout_path = runner.save(
        output_path=tracker_output,
        reference_path=reference_path,
        mode="sync",
        planner_mode=args.mode,
    )
    print(f"tracker_output={rollout_path.resolve()}")
    if video_path is not None:
        print(f"video={Path(video_path).resolve()}")
    print(output_dir.resolve())


def _run_buffer_frames(
    runner: HoloMotionRolloutRunner,
    reference_buffer: np.ndarray,
    plan_segments: list[dict[str, object]],
    *,
    cursor: int,
    frames: int,
) -> np.ndarray:
    cursor = int(cursor)
    remaining = int(frames)
    if remaining < 0:
        raise ValueError(f"frames must be non-negative, got {remaining}")
    executed_chunks: list[np.ndarray] = []
    while remaining > 0:
        segment = next(
            (
                item
                for item in plan_segments
                if int(item["start"]) <= cursor < int(item["end"])
            ),
            None,
        )
        if segment is None:
            raise RuntimeError(f"No motion-buffer segment covers tracker cursor {cursor}")
        segment_end = int(segment["end"])
        step_frames = min(remaining, segment_end - cursor)
        chunk = runner.run_reference_chunk(
            reference_buffer[cursor:],
            reference_fps=HOLOMOTION_TRACKER_FPS,
            steps=step_frames,
            plan_id=int(segment["plan_id"]),
            plan_start=cursor - int(segment["start"]),
            plan_horizon=segment_end - int(segment["start"]),
            overlay_text=str(segment.get("text", "")),
        )
        if chunk.frames != step_frames:
            raise RuntimeError(f"HoloMotion tracker executed {chunk.frames} frames, expected {step_frames}")
        executed_chunks.append(chunk.qpos_36)
        cursor += chunk.frames
        remaining -= chunk.frames
    if not executed_chunks:
        return np.zeros((0, 36), dtype=np.float32)
    return np.concatenate(executed_chunks, axis=0).astype(np.float32, copy=False)


def _audio_features_for_timeline_replan(
    audio_features: PipelineAudioCondition | None,
    *,
    plan_index: int,
    request_tracker_frame: int,
    target_fps: float,
    num_frames: int,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if audio_features is None:
        return None
    if bool(getattr(audio_features, "realtime_timeline", False)):
        audio_start_frame = int(
            np.floor(
                float(request_tracker_frame) * float(target_fps) / float(HOLOMOTION_TRACKER_FPS)
                + 1e-9
            )
        )
        return audio_features_for_timeline_frame(
            audio_features,
            audio_start_frame,
            num_frames=int(num_frames),
        )
    return audio_features_for_plan(
        audio_features,
        int(plan_index),
        num_frames=int(num_frames),
        sequence_length=int(sequence_length),
    )


def _run_async(
    args: argparse.Namespace,
    *,
    planner: OnnxDiffusionPlanner,
    seed_qpos: np.ndarray,
    target_fps: float,
    text: str,
    output_dir: Path,
    reference_path: Path,
    audio_features: PipelineAudioCondition | None = None,
    human_motion: PipelineHumanMotion | None = None,
) -> None:
    if args.holomotion_onnx is None:
        raise ValueError("--holomotion-onnx is required when --mode async")
    total_frames = int(args.num_frames)
    if total_frames <= 0:
        raise ValueError("--num-frames must be positive")
    plan_frames = int(planner.sequence_length)

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = None
    if args.video:
        video_path = Path(args.video_path) if args.video_path else output_dir / "holomotion_rollout.mp4"
    runner = HoloMotionRolloutRunner(
        holomotion_onnx=args.holomotion_onnx,
        target_fps=HOLOMOTION_TRACKER_FPS,
        robot_xml=args.robot_xml,
        providers=args.tracker_providers,
        control_substeps=int(args.control_substeps),
        action_clip=float(args.action_clip),
        video=bool(args.video),
        video_path=video_path,
        video_width=int(args.video_width),
        video_height=int(args.video_height),
        camera_view=args.camera_view,
        follow_mode=args.follow_mode,
        camera_distance=float(args.camera_distance),
        camera_elevation=float(args.camera_elevation),
        overlay_text=_static_video_overlay_text(args, text),
    )

    plan_qpos_chunks: list[np.ndarray] = []
    plan_feature_chunks: list[np.ndarray] = []
    replan_events: list[dict[str, object]] = []
    plan_metadata_events: list[dict[str, object]] = []
    replan_stats = ReplanMeanStats()
    async_events: list[dict[str, object]] = []
    plan_segments: list[dict[str, object]] = []
    reference_buffer = np.zeros((0, 36), dtype=np.float32)
    executed_frames = 0
    initial_planning_latency = 0.0
    replan_remaining_frames = int(args.async_replan_remaining_frames)
    if replan_remaining_frames < 0:
        raise ValueError(f"--async-replan-remaining-frames must be non-negative, got {replan_remaining_frames}")

    def append_plan(plan_id: int, plan, *, prompt: str, reference_start_frame: int = 0) -> tuple[int, int]:
        nonlocal reference_buffer
        reference = resample_qpos(
            plan.qpos_36,
            source_fps=plan.fps,
            target_fps=HOLOMOTION_TRACKER_FPS,
        )
        reference_start_frame = int(reference_start_frame)
        if reference_start_frame < 0:
            raise ValueError(f"reference_start_frame must be non-negative, got {reference_start_frame}")
        if reference_start_frame >= int(reference.shape[0]):
            raise RuntimeError(
                "Async replan latency consumed the entire planned tracker horizon: "
                f"skip={reference_start_frame}, horizon={reference.shape[0]}"
            )
        reference = reference[reference_start_frame:]
        start = int(reference_buffer.shape[0])
        reference_buffer = np.concatenate([reference_buffer, reference.astype(np.float32, copy=False)], axis=0)
        end = int(reference_buffer.shape[0])
        plan_segments.append({"plan_id": int(plan_id), "start": start, "end": end, "text": str(prompt)})
        plan_qpos_chunks.append(plan.qpos_36)
        plan_feature_chunks.append(plan.motion_features)
        return start, end

    def reset_future_buffer(cursor: int) -> None:
        nonlocal reference_buffer, plan_segments
        cursor = int(cursor)
        reference_buffer = reference_buffer[:cursor].astype(np.float32, copy=False)
        clipped_segments: list[dict[str, int]] = []
        for segment in plan_segments:
            start = int(segment["start"])
            end = int(segment["end"])
            if start >= cursor:
                continue
            clipped_segments.append(
                {
                    "plan_id": int(segment["plan_id"]),
                    "start": start,
                    "end": min(end, cursor),
                    "text": str(segment.get("text", "")),
                }
            )
        plan_segments = clipped_segments

    def dit_cache_event(plan) -> dict[str, object]:
        return {
            "enabled": bool(plan.metadata.get("dit_cache", False)),
            "threshold": plan.metadata.get("dit_cache_threshold"),
            "warmup_steps": plan.metadata.get("dit_cache_warmup_steps"),
            "max_consecutive": plan.metadata.get("dit_cache_max_consecutive"),
            "executed_steps": plan.metadata.get("dit_cache_executed_steps"),
            "skipped_steps": plan.metadata.get("dit_cache_skipped_steps"),
            "chunks": plan.metadata.get("dit_cache_chunks", []),
        }

    def log_replan_event(
        *,
        plan_id: int,
        plan,
        prompt: str,
        launch_step: int,
        activate_step: int,
    ) -> None:
        event = replan_stats.log(
            plan_id=plan_id,
            prompt=prompt,
            source="diffusion",
            launch_step=launch_step,
            activate_step=activate_step,
            timing_ms=plan.metadata.get("timing_ms", {}),
        )
        event["dit_cache"] = dit_cache_event(plan)
        replan_events.append(event)
        plan_metadata_events.append(
            {
                "plan_id": int(plan_id),
                "prompt": str(prompt),
                "launch_step": int(launch_step),
                "activate_step": int(activate_step),
                "metadata": plan.metadata,
            }
        )

    def execute_buffer_frames(frames: int) -> None:
        nonlocal executed_frames, seed_qpos
        frames = int(frames)
        if frames <= 0:
            return
        cursor = int(executed_frames)
        executed = _run_buffer_frames(
            runner,
            reference_buffer,
            plan_segments,
            cursor=cursor,
            frames=frames,
        )
        seed_qpos = _append_executed_history(
            seed_qpos,
            executed,
            executed_fps=HOLOMOTION_TRACKER_FPS,
            target_fps=target_fps,
            history_frames=planner.num_prev_states,
        )
        executed_frames += int(executed.shape[0])

    try:
        if args.video:
            runner.append_seed_video(_motion_seed_qpos(seed_qpos, planner.num_prev_states), reference_fps=target_fps)
        initial_audio_features = _audio_condition_for_plan(
            args,
            audio_features,
            plan_index=0,
            request_tracker_frame=0,
            target_fps=target_fps,
            num_frames=plan_frames,
            sequence_length=planner.sequence_length,
        )
        initial_human_motion = _human_condition_for_plan(
            args,
            human_motion,
            0,
            num_frames=plan_frames,
            sequence_length=planner.sequence_length,
        )
        initial_text = _text_for_plan(args, text, 0)
        started = time.perf_counter()
        initial_plan = planner.plan(
            seed_qpos_36=seed_qpos,
            text=initial_text,
            fps=target_fps,
            num_frames=plan_frames,
            cfg_scale=args.cfg_scale,
            cfg_text_scale=args.cfg_text_scale,
            cfg_audio_scale=args.cfg_audio_scale,
            cfg_human_scale=args.cfg_human_scale,
            audio_features=initial_audio_features,
            human_motion=initial_human_motion,
        )
        initial_planning_latency = time.perf_counter() - started
        append_plan(0, initial_plan, prompt=initial_text)
        log_replan_event(
            plan_id=0,
            plan=initial_plan,
            prompt=initial_text,
            launch_step=0,
            activate_step=0,
        )

        while executed_frames < total_frames:
            remaining_total = total_frames - executed_frames
            buffered_remaining = int(reference_buffer.shape[0]) - executed_frames
            if buffered_remaining <= 0:
                raise RuntimeError("Async motion buffer is empty before the requested rollout length")

            if remaining_total <= buffered_remaining:
                execute_buffer_frames(remaining_total)
                if executed_frames >= total_frames:
                    break
                continue

            if buffered_remaining > replan_remaining_frames:
                execute_buffer_frames(buffered_remaining - replan_remaining_frames)
                continue

            request_frame = executed_frames
            next_plan_id = len(plan_qpos_chunks)
            next_text = _text_for_plan(args, text, next_plan_id)
            next_audio_features = _audio_condition_for_plan(
                args,
                audio_features,
                plan_index=next_plan_id,
                request_tracker_frame=request_frame,
                target_fps=target_fps,
                num_frames=plan_frames,
                sequence_length=planner.sequence_length,
            )
            next_human_motion = _human_condition_for_plan(
                args,
                human_motion,
                next_plan_id,
                num_frames=plan_frames,
                sequence_length=planner.sequence_length,
            )
            started = time.perf_counter()
            next_plan = planner.plan(
                seed_qpos_36=seed_qpos,
                text=next_text,
                fps=target_fps,
                num_frames=plan_frames,
                cfg_scale=args.cfg_scale,
                cfg_text_scale=args.cfg_text_scale,
                cfg_audio_scale=args.cfg_audio_scale,
                cfg_human_scale=args.cfg_human_scale,
                audio_features=next_audio_features,
                human_motion=next_human_motion,
            )
            planning_latency = time.perf_counter() - started
            measured_elapsed_tracker_frames = _tracker_frames_from_latency(planning_latency)
            elapsed_tracker_frames = _async_elapsed_tracker_frames(args, planning_latency)

            bridge_frames = min(elapsed_tracker_frames, total_frames - executed_frames)
            buffered_remaining = int(reference_buffer.shape[0]) - executed_frames
            if bridge_frames > buffered_remaining:
                raise RuntimeError(
                    "Async motion buffer underrun: "
                    f"planner latency consumed {elapsed_tracker_frames} tracker frames, "
                    f"but the buffer has {buffered_remaining} unexecuted frames"
                )
            execute_buffer_frames(bridge_frames)

            reset_future_buffer(executed_frames)
            buffer_start, buffer_end = append_plan(
                next_plan_id,
                next_plan,
                prompt=next_text,
                reference_start_frame=bridge_frames,
            )
            log_replan_event(
                plan_id=next_plan_id,
                plan=next_plan,
                prompt=next_text,
                launch_step=request_frame,
                activate_step=buffer_start,
            )
            async_events.append(
                {
                    "plan_id": next_plan_id,
                    "prompt": next_text,
                    "source": "stream",
                    "seed_source": "executed_history",
                    "request_tracker_frame": request_frame,
                    "ready_tracker_frame": request_frame + elapsed_tracker_frames,
                    "buffer_start_frame": buffer_start,
                    "buffer_end_frame": buffer_end,
                    "buffered_frames_at_request": buffered_remaining,
                    "elapsed_tracker_frames": elapsed_tracker_frames,
                    "measured_elapsed_tracker_frames": measured_elapsed_tracker_frames,
                    "planning_latency_seconds": planning_latency,
                    "skipped_plan_tracker_frames": bridge_frames,
                    "activated": buffer_start < total_frames,
                }
            )
    finally:
        runner.close()

    _save_plan_chunks(
        qpos_chunks=plan_qpos_chunks,
        feature_chunks=plan_feature_chunks,
        fps=target_fps,
        output_dir=output_dir,
        metadata={
            "mode": args.mode,
            "seed_motion": str(args.seed_motion),
            "holomotion_onnx": str(Path(args.holomotion_onnx).expanduser()),
            "tracker_fps": HOLOMOTION_TRACKER_FPS,
            "total_tracker_frames": executed_frames,
            "plan_frames": plan_frames,
            "tracker_plan_horizon_frames": plan_segments[0]["end"] - plan_segments[0]["start"],
            "async_mode": "executed_history_buffer",
            "motion_buffer_frames": int(reference_buffer.shape[0]),
            "buffer_seed_history_frames": int(planner.num_prev_states),
            "replan_remaining_frames": int(replan_remaining_frames),
            "async_latency_frames": args.async_latency_frames,
            "diffusion_continuation_steps": 0,
            "audio_features": describe_pipeline_audio_features(audio_features),
            "human_motion": describe_pipeline_human_motion(human_motion),
            "num_replans": len(plan_qpos_chunks) - 1,
            "initial_planning_latency_seconds": initial_planning_latency,
            "replan_events": replan_events,
            "diffusion_plan_metadata": plan_metadata_events,
            "async_events": async_events,
            **_text_metadata(args, text),
        },
    )
    tracker_output = output_dir / "holomotion_rollout.npz"
    rollout_path = runner.save(
        output_path=tracker_output,
        reference_path=reference_path,
        mode="async",
        planner_mode=args.mode,
    )
    print(f"tracker_output={rollout_path.resolve()}")
    if video_path is not None:
        print(f"video={Path(video_path).resolve()}")
    print(output_dir.resolve())
