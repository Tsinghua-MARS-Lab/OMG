from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from omg.pipeline import MotionPlan, save_motion_plan
from omg.tracking.holomotion.reference import resample_qpos
from omg.tracking.holomotion.runner import HoloMotionRolloutRunner

HOLOMOTION_TRACKER_FPS = 50.0


def _output_name(args: argparse.Namespace) -> str:
    if args.output_name is not None:
        return str(args.output_name)
    if args.mode == "tracker-only":
        return "tracker_only_holomotion"
    if args.mode == "sync":
        return "sync_onnx_holomotion"
    if args.mode == "async":
        return "async_onnx_holomotion"
    if args.mode == "offline-track":
        return "offline_track_onnx_holomotion"
    return "diffusion_only_onnx"


def _resampled_frame_count(qpos_36: np.ndarray, source_fps: float, target_fps: float) -> int:
    return int(resample_qpos(qpos_36, source_fps=source_fps, target_fps=target_fps).shape[0])


def _append_executed_history(
    seed_qpos: np.ndarray,
    executed_qpos: np.ndarray,
    *,
    executed_fps: float,
    target_fps: float,
    history_frames: int,
) -> np.ndarray:
    executed_at_target_fps = resample_qpos(executed_qpos, source_fps=executed_fps, target_fps=target_fps)
    updated = np.concatenate([seed_qpos, executed_at_target_fps], axis=0)
    if updated.shape[0] < history_frames:
        raise ValueError(f"Replan history requires {history_frames} frames, got {updated.shape[0]}")
    return updated[-history_frames:].astype(np.float32, copy=False)


def _tracker_frames_from_latency(seconds: float) -> int:
    if not np.isfinite(float(seconds)) or float(seconds) < 0.0:
        raise ValueError(f"Planning latency must be finite and non-negative, got {seconds}")
    return int(np.floor(float(seconds) * HOLOMOTION_TRACKER_FPS + 1e-9))


def _source_cursor_from_tracker_cursor(tracker_cursor: int, source_fps: float, tracker_fps: float) -> int:
    cursor = int(tracker_cursor)
    if cursor < 0:
        raise ValueError(f"tracker_cursor must be non-negative, got {cursor}")
    if not np.isfinite(float(source_fps)) or float(source_fps) <= 0.0:
        raise ValueError(f"source_fps must be positive and finite, got {source_fps}")
    if not np.isfinite(float(tracker_fps)) or float(tracker_fps) <= 0.0:
        raise ValueError(f"tracker_fps must be positive and finite, got {tracker_fps}")
    return int(np.ceil((float(cursor) * float(source_fps) / float(tracker_fps)) - 1e-9))


def _async_elapsed_tracker_frames(args: argparse.Namespace, planning_latency_seconds: float) -> int:
    if args.async_latency_frames is None:
        return _tracker_frames_from_latency(planning_latency_seconds)
    latency_frames = int(args.async_latency_frames)
    if latency_frames < 0:
        raise ValueError(f"--async-latency-frames must be non-negative, got {latency_frames}")
    return latency_frames


def _tracking_error_stats(executed_qpos: np.ndarray, reference_qpos: np.ndarray) -> dict[str, float]:
    executed = np.asarray(executed_qpos, dtype=np.float32)
    reference = np.asarray(reference_qpos, dtype=np.float32)
    if executed.shape != reference.shape:
        raise ValueError(f"Tracking error expects matching shapes, got {executed.shape} and {reference.shape}")
    if executed.ndim != 2 or executed.shape[1] != 36:
        raise ValueError(f"Expected qpos shape (T,36), got {executed.shape}")
    if executed.shape[0] <= 0:
        return {
            "root_xy_error_mean": 0.0,
            "root_xy_error_max": 0.0,
            "joint_abs_error_mean": 0.0,
            "joint_abs_error_max": 0.0,
        }
    root_xy = np.linalg.norm(executed[:, :2] - reference[:, :2], axis=1)
    joint_abs = np.abs(executed[:, 7:36] - reference[:, 7:36])
    return {
        "root_xy_error_mean": float(root_xy.mean()),
        "root_xy_error_max": float(root_xy.max()),
        "joint_abs_error_mean": float(joint_abs.mean()),
        "joint_abs_error_max": float(joint_abs.max()),
    }


def _run_tracker_frames(
    runner: HoloMotionRolloutRunner,
    reference_qpos_tracker_fps: np.ndarray,
    *,
    cursor: int,
    frames: int,
    plan_id: int,
):
    chunk = runner.run_reference_chunk(
        reference_qpos_tracker_fps[int(cursor) :],
        reference_fps=HOLOMOTION_TRACKER_FPS,
        steps=int(frames),
        plan_id=int(plan_id),
        plan_start=int(cursor),
        plan_horizon=int(reference_qpos_tracker_fps.shape[0]),
    )
    if chunk.frames != int(frames):
        raise RuntimeError(f"HoloMotion tracker executed {chunk.frames} frames, expected {frames}")
    return chunk


def _save_plan_chunks(
    *,
    qpos_chunks: list[np.ndarray],
    feature_chunks: list[np.ndarray],
    fps: float,
    output_dir: Path,
    metadata: dict,
) -> Path:
    qpos = np.concatenate(qpos_chunks, axis=0).astype(np.float32, copy=False)
    features = np.concatenate(feature_chunks, axis=0).astype(np.float32, copy=False)
    return save_motion_plan(
        MotionPlan(qpos_36=qpos, motion_features=features, fps=float(fps), metadata=metadata),
        output_dir,
    )


class ReplanMeanStats:
    def __init__(self) -> None:
        self.count = 0
        self.text_encode_ms_sum = 0.0
        self.downsample_ms_sum = 0.0
        self.canonicalization_ms_sum = 0.0
        self.history_qpos_to_device_ms_sum = 0.0
        self.history_fk_ms_sum = 0.0
        self.history_features_ms_sum = 0.0
        self.continuation_ms_sum = 0.0
        self.diffusion_infer_ms_sum = 0.0
        self.ik_ms_sum = 0.0
        self.interpolation_ms_sum = 0.0

    def log(
        self,
        *,
        plan_id: int,
        prompt: str,
        source: str,
        launch_step: int,
        activate_step: int,
        timing_ms: dict,
    ) -> dict[str, object]:
        t5_ms = float(timing_ms.get("text_encode_ms", 0.0))
        ds_ms = float(timing_ms.get("downsample_ms", 0.0))
        can_ms = float(timing_ms.get("canonicalization_ms", 0.0))
        qpos_device_ms = float(timing_ms.get("history_qpos_to_device_ms", 0.0))
        fk_ms = float(timing_ms.get("history_fk_ms", 0.0))
        hist_feat_ms = float(timing_ms.get("history_features_ms", 0.0))
        cont_ms = float(timing_ms.get("continuation_ms", 0.0))
        diff_ms = float(timing_ms.get("diffusion_infer_ms", 0.0))
        ik_ms = float(timing_ms.get("ik_ms", 0.0))
        interp_ms = float(timing_ms.get("interpolation_ms", 0.0))
        total_ms = float(timing_ms.get("total_ms", 0.0))
        self.count += 1
        self.text_encode_ms_sum += t5_ms
        self.downsample_ms_sum += ds_ms
        self.canonicalization_ms_sum += can_ms
        self.history_qpos_to_device_ms_sum += qpos_device_ms
        self.history_fk_ms_sum += fk_ms
        self.history_features_ms_sum += hist_feat_ms
        self.continuation_ms_sum += cont_ms
        self.diffusion_infer_ms_sum += diff_ms
        self.ik_ms_sum += ik_ms
        self.interpolation_ms_sum += interp_ms
        count = float(self.count)
        print(
            f"[replan {int(plan_id):04d}] prompt={prompt!r} src={source} "
            f"launch={int(launch_step):05d} activate={int(activate_step):05d} "
            f"[mean] t5={self.text_encode_ms_sum / count:.3f}ms "
            f"ds={self.downsample_ms_sum / count:.3f}ms "
            f"can={self.canonicalization_ms_sum / count:.3f}ms "
            f"q2d={self.history_qpos_to_device_ms_sum / count:.3f}ms "
            f"fk={self.history_fk_ms_sum / count:.3f}ms "
            f"hist={self.history_features_ms_sum / count:.3f}ms "
            f"cont={self.continuation_ms_sum / count:.3f}ms "
            f"diff={self.diffusion_infer_ms_sum / count:.3f}ms "
            f"ik={self.ik_ms_sum / count:.3f}ms "
            f"interp={self.interpolation_ms_sum / count:.3f}ms",
            flush=True,
        )
        return {
            "plan_id": int(plan_id),
            "prompt": str(prompt),
            "source": str(source),
            "launch_step": int(launch_step),
            "activate_step": int(activate_step),
            "timing_ms": {
                "text_encode_ms": t5_ms,
                "downsample_ms": ds_ms,
                "canonicalization_ms": can_ms,
                "history_qpos_to_device_ms": qpos_device_ms,
                "history_fk_ms": fk_ms,
                "history_features_ms": hist_feat_ms,
                "continuation_ms": cont_ms,
                "diffusion_infer_ms": diff_ms,
                "ik_ms": ik_ms,
                "interpolation_ms": interp_ms,
                "total_ms": total_ms,
            },
        }


def _save_tracker_reference(reference_qpos: np.ndarray, reference_fps: float, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "qpos_36.npy", reference_qpos.astype(np.float32, copy=False))
    path = output_dir / "reference_motion.npz"
    np.savez_compressed(
        path,
        qpos_36=reference_qpos.astype(np.float32, copy=False),
        fps=np.asarray([float(reference_fps)], dtype=np.float32),
    )
    return path
