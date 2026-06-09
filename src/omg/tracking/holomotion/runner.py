from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from omg.runtime.onnx_providers import DEFAULT_TRACKER_ONNX_PROVIDERS_CSV


def configure_mujoco_env() -> None:
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    if os.environ.get("MUJOCO_GL") == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("EGL_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0") or "0")


configure_mujoco_env()

import mujoco  # noqa: E402

from omg.tracking.holomotion.augmentation import PushEvent, push_events_to_dicts  # noqa: E402
from omg.tracking.holomotion.io import load_reference_motion  # noqa: E402
from omg.tracking.holomotion.reference import (  # noqa: E402
    build_holomotion_obs,
    precompute_reference_features,
    resample_qpos,
)
from omg.tracking.holomotion.rollout import default_rollout_path, save_tracker_rollout  # noqa: E402
from omg.tracking.holomotion.runtime import (  # noqa: E402
    HoloMotionTrackerSession,
    apply_holomotion_action_pd,
    build_g1_state_handles,
    build_holomotion_handles,
    build_onnx_session,
    extract_g1_qpos,
    load_holomotion_metadata,
    resolve_robot_xml,
    set_g1_qpos,
)
from omg.tracking.holomotion.video import append_video_frame, close_video, open_video_writer  # noqa: E402


@dataclass(frozen=True)
class TrackerRunConfig:
    reference: str | Path
    holomotion_onnx: str | Path
    reference_fps: float | None = None
    target_fps: float = 50.0
    robot_xml: str | Path | None = None
    providers: str | list[str] = DEFAULT_TRACKER_ONNX_PROVIDERS_CSV
    steps: int | None = None
    control_substeps: int = 10
    action_clip: float = 10.0
    output: str | Path | None = None
    video: bool = False
    video_path: str | Path | None = None
    video_width: int = 640
    video_height: int = 480
    camera_view: str = "side"
    follow_mode: str = "xy"
    camera_distance: float = 4.0
    camera_elevation: float = -20.0
    overlay_text: str = ""
    mode: str = "tracker-only"
    planner_mode: str | None = None
    init_qpos_36: np.ndarray | None = None
    push_events: list[PushEvent] | None = None
    augmentation_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class TrackerRunResult:
    rollout_path: Path
    video_path: Path | None
    frames: int
    fps: float


@dataclass(frozen=True)
class TrackerChunkResult:
    frames: int
    qpos_36: np.ndarray


class HoloMotionRolloutRunner:
    def __init__(
        self,
        *,
        holomotion_onnx: str | Path,
        target_fps: float = 50.0,
        robot_xml: str | Path | None = None,
        providers: str | list[str] = DEFAULT_TRACKER_ONNX_PROVIDERS_CSV,
        control_substeps: int = 10,
        action_clip: float = 10.0,
        video: bool = False,
        video_path: str | Path | None = None,
        video_width: int = 640,
        video_height: int = 480,
        camera_view: str = "side",
        follow_mode: str = "xy",
        camera_distance: float = 4.0,
        camera_elevation: float = -20.0,
        overlay_text: str = "",
    ):
        self.holomotion_onnx = Path(holomotion_onnx).expanduser().resolve()
        self.target_fps = float(target_fps)
        self.control_substeps = int(control_substeps)
        self.action_clip = float(action_clip)
        self.camera_view = str(camera_view)
        self.follow_mode = str(follow_mode)
        self.camera_distance = float(camera_distance)
        self.camera_elevation = float(camera_elevation)
        self.overlay_text = str(overlay_text)
        self.robot_xml = resolve_robot_xml(robot_xml)
        self.model = mujoco.MjModel.from_xml_path(str(self.robot_xml))
        self.data = mujoco.MjData(self.model)
        self.g1_handles = build_g1_state_handles(self.model)
        self.session = build_onnx_session(self.holomotion_onnx, providers)
        self.metadata = load_holomotion_metadata(self.session)
        self.holomotion_handles = build_holomotion_handles(self.model, self.metadata)
        self.tracker = HoloMotionTrackerSession(self.session, self.metadata)
        self.last_action = np.zeros(len(self.metadata.joint_names), dtype=np.float32)
        self.initialized = False
        self.executed_qpos_36: list[np.ndarray] = []
        self.reference_qpos_36: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self._obs_history: dict[str, list[np.ndarray]] = {}
        self.plan_cursor: list[int] = []
        self.plan_id: list[int] = []
        self._body_id_cache: dict[str, int] = {}
        self.renderer, self.camera, self.writer = open_video_writer(
            video,
            self.model,
            video_path,
            width=int(video_width),
            height=int(video_height),
            fps=self.target_fps,
            camera_view=self.camera_view,
            camera_distance=self.camera_distance,
            camera_elevation=self.camera_elevation,
        )
        self.video_path = Path(video_path).expanduser().resolve() if video and video_path is not None else None

    def reset(self, qpos_36: np.ndarray) -> None:
        set_g1_qpos(self.model, self.data, self.g1_handles, qpos_36)
        self.tracker.reset()
        self.last_action = np.zeros(len(self.metadata.joint_names), dtype=np.float32)
        self._obs_history.clear()
        self.initialized = True

    def clear_rollout_buffers(self, *, reset_initialized: bool = True) -> None:
        self.executed_qpos_36.clear()
        self.reference_qpos_36.clear()
        self.actions.clear()
        self._obs_history.clear()
        self.plan_cursor.clear()
        self.plan_id.clear()
        if reset_initialized:
            self.initialized = False

    def close(self) -> None:
        close_video(self.renderer, self.writer)
        self.renderer = None
        self.writer = None

    def append_seed_video(self, qpos_36: np.ndarray, *, reference_fps: float) -> int:
        if self.writer is None:
            return 0
        qpos_ref = resample_qpos(
            qpos_36,
            source_fps=float(reference_fps),
            target_fps=self.target_fps,
        )
        if qpos_ref.shape[0] <= 0:
            return 0
        for frame_idx, qpos_frame in enumerate(qpos_ref):
            set_g1_qpos(self.model, self.data, self.g1_handles, qpos_frame)
            mujoco.mj_forward(self.model, self.data)
            overlay_lines = [
                f"text: {self.overlay_text}" if self.overlay_text else "",
                "caption: motion seed",
                f"seed frame: {frame_idx + 1}/{qpos_ref.shape[0]}",
            ]
            append_video_frame(
                self.renderer,
                self.camera,
                self.writer,
                self.data,
                track_pos=np.asarray(self.data.xpos[self.g1_handles["pelvis_body_id"]]),
                camera_view=self.camera_view,
                follow_mode=self.follow_mode,
                camera_elevation=self.camera_elevation,
                overlay_lines=overlay_lines,
            )
        return int(qpos_ref.shape[0])

    def _body_id(self, body_name: str) -> int:
        cached = self._body_id_cache.get(body_name)
        if cached is not None:
            return cached
        body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name))
        if body_id < 0:
            raise KeyError(f"Mujoco body not found for push event: {body_name}")
        self._body_id_cache[body_name] = body_id
        return body_id

    def _clear_external_forces(self) -> None:
        self.data.xfrc_applied[:, :] = 0.0

    def _apply_push_events(self, frame_idx: int, push_events: list[PushEvent] | None) -> None:
        if not push_events:
            return
        for event in push_events:
            if event.active_at(frame_idx):
                self.data.xfrc_applied[self._body_id(event.body_name), :3] += np.asarray(event.force_xyz, dtype=np.float64)

    def run_reference_chunk(
        self,
        qpos_36: np.ndarray,
        *,
        reference_fps: float,
        steps: int | None = None,
        plan_id: int = 0,
        plan_start: int = 0,
        plan_horizon: int | None = None,
        init_qpos_36: np.ndarray | None = None,
        push_events: list[PushEvent] | None = None,
        qpos_ref_resampled: np.ndarray | None = None,
        record_buffers: bool = True,
        frame_callback: Callable[[int, np.ndarray, np.ndarray, np.ndarray], None] | None = None,
        overlay_text: str | None = None,
        overlay_text_by_frame: list[str] | tuple[str, ...] | np.ndarray | None = None,
    ) -> TrackerChunkResult:
        if qpos_ref_resampled is not None:
            qpos_ref = np.asarray(qpos_ref_resampled, dtype=np.float32)
            if qpos_ref.ndim != 2 or qpos_ref.shape[1] != qpos_36.shape[1]:
                raise ValueError(
                    f"qpos_ref_resampled must be (T, {qpos_36.shape[1]}), got {qpos_ref.shape}"
                )
        else:
            qpos_ref = resample_qpos(qpos_36, source_fps=float(reference_fps), target_fps=self.target_fps)
        frame_limit = qpos_ref.shape[0] if steps is None else min(int(steps), qpos_ref.shape[0])
        plan_horizon_frames = int(qpos_ref.shape[0] if plan_horizon is None else plan_horizon)
        if plan_horizon_frames <= 0:
            raise ValueError(f"plan_horizon must be positive, got {plan_horizon_frames}")
        if frame_limit <= 0:
            raise ValueError("No rollout frames requested")
        if overlay_text_by_frame is not None and len(overlay_text_by_frame) < frame_limit:
            raise ValueError(
                f"overlay_text_by_frame must have at least {frame_limit} entries, got {len(overlay_text_by_frame)}"
            )
        if not self.initialized:
            init_qpos = qpos_ref[0] if init_qpos_36 is None else np.asarray(init_qpos_36, dtype=np.float32)
            self.reset(init_qpos)
        ref_features = precompute_reference_features(
            qpos_ref,
            fps=self.target_fps,
            onnx_to_g1=self.holomotion_handles.onnx_to_g1,
        )
        start = len(self.executed_qpos_36)
        for frame_idx in range(frame_limit):
            obs = build_holomotion_obs(
                ref_features=ref_features,
                frame_idx=frame_idx,
                n_fut_frames=self.metadata.n_fut_frames,
                data=self.data,
                g1_handles=self.g1_handles,
                holomotion_handles=self.holomotion_handles,
                last_action_onnx=self.last_action,
                context_length=self.metadata.context_length,
                robot_history=self._obs_history,
            )
            self._clear_external_forces()
            self._apply_push_events(frame_idx, push_events)
            action = self.tracker.run(obs)
            try:
                self.last_action = apply_holomotion_action_pd(
                    model=self.model,
                    data=self.data,
                    holomotion_handles=self.holomotion_handles,
                    action=action,
                    control_substeps=self.control_substeps,
                    action_clip=self.action_clip,
                )
            finally:
                self._clear_external_forces()
            self.executed_qpos_36.append(extract_g1_qpos(self.data, self.g1_handles))
            plan_cursor = int(plan_start) + int(frame_idx)
            if record_buffers:
                self.reference_qpos_36.append(np.asarray(qpos_ref[frame_idx], dtype=np.float32))
                self.actions.append(np.asarray(self.last_action, dtype=np.float32))
                self.plan_cursor.append(plan_cursor)
                self.plan_id.append(int(plan_id))
            
            if frame_callback is not None:
                frame_callback(
                    plan_cursor,
                    self.executed_qpos_36[-1],
                    self.reference_qpos_36[-1],
                    self.actions[-1],
                )
            if self.writer is not None:
                if overlay_text_by_frame is not None:
                    frame_overlay_text = str(overlay_text_by_frame[frame_idx])
                else:
                    frame_overlay_text = self.overlay_text if overlay_text is None else str(overlay_text)
                overlay_lines = [
                    f"text: {frame_overlay_text}" if frame_overlay_text else "",
                        f"plan: {int(plan_id)} cursor: {plan_cursor + 1}/{plan_horizon_frames} global: {len(self.executed_qpos_36)}",
                ]
                append_video_frame(
                    self.renderer,
                    self.camera,
                    self.writer,
                    self.data,
                    track_pos=np.asarray(self.data.xpos[self.g1_handles["pelvis_body_id"]]),
                    camera_view=self.camera_view,
                    follow_mode=self.follow_mode,
                    camera_elevation=self.camera_elevation,
                    overlay_lines=overlay_lines,
                )
        executed = np.asarray(self.executed_qpos_36[start:], dtype=np.float32)
        return TrackerChunkResult(frames=frame_limit, qpos_36=executed)

    def save(
        self,
        *,
        output_path: str | Path,
        reference_path: str | Path,
        mode: str,
        planner_mode: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        return save_tracker_rollout(
            output_path=output_path,
            executed_qpos_36=self.executed_qpos_36,
            reference_qpos_36=self.reference_qpos_36,
            actions=self.actions,
            fps=self.target_fps,
            reference_path=reference_path,
            holomotion_onnx=self.holomotion_onnx,
            robot_xml=self.robot_xml,
            mode=mode,
            plan_cursor=self.plan_cursor,
            plan_id=self.plan_id,
            planner_mode=planner_mode,
            metadata=metadata,
        )


def run_holomotion_tracker(config: TrackerRunConfig) -> TrackerRunResult:
    reference = load_reference_motion(config.reference, fps=config.reference_fps)
    output = Path(config.output).expanduser().resolve() if config.output else default_rollout_path(reference.path)
    video_path = None
    if config.video:
        video_path = Path(config.video_path).expanduser().resolve() if config.video_path else output.with_suffix(".mp4")
    runner = HoloMotionRolloutRunner(
        holomotion_onnx=config.holomotion_onnx,
        target_fps=float(config.target_fps),
        robot_xml=config.robot_xml,
        providers=config.providers,
        control_substeps=int(config.control_substeps),
        action_clip=float(config.action_clip),
        video=bool(config.video),
        video_path=video_path,
        video_width=int(config.video_width),
        video_height=int(config.video_height),
        camera_view=config.camera_view,
        follow_mode=config.follow_mode,
        camera_distance=float(config.camera_distance),
        camera_elevation=float(config.camera_elevation),
        overlay_text=config.overlay_text,
    )
    try:
        result = runner.run_reference_chunk(
            reference.qpos_36,
            reference_fps=reference.fps,
            steps=config.steps,
            init_qpos_36=config.init_qpos_36,
            push_events=config.push_events,
        )
        metadata = dict(config.augmentation_metadata or {})
        metadata.setdefault("push_events", push_events_to_dicts(config.push_events))
        rollout_path = runner.save(
            output_path=output,
            reference_path=reference.path,
            mode=str(config.mode),
            planner_mode=config.planner_mode,
            metadata=metadata,
        )
    finally:
        runner.close()
    return TrackerRunResult(
        rollout_path=rollout_path,
        video_path=video_path,
        frames=result.frames,
        fps=float(config.target_fps),
    )
