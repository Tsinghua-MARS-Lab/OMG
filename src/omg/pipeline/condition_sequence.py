from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from omg.pipeline.audio import (
    PipelineAudioCondition,
    describe_pipeline_audio_features,
    load_pipeline_audio_features,
    load_pipeline_realtime_audio_wav,
)
from omg.pipeline.human import (
    PipelineHumanMotion,
    describe_pipeline_human_motion,
    human_motion_for_timeline_frame,
    load_pipeline_human_motion,
)


@dataclass(frozen=True)
class PipelineConditionChunk:
    modality: str
    text: str = ""
    source_path: str | None = None
    audio_source_path: str | None = None
    human_source_path: str | None = None
    audio_features: PipelineAudioCondition | None = None
    human_motion: PipelineHumanMotion | None = None
    audio_start_frame: int = 0
    human_start_frame: int = 0
    audio_timeline_key: str | None = None

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "modality": self.modality,
            "text": self.text,
            "source_path": self.source_path,
            "audio_source_path": self.audio_source_path,
            "human_source_path": self.human_source_path,
            "audio_start_frame": int(self.audio_start_frame),
            "human_start_frame": int(self.human_start_frame),
            "audio_timeline_key": self.audio_timeline_key,
        }
        if self.audio_features is not None:
            payload["audio_features"] = describe_pipeline_audio_features(self.audio_features)
        if self.human_motion is not None:
            payload["human_motion"] = describe_pipeline_human_motion(self.human_motion)
        return payload


PipelineConditionSequence = list[PipelineConditionChunk]


@dataclass(frozen=True)
class ParsedConditionChunk:
    modalities: tuple[str, ...]
    value: str
    count: int | None


_PREFIX_RE = re.compile(r"^(?P<modality>[A-Za-z][A-Za-z_+\-]*)(?:\[(?P<count>[0-9]+)\])?$")
_SUPPORTED_MODALITIES = {"text", "audio", "humanref"}
_SUPPORTED_AUDIO_TYPES = {"audio", "feature"}


def _canonical_modality(part: str) -> str:
    modality = part.strip().lower().replace("_", "-")
    if modality in {"human", "human-reference", "humanref"}:
        return "humanref"
    if modality not in _SUPPORTED_MODALITIES:
        raise ValueError(f"Unsupported condition sequence modality {modality!r}")
    return modality


def _parse_modality_parts(raw: str) -> tuple[str, ...]:
    modalities: list[str] = []
    for raw_part in raw.split("+"):
        modality = _canonical_modality(raw_part)
        modalities.append(modality)
    if not modalities:
        raise ValueError("Condition sequence modality is empty")
    if len(set(modalities)) != len(modalities):
        raise ValueError(f"Condition sequence modality contains duplicates: {raw!r}")
    return tuple(modalities)


def _parse_prefix(prefix: str) -> tuple[tuple[str, ...], int | None]:
    match = _PREFIX_RE.fullmatch(prefix.strip())
    if match is None:
        raise ValueError(
            "Condition sequence prefix must be modality or modality[count], "
            f"got {prefix!r}"
        )
    modalities = _parse_modality_parts(match.group("modality"))
    count_text = match.group("count")
    count = None if count_text is None else int(count_text)
    if count is not None and count <= 0:
        raise ValueError(f"Condition sequence repeat count must be positive, got {count}")
    return modalities, count


def _format_modality(modalities: tuple[str, ...]) -> str:
    return "+".join(modalities)


def parse_condition_sequence(raw: str) -> list[ParsedConditionChunk]:
    chunks = [item.strip() for item in str(raw).split("|")]
    chunks = [item for item in chunks if item]
    if not chunks:
        raise ValueError("--condition-sequence must contain at least one non-empty chunk")
    parsed: list[ParsedConditionChunk] = []
    for chunk in chunks:
        if ":" in chunk:
            prefix, value = chunk.split(":", 1)
            modalities, count = _parse_prefix(prefix)
            value = value.strip()
        else:
            modalities = ("text",)
            count = 1
            value = chunk
        if not value:
            raise ValueError(f"Condition sequence chunk {chunk!r} has an empty value")
        parsed.append(
            ParsedConditionChunk(
                modalities=modalities,
                value=value,
                count=count,
            )
        )
    return parsed


def _split_composed_value(modalities: tuple[str, ...], value: str) -> dict[str, str]:
    if len(modalities) == 1:
        return {modalities[0]: value}
    parts = [item.strip() for item in value.split("+", len(modalities) - 1)]
    if len(parts) != len(modalities) or any(not item for item in parts):
        raise ValueError(
            "Composed condition chunks must provide one '+'-separated value per modality: "
            f"{_format_modality(modalities)}: {value!r}"
        )
    return {modality: part for modality, part in zip(modalities, parts, strict=True)}


def _wav_feature_frame_count(path: str | Path, *, target_fps: float) -> int:
    from scipy.io import wavfile

    audio_path = Path(path).expanduser()
    sample_rate, waveform = wavfile.read(audio_path)
    if int(sample_rate) <= 0:
        raise ValueError(f"Audio sample rate must be positive, got {sample_rate} for {audio_path}")
    samples = int(np.asarray(waveform).shape[0])
    if samples <= 0:
        raise ValueError(f"Audio wav is empty: {audio_path}")
    return max(1, int(math.ceil(float(samples) * float(target_fps) / float(sample_rate))))


def _audio_repeat_count_for_wav(
    path: str | Path,
    *,
    target_fps: float,
    audio_step_frames: int,
) -> int:
    audio_frames = _wav_feature_frame_count(path, target_fps=target_fps)
    return max(1, int(math.ceil(float(audio_frames) / float(audio_step_frames))))


def _load_audio_condition(
    path: str | Path,
    *,
    planner: Any,
    target_fps: float,
    audio_feature_type: str,
    num_frames: int,
    audio_type: str,
    cache: dict[tuple[str, int, str | None], PipelineAudioCondition],
) -> PipelineAudioCondition:
    audio_path = Path(path).expanduser()
    if not bool(getattr(planner, "use_audio", False)):
        raise ValueError("audio condition sequence chunks require an audio-conditioned diffusion ONNX")
    if audio_path.suffix.lower() != ".wav":
        raise ValueError(f"audio condition chunks require a .wav path, got {audio_path}")
    mode = str(audio_type).strip().lower()
    if mode not in _SUPPORTED_AUDIO_TYPES:
        raise ValueError(f"Unsupported audio_type={audio_type!r}; expected audio or feature")
    key = (str(audio_path), int(num_frames), f"{mode}:{audio_feature_type}")
    if key not in cache:
        if mode == "audio":
            cache[key] = load_pipeline_realtime_audio_wav(
                audio_path,
                fps=float(target_fps),
                audio_dim=int(planner.audio_dim),
            )
        else:
            cache[key] = load_pipeline_audio_features(
                audio_path,
                source_type="wav",
                fps=float(target_fps),
                audio_dim=int(planner.audio_dim),
                num_frames=int(num_frames),
                feature_type=str(audio_feature_type),
            )
    return cache[key]


def _load_human_condition(
    path: str | Path,
    *,
    planner: Any,
    target_fps: float,
    num_frames_per_chunk: int,
) -> PipelineHumanMotion:
    human_path = Path(path).expanduser()
    if not bool(getattr(planner, "use_human_motion", False)):
        raise ValueError("humanref condition sequence chunks require a human-reference-conditioned diffusion ONNX")
    return load_pipeline_human_motion(
        human_path,
        fps=float(target_fps),
        human_motion_dim=int(planner.human_motion_dim),
        num_frames=int(num_frames_per_chunk),
    )


def load_pipeline_condition_sequence(
    raw: str,
    *,
    planner: Any,
    target_fps: float,
    audio_feature_type: str,
    num_frames_per_chunk: int,
    audio_step_frames: int | None = None,
    audio_type: str = "audio",
) -> PipelineConditionSequence:
    sequence: PipelineConditionSequence = []
    audio_offsets: dict[tuple[str, str], int] = {}
    human_offsets: dict[tuple[str, str], int] = {}
    audio_cache: dict[tuple[str, int, str | None], PipelineAudioCondition] = {}
    audio_type = str(audio_type).strip().lower()
    if audio_type not in _SUPPORTED_AUDIO_TYPES:
        raise ValueError(f"Unsupported audio_type={audio_type!r}; expected audio or feature")
    audio_step = int(num_frames_per_chunk if audio_step_frames is None else audio_step_frames)
    if audio_step <= 0:
        raise ValueError(f"audio_step_frames must be positive, got {audio_step}")
    audio_timeline_group = 0
    for spec in parse_condition_sequence(raw):
        values = _split_composed_value(spec.modalities, spec.value)
        text = values.get("text", "")
        audio_source_path = str(Path(values["audio"]).expanduser()) if "audio" in values else None
        human_source_path = str(Path(values["humanref"]).expanduser()) if "humanref" in values else None
        if spec.count is None and audio_source_path is not None:
            repeat_count = _audio_repeat_count_for_wav(
                audio_source_path,
                target_fps=target_fps,
                audio_step_frames=audio_step,
            )
        else:
            repeat_count = 1 if spec.count is None else int(spec.count)
        audio_num_frames = (int(repeat_count) - 1) * audio_step + int(num_frames_per_chunk)
        audio = None
        audio_timeline_key = None
        if audio_source_path is not None:
            audio_timeline_key = f"audio:{audio_timeline_group}"
            audio_timeline_group += 1
        if audio_source_path is not None:
            audio = _load_audio_condition(
                audio_source_path,
                planner=planner,
                target_fps=target_fps,
                audio_feature_type=audio_feature_type,
                num_frames=audio_num_frames,
                audio_type=audio_type,
                cache=audio_cache,
            )
        human_motion = None
        if human_source_path is not None:
            human_motion = _load_human_condition(
                human_source_path,
                planner=planner,
                target_fps=target_fps,
                num_frames_per_chunk=num_frames_per_chunk,
            )

        for _ in range(int(repeat_count)):
            audio_start_frame = 0
            human_start_frame = 0
            if audio_source_path is not None:
                audio_key = (_format_modality(spec.modalities), audio_source_path)
                audio_start_frame = int(audio_offsets.get(audio_key, 0))
                audio_offsets[audio_key] = audio_start_frame + audio_step
            if human_source_path is not None:
                human_key = (_format_modality(spec.modalities), human_source_path)
                human_start_frame = int(human_offsets.get(human_key, 0))
                human_offsets[human_key] = human_start_frame + int(num_frames_per_chunk)
            source_paths = [path for path in (audio_source_path, human_source_path) if path is not None]
            sequence.append(
                PipelineConditionChunk(
                    modality=_format_modality(spec.modalities),
                    text=text,
                    source_path="+".join(source_paths) if source_paths else None,
                    audio_source_path=audio_source_path,
                    human_source_path=human_source_path,
                    audio_features=audio,
                    human_motion=human_motion,
                    audio_start_frame=audio_start_frame,
                    human_start_frame=human_start_frame,
                    audio_timeline_key=audio_timeline_key,
                )
            )
    return sequence


def condition_sequence_for_plan(
    condition_sequence: PipelineConditionSequence | None,
    plan_index: int,
) -> PipelineConditionChunk | None:
    if condition_sequence is None:
        return None
    index = int(plan_index)
    if index < 0 or index >= len(condition_sequence):
        raise ValueError(
            f"Condition sequence has {len(condition_sequence)} chunks, but plan index {index} was requested"
        )
    return condition_sequence[index]


def condition_sequence_text(
    condition_sequence: PipelineConditionSequence | None,
    fallback_text: str,
    plan_index: int,
) -> str:
    chunk = condition_sequence_for_plan(condition_sequence, int(plan_index))
    if chunk is None:
        return str(fallback_text)
    return chunk.text


def condition_sequence_audio(
    condition_sequence: PipelineConditionSequence | None,
    plan_index: int,
    *,
    request_tracker_frame: int | None = None,
    target_fps: float,
    tracker_fps: float,
    num_frames: int,
    timeline_starts: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    chunk = condition_sequence_for_plan(condition_sequence, int(plan_index))
    if chunk is None or chunk.audio_features is None:
        return None
    start_frame = int(chunk.audio_start_frame)
    if chunk.audio_timeline_key is not None and request_tracker_frame is not None:
        if timeline_starts is not None:
            segment_start = int(timeline_starts.setdefault(chunk.audio_timeline_key, int(request_tracker_frame)))
        else:
            segment_start = int(request_tracker_frame)
        elapsed_tracker_frames = max(0, int(request_tracker_frame) - segment_start)
        start_frame = int(np.floor(float(elapsed_tracker_frames) * float(target_fps) / float(tracker_fps) + 1e-9))
    return chunk.audio_features.features_for_frame(start_frame, num_frames=int(num_frames))


def condition_sequence_human(
    condition_sequence: PipelineConditionSequence | None,
    plan_index: int,
    *,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    chunk = condition_sequence_for_plan(condition_sequence, int(plan_index))
    if chunk is None or chunk.human_motion is None:
        return None
    return human_motion_for_timeline_frame(
        chunk.human_motion,
        int(chunk.human_start_frame),
        num_frames=int(num_frames),
    )


def describe_condition_sequence(condition_sequence: PipelineConditionSequence | None) -> dict[str, Any] | None:
    if condition_sequence is None:
        return None
    return {
        "chunks": [chunk.describe() for chunk in condition_sequence],
        "length": len(condition_sequence),
    }
