from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _mono_waveform(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono or stereo waveform, got shape {waveform.shape}")
    if waveform.size == 0:
        raise ValueError("Raw audio waveform is empty")
    return waveform


def _mono_peak_normalized_waveform(waveform: np.ndarray) -> np.ndarray:
    waveform = _mono_waveform(waveform)
    peak = float(np.max(np.abs(waveform)))
    if peak > 0.0:
        waveform = waveform / peak
    return waveform


def _frame_audio_features_current35(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    fps: int,
    audio_dim: int,
) -> np.ndarray:
    waveform = _mono_peak_normalized_waveform(waveform)
    hop = max(1, int(round(float(sample_rate) / float(fps))))
    window_size = max(hop, int(round(0.05 * float(sample_rate))))
    frame_count = max(1, int(np.ceil(waveform.shape[0] / hop)))
    window = np.hanning(window_size).astype(np.float32)
    prev_mag = None
    rows = []
    for frame_idx in range(frame_count):
        start = frame_idx * hop
        frame = waveform[start : start + window_size]
        if frame.shape[0] < window_size:
            frame = np.pad(frame, (0, window_size - frame.shape[0]))
        frame = frame * window
        rms = np.sqrt(np.mean(frame * frame) + 1e-8)
        zcr = np.mean(np.abs(np.diff(np.signbit(frame).astype(np.float32)))) if frame.shape[0] > 1 else 0.0
        mag = np.abs(np.fft.rfft(frame)).astype(np.float32)
        mag_sum = float(mag.sum() + 1e-8)
        flux = 0.0 if prev_mag is None else float(np.sqrt(np.mean((mag - prev_mag) ** 2)))
        prev_mag = mag

        bands = np.array_split(mag, 32)
        band_energy = np.asarray([float(np.mean(band * band)) if band.size else 0.0 for band in bands], dtype=np.float32)
        band_energy = np.log1p(band_energy)
        band_energy = band_energy / max(float(np.max(band_energy)), 1e-6)
        centroid = float((np.arange(mag.shape[0], dtype=np.float32) * mag).sum() / mag_sum)
        centroid = centroid / max(float(mag.shape[0] - 1), 1.0)
        rows.append(np.concatenate([band_energy, np.asarray([rms, zcr, flux + centroid], dtype=np.float32)]))
    features = np.stack(rows, axis=0).astype(np.float32)
    if features.shape[-1] != int(audio_dim):
        raise RuntimeError(f"current35 audio feature expected dim {audio_dim}, got {features.shape[-1]}")
    return features


def frame_audio_features(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    fps: int,
    audio_dim: int,
    feature_type: str,
) -> np.ndarray:
    if int(audio_dim) != 35:
        raise ValueError(f"Raw audio encoders currently produce 35-D features, got audio_dim={audio_dim}")
    if feature_type == "current35":
        return _frame_audio_features_current35(waveform, int(sample_rate), fps=int(fps), audio_dim=int(audio_dim))
    raise ValueError(f"Unsupported audio feature type: {feature_type!r}")


def _read_wav_float32(path: str | Path) -> tuple[int, np.ndarray]:
    from scipy.io import wavfile

    audio_path = Path(path).expanduser()
    sample_rate, waveform = wavfile.read(audio_path)
    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        waveform = waveform.astype(np.float32) / max(float(max(abs(info.min), info.max)), 1.0)
    else:
        waveform = waveform.astype(np.float32)
    return int(sample_rate), waveform


def _pad_or_trim_features(features: np.ndarray, num_frames: int) -> tuple[np.ndarray, np.ndarray, int | None]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected audio features with shape (T,D), got {features.shape}")
    frames = int(num_frames)
    if frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if features.shape[0] >= frames:
        return features[:frames].astype(np.float32, copy=False), np.ones(frames, dtype=bool), None
    if features.shape[0] <= 0:
        raise ValueError("Audio features must contain at least one frame")
    valid_frames = int(features.shape[0])
    pad = np.zeros((frames - valid_frames, features.shape[1]), dtype=np.float32)
    valid = np.zeros(frames, dtype=bool)
    valid[:valid_frames] = True
    return np.concatenate([features, pad], axis=0).astype(np.float32, copy=False), valid, valid_frames


@dataclass(frozen=True)
class PipelineAudioFeatures:
    features: np.ndarray
    mask: np.ndarray
    source_path: str
    source_type: str
    feature_type: str | None
    fps: float
    padded_from_frames: int | None = None
    realtime_timeline: bool = False

    def features_for_plan(
        self,
        plan_index: int,
        *,
        num_frames: int,
        sequence_length: int,
        allow_multi_chunk: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(self.features, dtype=np.float32)
        mask = np.asarray(self.mask, dtype=bool)
        if allow_multi_chunk:
            return features[: int(num_frames)], mask[: int(num_frames)]
        index = int(plan_index)
        start = index * int(sequence_length)
        end = start + int(num_frames)
        if end > features.shape[0]:
            raise ValueError(
                f"Audio condition has {features.shape[0]} frames, but plan {index} requests [{start}, {end})"
            )
        return features[start:end], mask[start:end]

    def features_for_frame(self, start_frame: int, *, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(self.features, dtype=np.float32)
        mask = np.asarray(self.mask, dtype=bool)
        start = int(start_frame)
        frames = int(num_frames)
        if start < 0:
            raise ValueError(f"start_frame must be non-negative, got {start_frame}")
        if frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        end = start + frames
        if end > features.shape[0]:
            raise ValueError(
                f"Realtime audio condition has {features.shape[0]} frames, but requests [{start}, {end})"
            )
        return features[start:end], mask[start:end]

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.source_type,
            "path": self.source_path,
            "feature_type": self.feature_type,
            "fps": float(self.fps),
            "shape": list(np.asarray(self.features).shape),
            "valid_frames": int(np.asarray(self.mask, dtype=bool).sum()),
            "padded_from_frames": self.padded_from_frames,
            "realtime_timeline": bool(self.realtime_timeline),
        }


@dataclass(frozen=True)
class PipelineRealtimeAudioWav:
    waveform: np.ndarray
    sample_rate: int
    source_path: str
    fps: float
    audio_dim: int
    feature_type: str = "current35"
    realtime_timeline: bool = True

    def features_for_frame(self, start_frame: int, *, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(start_frame)
        frames = int(num_frames)
        if start < 0:
            raise ValueError(f"start_frame must be non-negative, got {start_frame}")
        if frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        sample_start = int(round(float(start) * float(self.sample_rate) / float(self.fps)))
        sample_count = int(round(float(frames) * float(self.sample_rate) / float(self.fps)))
        sample_end = sample_start + sample_count
        waveform = np.asarray(self.waveform, dtype=np.float32)
        valid_samples = max(0, min(sample_end, int(waveform.shape[0])) - sample_start)
        valid_frames = max(
            0,
            min(
                frames,
                int(np.ceil(float(valid_samples) * float(self.fps) / float(self.sample_rate))),
            ),
        )
        frame_waveform = waveform[sample_start:min(sample_end, int(waveform.shape[0]))]
        if frame_waveform.shape[0] < sample_count:
            frame_waveform = np.pad(frame_waveform, (0, sample_count - frame_waveform.shape[0]))
        features = frame_audio_features(
            frame_waveform,
            int(self.sample_rate),
            fps=int(round(float(self.fps))),
            audio_dim=int(self.audio_dim),
            feature_type=self.feature_type,
        )
        if features.shape[0] != frames:
            raise RuntimeError(f"Realtime audio feature expected {frames} frames, got {features.shape[0]}")
        mask = np.zeros(frames, dtype=bool)
        mask[:valid_frames] = True
        return features, mask

    def describe(self) -> dict[str, Any]:
        waveform = np.asarray(self.waveform)
        return {
            "type": "realtime_wav",
            "path": self.source_path,
            "feature_type": self.feature_type,
            "fps": float(self.fps),
            "audio_dim": int(self.audio_dim),
            "sample_rate": int(self.sample_rate),
            "samples": int(waveform.shape[0]),
            "duration_seconds": float(waveform.shape[0]) / float(self.sample_rate),
            "realtime_timeline": True,
            "feature_extraction": "per_replan_current35",
        }


PipelineAudioCondition = PipelineAudioFeatures | PipelineRealtimeAudioWav


def load_pipeline_audio_features(
    path: str | Path,
    *,
    source_type: str,
    fps: float,
    audio_dim: int,
    num_frames: int,
    feature_type: str = "current35",
    realtime_timeline: bool = False,
) -> PipelineAudioFeatures:
    audio_path = Path(path).expanduser()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio condition path does not exist: {audio_path}")
    if source_type == "feature":
        features = np.asarray(np.load(audio_path), dtype=np.float32)
        if features.ndim != 2 or features.shape[-1] != int(audio_dim):
            raise ValueError(f"Expected audio feature .npy shape (T,{int(audio_dim)}), got {features.shape}")
        encoded_feature_type = None
    elif source_type == "wav":
        sample_rate, waveform = _read_wav_float32(audio_path)
        features = frame_audio_features(
            waveform,
            int(sample_rate),
            fps=int(round(float(fps))),
            audio_dim=int(audio_dim),
            feature_type=str(feature_type),
        )
        encoded_feature_type = str(feature_type)
    else:
        raise ValueError(f"Unsupported audio source_type={source_type!r}")
    features, mask, padded_from_frames = _pad_or_trim_features(features, int(num_frames))
    return PipelineAudioFeatures(
        features=features,
        mask=mask,
        source_path=str(audio_path),
        source_type=str(source_type),
        feature_type=encoded_feature_type,
        fps=float(fps),
        padded_from_frames=padded_from_frames,
        realtime_timeline=bool(realtime_timeline),
    )


def load_pipeline_realtime_audio_wav(
    path: str | Path,
    *,
    fps: float,
    audio_dim: int,
) -> PipelineRealtimeAudioWav:
    audio_path = Path(path).expanduser()
    if not audio_path.exists():
        raise FileNotFoundError(f"Realtime audio wav path does not exist: {audio_path}")
    sample_rate, waveform = _read_wav_float32(audio_path)
    return PipelineRealtimeAudioWav(
        waveform=_mono_waveform(waveform),
        sample_rate=sample_rate,
        source_path=str(audio_path),
        fps=float(fps),
        audio_dim=int(audio_dim),
        feature_type="current35",
    )


def audio_features_for_plan(
    audio_features: PipelineAudioCondition | None,
    plan_index: int,
    *,
    num_frames: int,
    sequence_length: int,
    allow_multi_chunk: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    if audio_features is None:
        return None
    if isinstance(audio_features, PipelineRealtimeAudioWav):
        if allow_multi_chunk:
            return audio_features.features_for_frame(0, num_frames=int(num_frames))
        return audio_features.features_for_frame(
            int(plan_index) * int(sequence_length),
            num_frames=int(num_frames),
        )
    return audio_features.features_for_plan(
        int(plan_index),
        num_frames=int(num_frames),
        sequence_length=int(sequence_length),
        allow_multi_chunk=bool(allow_multi_chunk),
    )


def audio_features_for_timeline_frame(
    audio_features: PipelineAudioCondition | None,
    start_frame: int,
    *,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if audio_features is None:
        return None
    return audio_features.features_for_frame(int(start_frame), num_frames=int(num_frames))


def describe_pipeline_audio_features(audio_features: PipelineAudioCondition | None) -> dict[str, Any] | None:
    if audio_features is None:
        return None
    return audio_features.describe()
