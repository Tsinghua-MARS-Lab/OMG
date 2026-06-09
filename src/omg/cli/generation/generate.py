from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from omg.data.datamodule import motion_collate_fn


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "generation"


def _load_text(args: argparse.Namespace) -> str:
    if args.labels_json is not None:
        text, _, _ = _load_labels_json_condition(args.labels_json, num_frames=None, fps=args.fps)
    elif args.text is not None:
        text = args.text.strip()
    elif args.text_file is not None:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
    else:
        text = ""
    return text


def _load_model(cfg, ckpt_path: str):
    model = instantiate(cfg.model)
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint with strict=True. "
            "This may mean --condition_injection does not match the checkpoint architecture. "
            "Check whether the checkpoint was trained with default/sum_to_time/separate_to_h/film/control_local_attn injection."
        ) from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def _condition_injection_from_arg(value: str) -> str:
    if value in {"default", "film"}:
        return "per_layer_film"
    return value


def _display_condition_injection(value: str | None) -> str:
    if value is None:
        return "default"
    value = str(value)
    return "film" if value == "per_layer_film" else value


def _condition_injection_banner(mode: str | None, source: str) -> str:
    display = _display_condition_injection(mode)
    return f"[INFO] 🧭 CONDITION_INJECTION={display} ({source}; model.frame_cond_injection={mode})"


def _apply_condition_injection_override(cfg, args: argparse.Namespace) -> None:
    field_name = "frame_cond_injection"
    configured = cfg.model.get(field_name, None)
    if args.condition_injection is not None:
        selected = _condition_injection_from_arg(args.condition_injection)
        cfg.model[field_name] = selected
        print(_condition_injection_banner(selected, f"--condition_injection {args.condition_injection}"))
    else:
        selected = configured if configured is not None else "per_layer_film"
        print(_condition_injection_banner(selected, "config"))


def resolve_cfg_kwargs(args: argparse.Namespace, active_modalities: list[str]) -> dict[str, float | None]:
    if len(active_modalities) == 0:
        return {
            "cfg_scale": None,
            "cfg_text_scale": None,
            "cfg_audio_scale": None,
            "cfg_human_scale": None,
        }

    if len(active_modalities) > 1:
        if any(scale is not None for scale in (args.cfg_text_scale, args.cfg_audio_scale, args.cfg_human_scale)):
            print("[WARN] Multi-modal generation uses unified cfg_scale; per-modality CFG scales are ignored.")
        return {
            "cfg_scale": args.cfg_scale,
            "cfg_text_scale": None,
            "cfg_audio_scale": None,
            "cfg_human_scale": None,
        }

    modality = active_modalities[0]
    if modality == "text":
        scale = args.cfg_text_scale if args.cfg_text_scale is not None else args.cfg_scale
        return {"cfg_scale": None, "cfg_text_scale": scale, "cfg_audio_scale": None, "cfg_human_scale": None}
    if modality == "audio":
        scale = args.cfg_audio_scale if args.cfg_audio_scale is not None else args.cfg_scale
        return {"cfg_scale": None, "cfg_text_scale": None, "cfg_audio_scale": scale, "cfg_human_scale": None}
    if modality == "human":
        scale = args.cfg_human_scale if args.cfg_human_scale is not None else args.cfg_scale
        return {"cfg_scale": None, "cfg_text_scale": None, "cfg_audio_scale": None, "cfg_human_scale": scale}
    raise ValueError(f"Unknown modality: {modality}")


def _round_up_frames(num_frames: int, chunk_len: int) -> int:
    num_frames = int(num_frames)
    chunk_len = int(chunk_len)
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")
    return int(math.ceil(num_frames / chunk_len) * chunk_len)


def _pad_or_trim_feature(features: torch.Tensor, num_frames: int) -> tuple[torch.Tensor, torch.Tensor, int | None]:
    if features.ndim != 2:
        raise ValueError(f"Expected music features with shape (T, D), got {tuple(features.shape)}")
    num_frames = int(num_frames)
    if features.shape[0] >= num_frames:
        valid = torch.ones(num_frames, dtype=torch.bool)
        return features[:num_frames], valid, None
    if features.shape[0] <= 0:
        raise ValueError("Music features must contain at least one frame")
    valid_frames = int(features.shape[0])
    pad = features.new_zeros(num_frames - valid_frames, features.shape[1])
    valid = torch.zeros(num_frames, dtype=torch.bool)
    valid[:valid_frames] = True
    return torch.cat([features, pad], dim=0), valid, valid_frames


def _pad_or_trim_array(array: np.ndarray, num_frames: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    num_frames = int(num_frames)
    if array.shape[0] >= num_frames:
        return array[:num_frames]
    if array.shape[0] <= 0:
        raise ValueError("Array must contain at least one frame")
    pad = np.repeat(array[-1:], num_frames - array.shape[0], axis=0)
    return np.concatenate([array, pad], axis=0)


def _first_label_action(label_data: dict) -> tuple[str, dict]:
    for list_key in ("segments", "actions"):
        segments = label_data.get(list_key, [])
        if isinstance(segments, list) and segments:
            segment = dict(segments[0] or {})
            action = str(segment.get("action", segment.get("caption", segment.get("text", "")))).strip()
            if action:
                return action, segment
    for key in ("action", "caption", "text", "summary", "video_summary"):
        value = str(label_data.get(key, "")).strip()
        if value:
            return value, {}
    raise ValueError("labels_json does not contain a usable action/caption/text")


def _segment_start_frame(segment: dict, fps: float) -> int:
    if "start_frame" in segment or "start frame" in segment:
        return max(0, int(segment.get("start_frame", segment.get("start frame", 0)) or 0))
    start_time = float(segment.get("start_time", segment.get("start time", 0.0)) or 0.0)
    return max(0, int(round(start_time * float(fps))))


def _resolve_gt_motion_from_label(label_path: Path) -> Path:
    candidates: list[Path] = []
    parts = list(label_path.parts)
    for idx, part in enumerate(parts):
        if part in {"labels", "text"}:
            replaced = Path(*parts[:idx], "g1", *parts[idx + 1 :]).with_suffix(".npz")
            candidates.append(replaced)
    dataset_dir = label_path.parent.parent if label_path.parent.name in {"labels", "text"} else label_path.parent
    candidates.extend(
        [
            dataset_dir / "g1" / f"{label_path.stem}.npz",
            dataset_dir / "g1" / f"{label_path.stem}_retarget.npz",
            label_path.with_suffix(".npz"),
        ]
    )
    existing = []
    seen = set()
    for candidate in candidates:
        if candidate.exists():
            key = str(candidate.resolve())
            if key not in seen:
                existing.append(candidate.resolve())
                seen.add(key)
    if len(existing) > 1:
        raise ValueError("Ambiguous GT motion files for labels_json: " + ", ".join(str(path) for path in existing))
    if not existing:
        raise FileNotFoundError(
            "Could not find GT motion npz for labels_json. Tried paths like "
            f"{dataset_dir / 'g1' / (label_path.stem + '.npz')}"
        )
    return existing[0]


def _load_labels_json_condition(
    labels_json: str | Path,
    *,
    num_frames: int | None,
    fps: float,
) -> tuple[str, np.ndarray | None, dict]:
    label_path = Path(labels_json)
    if not label_path.exists():
        raise FileNotFoundError(f"labels_json does not exist: {label_path}")
    with label_path.open("r", encoding="utf-8") as f:
        label_data = json.load(f)
    text, segment = _first_label_action(label_data)
    motion_path = _resolve_gt_motion_from_label(label_path)
    gt_qpos = None
    start = _segment_start_frame(segment, fps)
    with np.load(motion_path) as npz:
        if "qpos" in npz:
            qpos = np.asarray(npz["qpos"], dtype=np.float32)
        elif "qpos_36" in npz:
            qpos = np.asarray(npz["qpos_36"], dtype=np.float32)
        else:
            raise KeyError(f"GT motion file has no qpos/qpos_36 key: {motion_path}")
        source_fps = float(np.asarray(npz["fps"]).reshape(-1)[0]) if "fps" in npz else float(fps)
    available_frames = max(0, int(qpos.shape[0]) - int(start))
    if num_frames is not None:
        gt_qpos = _pad_or_trim_array(qpos[start:], int(num_frames))
    meta = {
        "label_path": str(label_path.resolve()),
        "source_file": str(motion_path.resolve()),
        "window_start": int(start),
        "available_frames": int(available_frames),
        "fps": float(source_fps),
        "segment": segment,
    }
    return text, gt_qpos, meta


def _load_music_feature(path: str | Path, *, audio_dim: int, num_frames: int, start_frame: int = 0) -> tuple[torch.Tensor, torch.Tensor, int | None]:
    music_path = Path(path)
    if not music_path.exists():
        raise FileNotFoundError(f"Music feature file does not exist: {music_path}")
    features_np = np.asarray(np.load(music_path), dtype=np.float32)
    start_frame = max(0, int(start_frame))
    features = torch.from_numpy(features_np[start_frame:])
    if features.ndim != 2 or features.shape[-1] != int(audio_dim):
        raise ValueError(
            f"Expected music feature .npy with shape (T, {int(audio_dim)}), "
            f"got {tuple(features.shape)} at {music_path}"
        )
    return _pad_or_trim_feature(features, num_frames)


def _frame_audio_features(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    fps: int,
    audio_dim: int,
    feature_type: str = "current35",
) -> np.ndarray:
    if audio_dim != 35:
        raise ValueError(f"Raw music encoder currently produces 35-D features, got audio_dim={audio_dim}")
    feature_type = str(feature_type)
    if feature_type == "current35":
        return _frame_audio_features_current35(waveform, sample_rate, fps=fps, audio_dim=audio_dim)
    if feature_type == "aistpp_librosa35":
        return _frame_audio_features_aistpp_librosa35(waveform, sample_rate, fps=fps, audio_dim=audio_dim)
    raise ValueError(f"Unsupported audio feature_type={feature_type!r}")


def _mono_peak_normalized_waveform(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono or stereo waveform, got shape {waveform.shape}")
    if waveform.size == 0:
        raise ValueError("Raw music waveform is empty")
    peak = float(np.max(np.abs(waveform)))
    if peak > 0.0:
        waveform = waveform / peak
    return waveform


def _pad_or_trim_np(features: np.ndarray, frame_count: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] >= frame_count:
        return features[:frame_count]
    pad = np.zeros((frame_count - features.shape[0], features.shape[1]), dtype=np.float32)
    return np.concatenate([features, pad], axis=0)


def _frame_audio_features_current35(waveform: np.ndarray, sample_rate: int, *, fps: int, audio_dim: int) -> np.ndarray:
    waveform = _mono_peak_normalized_waveform(waveform)

    hop = max(1, int(round(float(sample_rate) / float(fps))))
    window_size = max(hop, int(round(0.05 * float(sample_rate))))
    frame_count = max(1, int(np.ceil(waveform.shape[0] / hop)))
    window = np.hanning(window_size).astype(np.float32)
    prev_mag = None
    rows = []
    for frame_idx in range(frame_count):
        start = frame_idx * hop
        end = start + window_size
        frame = waveform[start:end]
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
    return np.stack(rows, axis=0).astype(np.float32)


def _frame_audio_features_aistpp_librosa35(waveform: np.ndarray, sample_rate: int, *, fps: int, audio_dim: int) -> np.ndarray:
    import librosa

    waveform = _mono_peak_normalized_waveform(waveform)
    hop = max(1, int(round(float(sample_rate) / float(fps))))
    frame_count = max(1, int(np.ceil(waveform.shape[0] / hop)))
    window_size = max(hop, int(round(0.05 * float(sample_rate))))
    n_fft = 1 << int(np.ceil(np.log2(max(window_size, 2048))))

    onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate, hop_length=hop, n_fft=n_fft)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=20, hop_length=hop, n_fft=n_fft)
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sample_rate, hop_length=hop, n_fft=n_fft)
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.05, wait=3)
    _, beat_frames = librosa.beat.beat_track(y=waveform, sr=sample_rate, hop_length=hop)

    onset_env = _pad_or_trim_np(onset_env[:, None], frame_count)
    mfcc = _pad_or_trim_np(mfcc.T, frame_count)
    chroma = _pad_or_trim_np(chroma.T, frame_count)
    peak_binary = np.zeros((frame_count, 1), dtype=np.float32)
    beat_binary = np.zeros((frame_count, 1), dtype=np.float32)
    peak_binary[np.asarray(peaks, dtype=np.int64).clip(0, frame_count - 1), 0] = 1.0
    beat_binary[np.asarray(beat_frames, dtype=np.int64).clip(0, frame_count - 1), 0] = 1.0

    features = np.concatenate([onset_env, mfcc, chroma, peak_binary, beat_binary], axis=-1).astype(np.float32)
    if features.shape[-1] != audio_dim:
        raise RuntimeError(f"AIST++ audio feature expected dim {audio_dim}, got {features.shape[-1]}")
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-6)).astype(np.float32)

def _encode_raw_music(
    path: str | Path,
    *,
    fps: int,
    audio_dim: int,
    num_frames: int,
    start_frame: int = 0,
    feature_type: str = "current35",
) -> tuple[torch.Tensor, torch.Tensor, int | None]:
    from scipy.io import wavfile

    music_path = Path(path)
    if not music_path.exists():
        raise FileNotFoundError(f"Raw music file does not exist: {music_path}")
    sample_rate, waveform = wavfile.read(music_path)
    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        waveform = waveform.astype(np.float32) / max(float(max(abs(info.min), info.max)), 1.0)
    else:
        waveform = waveform.astype(np.float32)
    sr = int(sample_rate)
    hop = max(1, int(round(float(sr) / float(fps))))
    start_frame = max(0, int(start_frame))
    start_sample = start_frame * hop
    if waveform.ndim == 1:
        waveform = waveform[start_sample:]
    elif waveform.ndim == 2:
        waveform = waveform[start_sample:, :]
    else:
        raise ValueError(f"Expected mono/stereo waveform, got shape {waveform.shape}")
    features = torch.from_numpy(
        _frame_audio_features(waveform, sr, fps=int(fps), audio_dim=int(audio_dim), feature_type=feature_type)
    )
    return _pad_or_trim_feature(features, num_frames)


def _load_music(
    args: argparse.Namespace,
    cfg,
    dataset,
    meta: dict,
    *,
    start_frame: int = 0,
    num_frames: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int | None, str | None]:
    if not bool(cfg.model.get("use_audio", False)):
        return None, None, None, None
    audio_dim = int(cfg.model.get("audio_dim", 35))
    requested_frames = int(args.num_frames if num_frames is None else num_frames)
    if args.music is not None:
        if args.music_mod == "feature":
            features, has_audio, music_end_frame = _load_music_feature(
                args.music, audio_dim=audio_dim, num_frames=requested_frames, start_frame=start_frame
            )
        elif args.music_mod == "raw":
            features, has_audio, music_end_frame = _encode_raw_music(
                args.music,
                fps=args.fps,
                audio_dim=audio_dim,
                num_frames=requested_frames,
                start_frame=start_frame,
                feature_type=args.music_feature_type,
            )
        else:
            raise ValueError(f"Unsupported music_mod: {args.music_mod}")
        return features, has_audio, music_end_frame, str(Path(args.music).resolve())

    return None, None, None, None


def _strip_retarget_suffix(stem: str) -> str:
    return stem[: -len("_retarget")] if stem.endswith("_retarget") else stem


def _aistpp_music_id(stem: str) -> str | None:
    for token in _strip_retarget_suffix(stem).split("_"):
        if len(token) >= 3 and token[0] == "m" and token[1:].isalnum():
            return token
    return None


def _unique_existing_path(candidates: list[Path], label: str) -> Path | None:
    existing = []
    seen = set()
    for path in candidates:
        if not path.exists():
            continue
        key = str(path.resolve())
        if key not in seen:
            existing.append(path)
            seen.add(key)
    if len(existing) > 1:
        raise ValueError(f"Ambiguous {label} files: " + ", ".join(str(path) for path in existing))
    return existing[0].resolve() if existing else None


def _dataset_dir_from_condition_path(path: Path) -> Path:
    if path.parent.name in {"g1", "music_npy", "music_wav"}:
        return path.parent.parent
    return path.parent


def _resolve_music_wav_from_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    source_path = Path(path)
    if source_path.suffix.lower() == ".wav":
        if not source_path.exists():
            raise FileNotFoundError(f"Music wav file does not exist: {source_path}")
        return source_path.resolve()

    dataset_dir = _dataset_dir_from_condition_path(source_path)
    wav_dir = dataset_dir / "music_wav"
    stem = _strip_retarget_suffix(source_path.stem)
    stems = [stem]
    music_id = _aistpp_music_id(stem)
    if music_id is not None and music_id not in stems:
        stems.append(music_id)
    candidates = [wav_dir / f"{candidate}.wav" for candidate in stems]
    return _unique_existing_path(candidates, f"music wav for {source_path}")


def _resolve_render_music_wav(args: argparse.Namespace, music_path: str | None, meta: dict) -> Path | None:
    if music_path is None:
        return None
    if args.music is not None:
        return _resolve_music_wav_from_path(args.music)
    source_file = meta.get("source_file")
    return _resolve_music_wav_from_path(source_file or music_path)


def _load_human_motion_path(path: str | Path, expected_dim: int) -> np.ndarray:
    ref_path = Path(path)
    if not ref_path.exists():
        raise FileNotFoundError(f"Human motion file does not exist: {ref_path}")
    if ref_path.suffix == ".npy":
        human = np.load(ref_path)
    elif ref_path.suffix == ".npz":
        with np.load(ref_path) as npz:
            for key in ("human_motion", "human_joints", "joints", "poses"):
                if key in npz:
                    human = np.asarray(npz[key])
                    break
            else:
                raise KeyError(f"No human_motion/human_joints/joints/poses key found in {ref_path}")
    else:
        raise ValueError(f"Unsupported human motion extension: {ref_path.suffix}")
    human = np.asarray(human, dtype=np.float32)
    if human.ndim == 3:
        if human.shape[-1] != 3:
            raise ValueError(f"Expected human joints shape (T,J,3), got {human.shape}")
        human = human.reshape(human.shape[0], -1)
    if human.ndim != 2 or human.shape[-1] != int(expected_dim):
        raise ValueError(f"Expected human motion shape (T,{int(expected_dim)}), got {human.shape}")
    return human


def _load_human_motion_from_dataset(dataset, meta: dict) -> tuple[np.ndarray | None, str | None]:
    source = meta.get("source_file")
    if not source:
        return None, None
    source_path = Path(source)
    if not source_path.exists() or not hasattr(dataset, "_load_sequence"):
        return None, None
    sequence = dataset._load_sequence(source_path)
    human = sequence.get("human_motion")
    if human is None:
        return None, None
    return np.asarray(human, dtype=np.float32), str(source_path.resolve())


def _load_audio_from_dataset(dataset, meta: dict) -> tuple[np.ndarray | None, str | None]:
    if not getattr(dataset, "use_audio", False):
        return None, None
    source = meta.get("source_file")
    if not source:
        return None, None
    source_path = Path(source)
    if not source_path.exists() or not hasattr(dataset, "_load_sequence"):
        return None, None
    sequence = dataset._load_sequence(source_path)
    audio = sequence.get("audio_features")
    if audio is None:
        return None, None
    return np.asarray(audio, dtype=np.float32), str(source_path.resolve())


def _slice_condition_array(array: np.ndarray, *, start_frame: int, num_frames: int) -> tuple[torch.Tensor, torch.Tensor, int | None]:
    start_frame = max(0, int(start_frame))
    sliced = torch.from_numpy(np.asarray(array[start_frame:], dtype=np.float32))
    return _pad_or_trim_feature(sliced, int(num_frames))


def _load_human_motion(
    args: argparse.Namespace,
    cfg,
    dataset,
    meta: dict,
    *,
    start_frame: int,
    num_frames: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int | None, str | None]:
    if not bool(cfg.model.get("use_human_motion", False)):
        return None, None, None, None
    expected_dim = int(cfg.model.get("human_motion_dim", 66))
    if args.human_motion is not None:
        human_np = _load_human_motion_path(args.human_motion, expected_dim)
        source = str(Path(args.human_motion).resolve())
    else:
        return None, None, None, None
    if human_np is None:
        raise ValueError(
            "This checkpoint expects human reference conditioning, but no human motion was found. "
            "Pass --human_motion or enable/provide human_motion in the selected dataset config."
        )
    features, has_human, end_frame = _slice_condition_array(human_np, start_frame=start_frame, num_frames=num_frames)
    return features, has_human, end_frame, source


def _attach_music(batch: dict, music_features: torch.Tensor, has_audio: torch.Tensor) -> None:
    if music_features.ndim != 2:
        raise ValueError(f"Expected music_features shape (T, D), got {tuple(music_features.shape)}")
    if has_audio.ndim != 1 or has_audio.shape[0] != music_features.shape[0]:
        raise ValueError(f"Expected has_audio shape ({music_features.shape[0]},), got {tuple(has_audio.shape)}")
    batch["audio_features"] = music_features.unsqueeze(0).to(dtype=torch.float32)
    batch.setdefault("mask", {})
    batch["mask"]["has_audio"] = has_audio.unsqueeze(0).to(dtype=torch.bool)
    batch["mask"]["valid"] = torch.ones(1, music_features.shape[0], dtype=torch.bool)


def _attach_human_motion(batch: dict, human_motion: torch.Tensor, has_human_motion: torch.Tensor) -> None:
    if human_motion.ndim != 2:
        raise ValueError(f"Expected human_motion shape (T, D), got {tuple(human_motion.shape)}")
    if has_human_motion.ndim != 1 or has_human_motion.shape[0] != human_motion.shape[0]:
        raise ValueError(f"Expected has_human_motion shape ({human_motion.shape[0]},), got {tuple(has_human_motion.shape)}")
    batch["human_motion"] = human_motion.unsqueeze(0).to(dtype=torch.float32)
    batch.setdefault("mask", {})
    batch["mask"]["has_human_motion"] = has_human_motion.unsqueeze(0).to(dtype=torch.bool)
    batch["mask"]["valid"] = torch.ones(1, human_motion.shape[0], dtype=torch.bool)


def _resolve_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _mux_wav_audio(
    video_path: Path,
    wav_path: str | Path | None,
    *,
    duration_seconds: float | None = None,
    audio_start_seconds: float | None = None,
) -> Path | None:
    if wav_path is None:
        return None
    wav = Path(wav_path)
    if wav.suffix.lower() != ".wav" or not wav.exists():
        return None
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        return None
    output_path = video_path.with_name(f"{video_path.stem}_with_audio{video_path.suffix}")
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
    ]
    if audio_start_seconds is not None and float(audio_start_seconds) > 0.0:
        cmd.extend(["-ss", f"{float(audio_start_seconds):.6f}"])
    cmd.extend(
        [
            "-i",
            str(wav),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
        ]
    )
    if duration_seconds is not None:
        cmd.extend(["-t", f"{float(duration_seconds):.6f}"])
    cmd.append(str(output_path))
    try:
        subprocess.run(cmd, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return output_path


def _val_dataset(cfg):
    datamodule = instantiate(cfg.data, _recursive_=False)
    dataset_cfg = next(iter(cfg.data.dataset_opts.val.values()))
    return datamodule._instantiate_dataset(dataset_cfg)


def _history_batch(dataset, history_val_index: int | None) -> dict:
    index = 0 if history_val_index is None else int(history_val_index)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"history_val_index={index} out of range [0, {len(dataset)})")
    return motion_collate_fn([dataset[index]])


def _first_meta(batch: dict) -> dict:
    meta = batch.get("meta")
    if isinstance(meta, list) and meta:
        return dict(meta[0])
    return {}


def _jsonable_meta(meta: dict) -> dict:
    out = {}
    for key, value in meta.items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _safe_tag(value: str) -> str:
    value = str(value).strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return "_".join(part for part in safe.split("_") if part)


def _default_output_tag(args: argparse.Namespace) -> str:
    if args.output_tag is not None:
        return _safe_tag(args.output_tag)
    ckpt_path = Path(args.ckpt_path)
    if ckpt_path.parent.name == "checkpoints" and ckpt_path.parent.parent.name:
        return _safe_tag(ckpt_path.parent.parent.name)
    return _safe_tag(ckpt_path.stem)


def _tagged_name(stem: str, tag: str, suffix: str) -> str:
    return f"{stem}_{tag}{suffix}" if tag else f"{stem}{suffix}"


def _load_gt_qpos(meta: dict, num_frames: int) -> np.ndarray | None:
    source = meta.get("source_file")
    if not source:
        return None
    source_path = Path(source)
    if not source_path.exists():
        return None
    start = int(meta.get("window_start", 0))
    with np.load(source_path) as npz:
        if "qpos" not in npz:
            return None
        qpos = np.asarray(npz["qpos"], dtype=np.float32)
    meta["available_frames"] = max(0, int(qpos.shape[0]) - int(start))
    return _pad_or_trim_array(qpos[start:], int(num_frames))


def _render_overlay_lines(
    *,
    text: str,
    condition_injection: str,
    music_path: str | None,
    music_wav_path: Path | None,
    human_motion_path: str | None,
    args: argparse.Namespace,
) -> list[str]:
    lines = [f"condition injection: {_display_condition_injection(condition_injection)}"]
    if text != "":
        lines.append(f"text: {text}")
    if music_path is not None:
        if args.music is not None:
            suffix = Path(music_path).suffix.lower()
            if args.music_mod == "raw" and suffix == ".wav":
                lines.append(f"audio wav: {Path(music_path).name}")
            else:
                lines.append(f"audio feature: {Path(music_path).name}")
        else:
            lines.append(f"audio dataset: {Path(music_path).name}")
    if music_wav_path is not None:
        lines.append(f"audio wav: {music_wav_path.name}")
    if human_motion_path is not None:
        lines.append(f"human motion: {Path(human_motion_path).name}")
    return lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate G1 motion from a trained generation checkpoint.")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--output_root", default="outputs_generate")
    parser.add_argument(
        "--output_tag",
        default=None,
        help="Suffix for generated artifact filenames. Defaults to the checkpoint experiment directory name.",
    )
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    text_group = parser.add_mutually_exclusive_group(required=False)
    text_group.add_argument("--text")
    text_group.add_argument("--text_file")
    text_group.add_argument("--labels_json", help="Use the first action in this labels JSON as text and render its matched GT motion.")
    parser.add_argument(
        "--gt_labels_json",
        default=None,
        help="Optional labels JSON used only to load/render matched GT motion, without using its text as a condition.",
    )
    parser.add_argument("--history_val_index", type=int, default=130)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--cfg_text_scale", type=float, default=None)
    parser.add_argument("--cfg_audio_scale", type=float, default=None)
    parser.add_argument("--cfg_human_scale", type=float, default=None)
    parser.add_argument(
        "--condition_injection",
        choices=["default", "sum_to_time", "separate_to_h", "film", "control_local_attn"],
        default=None,
        help=(
            "Override frame condition injection mode at generation time. "
            "If omitted, use the mode from the Hydra/checkpoint config."
        ),
    )
    parser.add_argument(
        "--music",
        default=None,
        help="Optional override: .npy (T,D) features or .wav with --music_mod raw. "
        "If omitted and model.use_audio is true, audio_features are loaded from the val dataset (same as training). "
        "External .npy is sliced from window_start of --history_val_index; dataset audio uses the same window.",
    )
    parser.add_argument("--music_mod", choices=["feature", "raw"], default="feature")
    parser.add_argument("--music_feature_type", choices=["current35", "aistpp_librosa35"], default="current35")
    parser.add_argument(
        "--disable_audio_condition",
        action="store_true",
        help="Instantiate audio-capable checkpoints but feed null audio at generation time.",
    )
    parser.add_argument("--human_motion", default=None, help="Optional human reference condition path: .npy or .npz with human_motion/human_joints.")
    parser.add_argument(
        "--disable_human_motion_condition",
        action="store_true",
        help="Instantiate human-ref checkpoints but feed null human reference at generation time.",
    )
    parser.add_argument("--render_human_ref", action="store_true", help="Render human reference and robot-vs-human comparison when human conditioning is active.")
    parser.add_argument(
        "--aligned_gt_comparison",
        action="store_true",
        help="Shortcut: save GT qpos slice and render generated-vs-GT comparison video.",
    )
    parser.add_argument("--save_gt_motion", action="store_true", help="Save the GT qpos slice matched by --labels_json.")
    parser.add_argument(
        "--render_comparison_video",
        action="store_true",
        help="Render generated motion on the left and --labels_json matched GT motion on the right.",
    )
    parser.add_argument(
        "--overlay_gt_on_generated",
        action="store_true",
        help="When rendering gen-vs-GT comparison, also draw a translucent GT robot over the generated robot in the left panel.",
    )
    parser.add_argument(
        "--overlay_gt_alpha",
        type=float,
        default=0.28,
        help="Alpha for the translucent GT robot overlaid on the generated robot.",
    )
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera_view", choices=["iso", "side", "both"], default="iso")
    parser.add_argument("--follow_mode", choices=["none", "xy", "xyz"], default="xy")
    parser.add_argument("--scene_preset", choices=["minimal", "studio"], default="studio")
    parser.add_argument("--title", default="G1 Motion")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides, e.g. data=... model.text_encoder.model_name=...")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.aligned_gt_comparison:
        args.save_gt_motion = True
        args.render_comparison_video = True
    if (args.labels_json is not None or args.gt_labels_json is not None) and args.render_video:
        args.save_gt_motion = True
        args.render_comparison_video = True
    if args.render_human_ref:
        args.render_video = True
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    text = _load_text(args)
    with initialize_config_dir(config_dir=str(_config_dir()), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"exp={args.exp}",
                "logger=none",
                "trainer=1gpu",
                *args.overrides,
            ],
        )
    _apply_condition_injection_override(cfg, args)

    dataset = _val_dataset(cfg)
    history_val_index = int(args.history_val_index)

    print(f"[INFO] Checkpoint path: {Path(args.ckpt_path).resolve()}")
    model = _load_model(cfg, args.ckpt_path)
    print(_condition_injection_banner(getattr(model, "frame_cond_injection", None), "loaded model"))
    requested_num_frames = int(args.num_frames)
    sample_num_frames = _round_up_frames(requested_num_frames, int(model.representation.sequence_length))
    batch = _history_batch(dataset, history_val_index)
    history_meta = _jsonable_meta(_first_meta(batch))
    print(history_meta)
    use_audio = bool(cfg.model.get("use_audio", False))
    music_start_frame = int(history_meta.get("window_start", 0)) if use_audio else 0
    condition_start_frame = 0 if args.human_motion is not None else int(history_meta.get("window_start", 0))
    if args.disable_audio_condition and bool(cfg.model.get("use_audio", False)):
        audio_dim = int(cfg.model.get("audio_dim", 35))
        music_features = torch.zeros(sample_num_frames, audio_dim, dtype=torch.float32)
        has_audio = torch.zeros(sample_num_frames, dtype=torch.bool)
        music_end_frame = None
        music_path = None
    else:
        music_features, has_audio, music_end_frame, music_path = _load_music(
            args,
            cfg,
            dataset,
            history_meta,
            start_frame=music_start_frame,
            num_frames=sample_num_frames,
        )
    human_motion, has_human_motion, human_motion_end_frame, human_motion_path = _load_human_motion(
        args,
        cfg,
        dataset,
        history_meta,
        start_frame=condition_start_frame,
        num_frames=sample_num_frames,
    )
    if args.disable_human_motion_condition and bool(cfg.model.get("use_human_motion", False)):
        human_motion_dim = int(cfg.model.get("human_motion_dim", 66))
        human_motion = torch.zeros(sample_num_frames, human_motion_dim, dtype=torch.float32)
        has_human_motion = torch.zeros(sample_num_frames, dtype=torch.bool)
        human_motion_end_frame = None
        human_motion_path = None
    batch["caption"] = [text]
    batch["has_text"] = torch.tensor([text != ""], dtype=torch.bool)
    if music_features is not None:
        _attach_music(batch, music_features, has_audio)
    if human_motion is not None:
        _attach_human_motion(batch, human_motion, has_human_motion)

    active_modalities = []
    if text != "" and cfg.model.get("text_encoder") is not None:
        active_modalities.append("text")
    if music_features is not None and has_audio is not None and bool(has_audio.any().item()):
        active_modalities.append("audio")
    if human_motion is not None and has_human_motion is not None and bool(has_human_motion.any().item()):
        active_modalities.append("human")
    cfg_kwargs = resolve_cfg_kwargs(args, active_modalities)
    if len(active_modalities) > 1 and cfg_kwargs["cfg_scale"] is None:
        cfg_kwargs["cfg_scale"] = float(getattr(model.diffusion, "cfg_scale", 1.0))
    print(f"[INFO] Active modalities: {active_modalities}")
    print(f"[INFO] Resolved CFG scales: {cfg_kwargs}")

    with torch.no_grad():
        sample = model.generate(
            batch,
            num_frames=sample_num_frames,
            **cfg_kwargs,
        )

    output_tag = _default_output_tag(args)
    output_dir = Path(args.output_root) / args.exp / output_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    qpos = sample["qpos_36"].detach().cpu()[:, :requested_num_frames]
    motion_features = sample["motion_features"].detach().cpu()[:, :requested_num_frames]
    labels_text = None
    labels_meta = None
    labels_gt_qpos = None
    gt_labels_json = args.labels_json if args.labels_json is not None else args.gt_labels_json
    if gt_labels_json is not None:
        labels_text, labels_gt_qpos, labels_meta = _load_labels_json_condition(
            gt_labels_json,
            num_frames=requested_num_frames,
            fps=args.fps,
        )
    gt_qpos = labels_gt_qpos
    if gt_labels_json is None and (args.save_gt_motion or args.render_comparison_video):
        print(
            "[WARN] GT rendering/comparison requires --labels_json or --gt_labels_json. "
            "No GT labels path was provided, so GT motion and comparison video will be skipped."
        )
    gt_available_frames = None
    if gt_qpos is not None:
        gt_meta_for_frames = labels_meta if labels_meta is not None else history_meta
        gt_available_frames = int(gt_meta_for_frames.get("available_frames", gt_qpos.shape[0]))
        if gt_available_frames <= 0:
            raise ValueError("Matched GT motion has no frames available for rendering")
    render_num_frames = requested_num_frames if gt_available_frames is None else min(requested_num_frames, gt_available_frames)
    if render_num_frames != requested_num_frames:
        print(
            "[INFO] Clip generated render/save frames to "
            f"{render_num_frames} because matched GT has {gt_available_frames} available frames "
            f"(requested {requested_num_frames})."
        )
    qpos = qpos[:, :render_num_frames]
    motion_features = motion_features[:, :render_num_frames]
    if gt_qpos is not None:
        gt_qpos = gt_qpos[:render_num_frames]
    human_motion_render = None if human_motion is None else human_motion[:render_num_frames].detach().cpu().numpy()
    music_wav_path = _resolve_render_music_wav(args, music_path, history_meta)
    render_overlay_lines = _render_overlay_lines(
        text=text,
        condition_injection=cfg.model.get("frame_cond_injection", "per_layer_film"),
        music_path=music_path,
        music_wav_path=music_wav_path,
        human_motion_path=human_motion_path,
        args=args,
    )
    torch.save({"qpos_36": qpos, "motion_features": motion_features}, output_dir / "sample.pt")
    np.save(output_dir / "qpos_36.npy", qpos[0].numpy().astype(np.float32, copy=False))
    if gt_qpos is not None:
        gt_meta = labels_meta if labels_meta is not None else history_meta
        np.save(output_dir / "gt_qpos_36.npy", gt_qpos.astype(np.float32, copy=False))
        np.savez_compressed(
            output_dir / "gt_reference_motion.npz",
            qpos_36=gt_qpos.astype(np.float32, copy=False),
            fps=np.asarray([float(args.fps)], dtype=np.float32),
            source_file=np.asarray([str(gt_meta.get("source_file", ""))], dtype=np.str_),
            history_val_index=np.asarray([history_val_index], dtype=np.int32),
            window_start=np.asarray([int(gt_meta.get("window_start", 0))], dtype=np.int32),
            window_end=np.asarray([int(gt_meta.get("window_start", 0)) + int(gt_qpos.shape[0])], dtype=np.int32),
            music_path=np.asarray([] if music_path is None else [music_path], dtype=np.str_),
            music_wav_path=np.asarray([] if music_wav_path is None else [str(music_wav_path)], dtype=np.str_),
        )
    np.savez_compressed(
        output_dir / "reference_motion.npz",
        qpos_36=qpos[0].numpy().astype(np.float32, copy=False),
        fps=np.asarray([float(args.fps)], dtype=np.float32),
        text=np.asarray([text], dtype=np.str_),
        exp=np.asarray([str(args.exp)], dtype=np.str_),
        ckpt_path=np.asarray([str(Path(args.ckpt_path).resolve())], dtype=np.str_),
        history_val_index=np.asarray([history_val_index], dtype=np.int32),
        window_start=np.asarray([int(history_meta.get("window_start", 0))], dtype=np.int32),
        source_file=np.asarray([str(history_meta.get("source_file", ""))], dtype=np.str_),
        music_path=np.asarray([] if music_path is None else [music_path], dtype=np.str_),
        music_wav_path=np.asarray([] if music_wav_path is None else [str(music_wav_path)], dtype=np.str_),
        music_mod=np.asarray(
            []
            if music_features is None
            else [args.music_mod if args.music is not None else "dataset"],
            dtype=np.str_,
        ),
    )
    if music_features is not None:
        np.save(output_dir / "music_features.npy", music_features[:render_num_frames].numpy().astype(np.float32, copy=False))
        np.save(output_dir / "has_audio.npy", has_audio[:render_num_frames].numpy().astype(np.bool_, copy=False))
    if human_motion is not None:
        np.save(output_dir / "human_motion.npy", human_motion[:render_num_frames].numpy().astype(np.float32, copy=False))
        np.save(output_dir / "has_human_motion.npy", has_human_motion[:render_num_frames].numpy().astype(np.bool_, copy=False))
    video_path = None
    video_with_audio_path = None
    comparison_video_path = None
    comparison_video_with_audio_path = None
    gt_video_path = None
    human_video_path = None
    robot_human_comparison_video_path = None
    if args.render_video:
        from omg.render.mujoco import render_qpos_video

        condition_end_frame = human_motion_end_frame if human_motion_path is not None else music_end_frame
        ended_message = (
            "Human reference ended; using null reference"
            if human_motion_path is not None and human_motion_end_frame is not None
            else "Music ended; using null audio"
        )
        video_path = render_qpos_video(
            qpos[0],
            output_dir / _tagged_name("qpos_36_mujoco", output_tag, ".mp4"),
            fps=args.fps,
            width=args.width,
            height=args.height,
            camera_view=args.camera_view,
            follow_mode=args.follow_mode,
            scene_preset=args.scene_preset,
            title=args.title,
            overlay_lines=render_overlay_lines,
            music_end_frame=condition_end_frame,
            ended_message=ended_message,
        )

        if music_wav_path is not None:
            audio_start = float(music_start_frame) / float(args.fps) if music_start_frame > 0 else None
            video_with_audio_path = _mux_wav_audio(
                video_path,
                music_wav_path,
                duration_seconds=float(render_num_frames) / float(args.fps),
                audio_start_seconds=audio_start,
            )
            if video_with_audio_path is not None:
                video_path = video_with_audio_path
    if args.render_video and gt_qpos is not None:
        from omg.render.mujoco import render_qpos_video

        gt_video_path = render_qpos_video(
            torch.as_tensor(gt_qpos, dtype=torch.float32),
            output_dir / _tagged_name("gt_qpos_36_mujoco", output_tag, ".mp4"),
            fps=args.fps,
            width=args.width,
            height=args.height,
            camera_view=args.camera_view,
            follow_mode=args.follow_mode,
            scene_preset=args.scene_preset,
            title="Ground Truth",
            overlay_lines=render_overlay_lines,
        )
    if args.render_comparison_video and gt_qpos is not None:
        from omg.render.mujoco import render_qpos_comparison_video

        comparison_video_path = render_qpos_comparison_video(
            qpos[0],
            gt_qpos,
            output_dir
            / _tagged_name(
                "qpos_36_overlay_gt_vs_gt_mujoco" if args.overlay_gt_on_generated else "qpos_36_vs_gt_mujoco",
                output_tag,
                ".mp4",
            ),
            fps=args.fps,
            width=args.width,
            height=args.height,
            camera_view=args.camera_view,
            follow_mode=args.follow_mode,
            scene_preset=args.scene_preset,
            left_title="Generated",
            right_title="Ground Truth",
            overlay_lines=render_overlay_lines,
            music_end_frame=music_end_frame,
            left_ghost_qpos_36=gt_qpos if args.overlay_gt_on_generated else None,
            left_ghost_alpha=args.overlay_gt_alpha,
        )
        if music_wav_path is not None:
            audio_start = float(music_start_frame) / float(args.fps) if music_start_frame > 0 else None
            comparison_video_with_audio_path = _mux_wav_audio(
                comparison_video_path,
                music_wav_path,
                duration_seconds=float(render_num_frames) / float(args.fps),
                audio_start_seconds=audio_start,
            )
            if comparison_video_with_audio_path is not None:
                comparison_video_path = comparison_video_with_audio_path
    if args.render_human_ref:
        if human_motion_render is None or human_motion_path is None:
            raise ValueError("--render_human_ref requires human reference conditioning")
        from omg.render.mujoco import render_human_motion_video, render_robot_human_comparison_video

        human_video_path = render_human_motion_video(
            human_motion_render,
            output_dir / _tagged_name("human_ref", output_tag, ".mp4"),
            fps=args.fps,
            width=args.width,
            height=args.height,
            title="Human Reference",
            overlay_lines=render_overlay_lines,
            ended_frame=human_motion_end_frame,
        )
        robot_human_comparison_video_path = render_robot_human_comparison_video(
            qpos[0],
            human_motion_render,
            output_dir / _tagged_name("qpos_36_vs_human_ref", output_tag, ".mp4"),
            fps=args.fps,
            width=args.width,
            height=args.height,
            camera_view=args.camera_view,
            follow_mode=args.follow_mode,
            scene_preset=args.scene_preset,
            overlay_lines=render_overlay_lines,
            ended_frame=human_motion_end_frame,
            left_ghost_qpos_36=gt_qpos if args.overlay_gt_on_generated else None,
            left_ghost_alpha=args.overlay_gt_alpha,
        )
    metadata = {
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "exp": args.exp,
        "output_tag": output_tag,
        "num_frames": int(args.num_frames),
        "render_num_frames": int(render_num_frames),
        "gt_available_frames": gt_available_frames,
        "sample_num_frames": int(sample_num_frames),
        "seed": int(args.seed),
        "text": text,
        "labels_json": None if args.labels_json is None else str(Path(args.labels_json).resolve()),
        "gt_labels_json": None if args.gt_labels_json is None else str(Path(args.gt_labels_json).resolve()),
        "labels_text": labels_text,
        "labels_meta": labels_meta,
        "history_val_index": history_val_index,
        "aligned_gt_comparison": bool(args.aligned_gt_comparison),
        "overlay_gt_on_generated": bool(args.overlay_gt_on_generated),
        "overlay_gt_alpha": float(args.overlay_gt_alpha),
        "history_meta": history_meta,
        "music_start_frame": int(music_start_frame),
        "active_modalities": active_modalities,
        "condition_injection": cfg.model.get("frame_cond_injection", "per_layer_film"),
        "condition_injection_display": _display_condition_injection(cfg.model.get("frame_cond_injection", None)),
        "condition_injection_override": args.condition_injection,
        "cfg_scale": args.cfg_scale,
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_audio_scale": args.cfg_audio_scale,
        "cfg_human_scale": args.cfg_human_scale,
        "resolved_cfg": cfg_kwargs,
        "music_path": music_path,
        "music_wav_path": None if music_wav_path is None else str(music_wav_path.resolve()),
        "music_mod": None
        if music_features is None
        else (args.music_mod if args.music is not None else "dataset"),
        "music_end_frame": music_end_frame,
        "human_motion_path": human_motion_path,
        "human_motion_end_frame": human_motion_end_frame,
        "music_features_path": None if music_features is None else str((output_dir / "music_features.npy").resolve()),
        "has_audio_path": None if has_audio is None else str((output_dir / "has_audio.npy").resolve()),
        "human_motion_features_path": None if human_motion is None else str((output_dir / "human_motion.npy").resolve()),
        "has_human_motion_path": None if has_human_motion is None else str((output_dir / "has_human_motion.npy").resolve()),
        "gt_qpos_36_path": None if gt_qpos is None else str((output_dir / "gt_qpos_36.npy").resolve()),
        "gt_reference_motion_path": None if gt_qpos is None else str((output_dir / "gt_reference_motion.npz").resolve()),
        "video_path": None if video_path is None else str(video_path.resolve()),
        "video_with_audio_path": None if video_with_audio_path is None else str(video_with_audio_path.resolve()),
        "comparison_video_path": None if comparison_video_path is None else str(comparison_video_path.resolve()),
        "gt_video_path": None if gt_video_path is None else str(gt_video_path.resolve()),
        "comparison_video_with_audio_path": None
        if comparison_video_with_audio_path is None
        else str(comparison_video_with_audio_path.resolve()),
        "human_video_path": None if human_video_path is None else str(human_video_path.resolve()),
        "robot_human_comparison_video_path": None if robot_human_comparison_video_path is None else str(robot_human_comparison_video_path.resolve()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
