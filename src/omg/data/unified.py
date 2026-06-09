from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import yaml

from omg.data.split_yaml import flatten_dataset_split_paths


class UnifiedG1MotionIndex:
    """Index unified OMG-Data source datasets.

    Expected layout:

    ``g1/<entry>.npz``
    ``labels/<entry>.json`` or ``texts/<entry>.txt``
    ``music_npy/<entry>.npy`` for optional audio features
    ``info.yaml`` with train/val/test split entries
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        info_path: str | Path | None = None,
        labels_root: str | Path | None = None,
        text_root: str | Path | None = None,
        sample_by_segment: bool = True,
        include_style_in_caption: bool = True,
        eval_window_policy: str = "uniform",
        eval_num_windows: int = 3,
        skip_missing_labels: bool = False,
        window_size: int = 60,
        default_fps: float = 30.0,
        training: bool = True,
        max_entries: int | None = None,
        recursive_search: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.dataset_dir = self.dataset_root.parent if self.dataset_root.name == "g1" else self.dataset_root
        self.g1_root = self.dataset_dir / "g1"
        self.info_path = Path(info_path) if info_path is not None else self.dataset_dir / "info.yaml"
        self.labels_root = Path(labels_root) if labels_root is not None else self._optional_dir("labels")
        self.text_root = Path(text_root) if text_root is not None else self._optional_dir("text")
        self.audio_dir = self._optional_dir("music_npy")
        self.split = str(split)
        self.sample_by_segment = bool(sample_by_segment)
        self.include_style_in_caption = bool(include_style_in_caption)
        self.eval_window_policy = str(eval_window_policy)
        self.eval_num_windows = max(int(eval_num_windows), 1)
        self.skip_missing_labels = bool(skip_missing_labels)
        self.window_size = int(window_size)
        self.default_fps = float(default_fps)
        self.training = bool(training)
        self.max_entries = None if max_entries is None else int(max_entries)
        self.recursive_search = bool(recursive_search)
        self._motion_files: dict[str, list[Path]] | None = None
        self._label_files: dict[str, list[Path]] | None = None
        self._text_files: dict[str, list[Path]] | None = None
        self._cache_file, self._cache_lock = self._cache_paths()

        cached = self._load_cache()
        if cached is None:
            self.entries, self.samples = self._build_and_cache()
        else:
            self.entries, self.samples = cached
        if self.sample_by_segment and not self.samples:
            source_root = self.labels_root if self.labels_root is not None else self.text_root
            raise ValueError(f"No unified samples found for split `{split}` in {source_root}")

    def _optional_dir(self, name: str) -> Path | None:
        path = self.dataset_dir / name
        return path if path.exists() else None

    @staticmethod
    def _index_files(root: Path | None, suffix: str) -> dict[str, list[Path]]:
        if root is None or not root.exists():
            return {}
        files: dict[str, list[Path]] = {}
        for path in sorted(root.rglob(f"*{suffix}")):
            files.setdefault(path.stem, []).append(path)
        return files

    def _get_motion_files(self) -> dict[str, list[Path]]:
        if self._motion_files is None:
            self._motion_files = self._index_files(self.g1_root, ".npz") if self.recursive_search else {}
        return self._motion_files

    def _get_label_files(self) -> dict[str, list[Path]]:
        if self._label_files is None:
            self._label_files = (
                self._index_files(self.labels_root, ".json") if self.recursive_search and self.labels_root else {}
            )
        return self._label_files

    def _get_text_files(self) -> dict[str, list[Path]]:
        if self._text_files is None:
            self._text_files = self._index_files(self.text_root, ".txt") if self.recursive_search and self.text_root else {}
        return self._text_files

    def _cache_paths(self) -> tuple[Path, Path]:
        cache_root = Path(os.environ.get("OMG_DATASET_CACHE_DIR") or os.environ.get("TMPDIR") or "/tmp")
        cache_root = cache_root / "omg_unified_index_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_dir": str(self.dataset_dir),
            "g1_root": str(self.g1_root),
            "info_path": str(self.info_path),
            "labels_root": str(self.labels_root) if self.labels_root is not None else None,
            "text_root": str(self.text_root) if self.text_root is not None else None,
            "split": self.split,
            "sample_by_segment": self.sample_by_segment,
            "include_style_in_caption": self.include_style_in_caption,
            "eval_window_policy": self.eval_window_policy,
            "eval_num_windows": self.eval_num_windows,
            "skip_missing_labels": self.skip_missing_labels,
            "window_size": self.window_size,
            "default_fps": self.default_fps,
            "training": self.training,
            "max_entries": self.max_entries,
            "recursive_search": self.recursive_search,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        cache_file = cache_root / f"{digest}.pkl"
        lock_file = cache_root / f"{digest}.lock"
        return cache_file, lock_file

    @staticmethod
    def _normalize_records(records: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for record in records:
            normalized.append(
                {key: (str(value) if isinstance(value, Path) else value) for key, value in record.items()}
            )
        return normalized

    def _load_cache(self) -> tuple[list[dict], list[dict]] | None:
        if not self._cache_file.exists():
            return None
        with self._cache_file.open("rb") as f:
            payload = pickle.load(f)
        print(f"[INFO] UnifiedG1MotionIndex loaded cache split={self.split} path={self._cache_file}")
        return payload["entries"], payload["samples"]

    def _write_cache(self, entries: list[dict], samples: list[dict]) -> None:
        tmp_path = self._cache_file.with_suffix(".tmp")
        payload = {
            "entries": self._normalize_records(entries),
            "samples": self._normalize_records(samples),
        }
        with tmp_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self._cache_file)

    def _build_and_cache(self) -> tuple[list[dict], list[dict]]:
        with self._cache_lock.open("a+b") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            cached = self._load_cache()
            if cached is not None:
                return cached
            print(f"[INFO] UnifiedG1MotionIndex building cache split={self.split} path={self._cache_file}")
            entries = self._build_entries()
            samples = self._build_samples(entries) if self.sample_by_segment else []
            self._write_cache(entries, samples)
            return self._load_cache() or (self._normalize_records(entries), self._normalize_records(samples))

    def _load_split_entries(self) -> list[str]:
        if not self.info_path.exists():
            pattern = "**/*.npz" if self.recursive_search else "*.npz"
            entries = sorted(str(path.relative_to(self.g1_root)) for path in self.g1_root.glob(pattern))
            if not entries:
                raise FileNotFoundError(
                    f"Unified dataset info_path does not exist and no motion files were found under {self.g1_root}"
                )
            if self.max_entries is not None:
                entries = entries[: self.max_entries]
            print(
                f"[INFO] Unified dataset info_path does not exist: {self.info_path}; "
                f"using {len(entries)} sorted motion files from {self.g1_root}"
            )
            return entries
        with self.info_path.open("r", encoding="utf-8") as f:
            info = yaml.safe_load(f) or {}
        if self.split not in info:
            raise ValueError(f"Split `{self.split}` not found in {self.info_path}")
        split_data = info[self.split]
        entries = flatten_dataset_split_paths(split_data)
        return entries[: self.max_entries] if self.max_entries is not None else entries

    @staticmethod
    def _entry_to_stem(entry: str) -> str:
        return Path(str(entry)).stem

    @staticmethod
    def _strip_retarget_suffix(stem: str) -> str:
        return stem[: -len("_retarget")] if stem.endswith("_retarget") else stem

    def _resolve_motion_path(self, entry: str) -> Path | None:
        rel = Path(str(entry))
        rel_npz = rel if rel.suffix == ".npz" else rel.with_suffix(".npz")
        direct = self.g1_root / rel_npz
        if direct.exists():
            return direct
        stem = self._entry_to_stem(entry)
        stem_no_retarget = self._strip_retarget_suffix(stem)
        for candidate in (
            self.g1_root / f"{stem}.npz",
            self.g1_root / f"{stem_no_retarget}.npz",
            self.g1_root / f"{stem_no_retarget}_retarget.npz",
        ):
            if candidate.exists():
                return candidate
        motion_files = self._get_motion_files()
        for key in (stem, stem_no_retarget, f"{stem_no_retarget}_retarget"):
            paths = motion_files.get(key, [])
            if len(paths) == 1:
                return paths[0]
            if len(paths) > 1:
                rel_parts = Path(str(entry)).with_suffix("").parts
                matches = [path for path in paths if path.with_suffix("").parts[-len(rel_parts) :] == rel_parts]
                if len(matches) == 1:
                    return matches[0]
                raise ValueError(f"Ambiguous unified motion files for {entry}: {', '.join(str(path) for path in paths)}")
        return None

    def _build_entries(self) -> list[dict]:
        entries = []
        missing = 0
        for entry in self._load_split_entries():
            path = self._resolve_motion_path(entry)
            if path is None:
                missing += 1
                continue
            entries.append({"entry": entry, "path": path, "sequence_name": path.stem, "label_stem": self._entry_to_stem(entry)})
        if not entries:
            raise ValueError(f"No unified motion files found for split `{self.split}` under {self.g1_root}")
        if missing:
            print(f"[INFO] Unified split={self.split} skipped {missing} missing motion files under {self.g1_root}")
        return entries

    def _get_eval_window_starts(self, segment_start: int, segment_end: int) -> list[int]:
        max_offset = segment_end - segment_start - self.window_size
        if max_offset <= 0:
            return [segment_start]
        if self.eval_window_policy == "single":
            offsets = [0]
        elif self.eval_window_policy == "uniform":
            offsets = [0] if self.eval_num_windows == 1 else [
                int(round(x)) for x in np.linspace(0, max_offset, num=self.eval_num_windows).tolist()
            ]
        else:
            raise ValueError(f"Unsupported eval_window_policy: {self.eval_window_policy}")
        return sorted({segment_start + offset for offset in offsets})

    def _resolve_label_or_text_path(self, entry_info: dict) -> Path:
        stems = [
            str(entry_info["label_stem"]),
            str(entry_info["sequence_name"]),
            self._strip_retarget_suffix(str(entry_info["sequence_name"])),
        ]
        rel = Path(str(entry_info["entry"])).with_suffix("")
        for root, suffix in ((self.labels_root, ".json"), (self.text_root, ".txt")):
            if root is None:
                continue
            for stem in (rel, *(Path(stem) for stem in stems)):
                candidate = root / stem.with_suffix(suffix)
                if candidate.exists():
                    return candidate
        label_files = self._get_label_files()
        text_files = self._get_text_files()
        for stem in stems:
            label_paths = label_files.get(stem, [])
            text_paths = text_files.get(stem, [])
            paths = label_paths + text_paths
            if len(paths) == 1:
                return paths[0]
            if len(paths) > 1:
                rel_parts = Path(str(entry_info["entry"])).with_suffix("").parts
                matches = [path for path in paths if path.with_suffix("").parts[-len(rel_parts) :] == rel_parts]
                if len(matches) == 1:
                    return matches[0]
                raise ValueError(
                    f"Ambiguous unified label/text files for {entry_info['entry']}: "
                    + ", ".join(str(path) for path in paths)
                )
        raise FileNotFoundError(
            f"Missing unified label/text file for {entry_info['entry']} "
            f"under labels_root={self.labels_root} text_root={self.text_root}"
        )

    @staticmethod
    def _read_text(path: Path) -> str:
        return " ".join(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    @staticmethod
    def _read_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _summary(label_data: dict) -> str:
        return str(
            label_data.get(
                "video_summary",
                label_data.get("video summary", label_data.get("summary", label_data.get("caption", ""))),
            )
        ).strip()

    def _format_segment_caption(self, segment: dict, video_summary: str) -> str:
        action = str(segment.get("action", segment.get("caption", segment.get("text", "")))).strip()
        style = str(segment.get("style", "")).strip()
        if action and style and self.include_style_in_caption:
            return f"{action}; style: {style}"
        if action:
            return action
        if style:
            return f"style: {style}"
        return video_summary

    def _make_sample(
        self,
        entry_info: dict,
        fps: float,
        total_len: int,
        start_frame: int,
        end_frame: int,
        caption: str,
        *,
        label_path: Path | None,
        segment: dict | None = None,
        segment_index: int = 0,
        video_summary: str = "",
    ) -> list[dict]:
        if start_frame >= total_len or end_frame <= start_frame:
            return []
        window_starts = [start_frame] if self.training else self._get_eval_window_starts(start_frame, end_frame)
        segment = segment or {}
        return [
            {
                **entry_info,
                "fps": fps,
                "label_path": str(label_path) if label_path is not None and label_path.suffix == ".json" else None,
                "text_path": str(label_path) if label_path is not None and label_path.suffix == ".txt" else None,
                "segment_index": segment_index,
                "segment_frame_start": start_frame,
                "segment_frame_end": end_frame,
                "segment_caption": caption,
                "segment_style": str(segment.get("style", "")).strip(),
                "segment_action": str(segment.get("action", caption)).strip(),
                "video_summary": video_summary or caption,
                "eval_window_index": window_index,
                "eval_num_windows": len(window_starts),
                "fixed_window_start": window_start if not self.training else None,
            }
            for window_index, window_start in enumerate(window_starts)
        ]

    def _samples_from_label(self, entry_info: dict, label_path: Path, total_len: int, fps: float) -> list[dict]:
        if label_path.suffix == ".txt":
            caption = self._read_text(label_path)
            return self._make_sample(entry_info, fps, total_len, 0, total_len, caption, label_path=label_path)
        label_data = self._read_json(label_path)
        video_summary = self._summary(label_data)
        segments = label_data.get("segments", [])
        if not segments:
            return self._make_sample(entry_info, fps, total_len, 0, total_len, video_summary, label_path=label_path)
        samples = []
        for segment_index, segment in enumerate(segments):
            start_frame_value = segment.get("start_frame", segment.get("start frame", None))
            end_frame_value = segment.get("end_frame", segment.get("end frame", None))
            if start_frame_value is not None or end_frame_value is not None:
                start_frame = int(start_frame_value or 0)
                end_frame = int(end_frame_value or total_len)
            else:
                start_time = float(segment.get("start_time", segment.get("start time", 0.0)) or 0.0)
                end_time = float(segment.get("end_time", segment.get("end time", total_len / fps)) or (total_len / fps))
                start_frame = int(round(start_time * fps))
                end_frame = int(round(end_time * fps))
            start_frame = max(0, min(start_frame, total_len))
            end_frame = max(start_frame + 1, min(end_frame, total_len))
            samples.extend(
                self._make_sample(
                    entry_info,
                    fps,
                    total_len,
                    start_frame,
                    end_frame,
                    self._format_segment_caption(segment, video_summary),
                    label_path=label_path,
                    segment=segment,
                    segment_index=segment_index,
                    video_summary=video_summary,
                )
            )
        return samples

    def _build_samples(self, entries: list[dict] | None = None) -> list[dict]:
        samples = []
        for entry_info in entries or self.entries:
            try:
                label_path = self._resolve_label_or_text_path(entry_info)
            except FileNotFoundError:
                if self.labels_root is None and self.text_root is None:
                    label_path = None
                elif self.skip_missing_labels:
                    continue
                else:
                    raise
            with np.load(entry_info["path"], mmap_mode="r") as npz:
                total_len = int(npz["qpos"].shape[0])
                fps = float(npz["fps"]) if "fps" in npz else self.default_fps
            if total_len <= 0:
                continue
            if label_path is None:
                samples.extend(
                    self._make_sample(
                        entry_info,
                        fps,
                        total_len,
                        0,
                        total_len,
                        caption="",
                        label_path=None,
                    )
                )
            else:
                samples.extend(self._samples_from_label(entry_info, label_path, total_len, fps))
        return samples
