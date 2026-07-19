from __future__ import annotations

from collections import OrderedDict
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from omg.benchmarks.evaluator.motion_encoder import MotionEncoder
from omg.benchmarks.lerobot import BENCHMARK_SAMPLE_SCHEMA, build_lerobot_benchmark_views
from omg.benchmarks.evaluator.representation import (
    canonical_body_positions_from_qpos,
    motion_input_dim,
)
from omg.benchmarks.metrics import diversity, motion_fid, motion_kid
from omg.core.paths import resolve_repo_path
from omg.generation.metrics import physical_qpos_metrics
from omg.data.lerobot_dataset import LeRobotG1MotionDataset


@dataclass(frozen=True)
class SampleRecord:
    dataset: str
    index: int | None
    global_index: int | None = None
    caption: str | None = None
    meta: dict[str, Any] | None = None
    schema: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    split: str | None = None
    episode_index: int | None = None
    window_start: int | None = None
    num_frames: int | None = None
    source_id: str | None = None
    segment_index: int | None = None
    source_start_frame: int | None = None
    source_end_frame: int | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: dict[str, Any]
    text_embeddings: np.ndarray | None = None


def _config_dir() -> Path:
    return resolve_repo_path("configs/generation")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return str(value)


def _parse_cfg_scale_value(value: str) -> float | None:
    normalized = str(value).strip().lower()
    if normalized in {"default", "none", "null", "config"}:
        return None
    scale = float(value)
    if not math.isfinite(scale):
        raise ValueError("cfg scale values must be finite")
    return scale


def _cfg_scale_json(value: float | None) -> float | None:
    return None if value is None else float(value)


def _cfg_output_name(value: float | None) -> str:
    if value is None:
        return "cfg_default"
    text = f"{float(value):g}".replace("-", "neg").replace(".", "p")
    return f"cfg_{text}"


def _sample_file_path(output_dir: Path, samples_path: str | None) -> Path:
    return Path(samples_path) if samples_path is not None else output_dir / "samples.jsonl"


def _load_sample_records(path: Path) -> list[SampleRecord]:
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            if payload.get("schema") != BENCHMARK_SAMPLE_SCHEMA:
                raise ValueError(
                    f"Unsupported benchmark sample schema at {path}:{line_no}: "
                    f"{payload.get('schema')!r}; expected {BENCHMARK_SAMPLE_SCHEMA!r}"
                )
            dataset_name = str(payload["dataset"])
            source_dataset = str(payload["source_dataset"])
            if not dataset_name or dataset_name != source_dataset:
                raise ValueError(
                    f"Benchmark dataset/source_dataset mismatch at {path}:{line_no}: "
                    f"{dataset_name!r} != {source_dataset!r}"
                )
            strings = {
                key: str(payload[key])
                for key in ("repo_id", "revision", "split", "source_id")
            }
            empty = [key for key, value in strings.items() if not value.strip()]
            if empty:
                raise ValueError(f"Empty benchmark identity fields at {path}:{line_no}: {empty}")
            integers = {
                key: int(payload[key])
                for key in (
                    "episode_index",
                    "window_start",
                    "num_frames",
                    "segment_index",
                    "source_start_frame",
                    "source_end_frame",
                )
            }
            if integers["episode_index"] < 0 or integers["window_start"] < 0 or integers["num_frames"] <= 0:
                raise ValueError(f"Invalid benchmark window identity at {path}:{line_no}: {integers}")
            if integers["source_start_frame"] < 0 or integers["source_end_frame"] <= integers["source_start_frame"]:
                raise ValueError(f"Invalid benchmark source interval at {path}:{line_no}: {integers}")
            records.append(
                SampleRecord(
                    dataset=dataset_name,
                    index=None,
                    caption=payload.get("caption"),
                    meta=payload.get("meta"),
                    schema=str(payload["schema"]),
                    repo_id=strings["repo_id"],
                    revision=strings["revision"],
                    split=strings["split"],
                    episode_index=integers["episode_index"],
                    window_start=integers["window_start"],
                    num_frames=integers["num_frames"],
                    source_id=strings["source_id"],
                    segment_index=integers["segment_index"],
                    source_start_frame=integers["source_start_frame"],
                    source_end_frame=integers["source_end_frame"],
                )
            )
        except KeyError as exc:
            raise KeyError(f"Missing key {exc} in sample file {path}:{line_no}") from exc
    if not records:
        raise ValueError(f"Sample file is empty: {path}")
    return records


def _write_sample_records(path: Path, records: list[SampleRecord], datasets: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record.index is None:
                raise ValueError(f"Cannot write unresolved benchmark record: {record}")
            dataset = datasets[record.dataset]
            if not hasattr(dataset, "sample_identity"):
                raise TypeError(f"Dataset {record.dataset!r} does not expose stable LeRobot sample identities")
            item = datasets[record.dataset][record.index]
            payload = {
                **dataset.sample_identity(record.index),
                "dataset": record.dataset,
            }
            payload["caption"] = str(item.get("caption", ""))
            payload["meta"] = _jsonable(item.get("meta", {}))
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _resolve_sample_records(records: list[SampleRecord], datasets: dict[str, Any]) -> list[SampleRecord]:
    resolved: list[SampleRecord] = []
    for record in records:
        if record.dataset not in datasets:
            raise KeyError(f"Sample references unknown dataset {record.dataset!r}")
        if record.schema is None:
            if record.index is None:
                raise ValueError(f"Runtime sample record has neither stable identity nor index: {record}")
            resolved.append(record)
            continue
        dataset = datasets[record.dataset]
        if not hasattr(dataset, "resolve_identity"):
            raise TypeError(f"Dataset {record.dataset!r} cannot resolve stable LeRobot sample identities")
        identity = {
            "schema": record.schema,
            "repo_id": record.repo_id,
            "revision": record.revision,
            "split": record.split,
            "episode_index": record.episode_index,
            "window_start": record.window_start,
            "num_frames": record.num_frames,
            "source_dataset": record.dataset,
            "source_id": record.source_id,
            "segment_index": record.segment_index,
            "source_start_frame": record.source_start_frame,
            "source_end_frame": record.source_end_frame,
        }
        resolved.append(replace(record, index=int(dataset.resolve_identity(identity))))
    return resolved


def _dataset_names_from_records(records: list[SampleRecord]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for record in records:
        name = str(record.dataset)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def _validate_sample_records(records: list[SampleRecord], datasets: dict[str, Any]) -> None:
    for record in records:
        if record.dataset not in datasets:
            raise KeyError(f"Sample references unknown dataset '{record.dataset}'")
        if record.index is None:
            raise ValueError(f"Sample record is unresolved: {record}")
        dataset_len = len(datasets[record.dataset])
        if record.index < 0 or record.index >= dataset_len:
            raise IndexError(
                f"Sample index out of range for {record.dataset}: "
                f"{record.index} not in [0, {dataset_len})"
            )


def _dataset_target(cfg: Any) -> str | None:
    if isinstance(cfg, dict):
        return cfg.get("_target_")
    if hasattr(cfg, "get"):
        return cfg.get("_target_")
    return getattr(cfg, "_target_", None)


def _with_representation_rotation(
    dataset_cfg: Any,
    rotation_representation: str | None,
    *,
    num_frames: int | None = None,
) -> Any:
    if _dataset_target(dataset_cfg) != "omg.data.lerobot_dataset.LeRobotG1MotionDataset":
        raise TypeError("Benchmarks require LeRobotG1MotionDataset data configs")
    patched = OmegaConf.to_container(dataset_cfg, resolve=True)
    if not isinstance(patched, dict):
        raise TypeError(f"Expected dataset config to convert to dict, got {type(patched)!r}")
    if rotation_representation is not None:
        patched["rotation_representation"] = str(rotation_representation)
    if num_frames is not None:
        fps = float(patched.get("fps", 30.0))
        patched["sequence_duration"] = int(num_frames) / fps
    return patched


def _build_datasets(
    cfg: Any,
    split: str,
    include: list[str] | None = None,
    *,
    num_frames: int | None = None,
) -> dict[str, Any]:
    dataset_opts = cfg.data.dataset_opts
    if split not in dataset_opts:
        raise KeyError(f"Data config has no split {split!r}")
    rotation_representation = cfg.representation.get("rotation_representation") if "representation" in cfg else None
    configs = list(dataset_opts[split].items())
    if len(configs) != 1:
        raise ValueError(f"LeRobot benchmark config requires exactly one dataset for {split!r}, got {len(configs)}")
    name, dataset_cfg = configs[0]
    dataset = instantiate(
        _with_representation_rotation(
            dataset_cfg,
            rotation_representation,
            num_frames=num_frames,
        )
    )
    if not isinstance(dataset, LeRobotG1MotionDataset):
        raise TypeError(f"Expected LeRobotG1MotionDataset for {name!r}, got {type(dataset).__name__}")
    return build_lerobot_benchmark_views(dataset, include=include)


def _load_model(cfg: Any, ckpt_path: str, device: torch.device):
    model = instantiate(cfg.model)
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict, strict=True)
    if isinstance(payload, dict):
        print(
            f"[INFO] Loaded generation checkpoint path={ckpt_path} "
            f"global_step={payload.get('global_step')} epoch={payload.get('epoch')}"
        )
    return model.to(device).eval()


def _motion_input_dim(motion_key: str) -> int:
    return motion_input_dim(motion_key)


def _motion_encoder_kwargs(config: dict[str, Any], *, input_dim: int, output_dim: int) -> dict[str, Any]:
    kwargs = {"input_dim": input_dim, "output_dim": output_dim}
    for key in (
        "movement_dim",
        "hidden_dim",
        "movement_mode",
        "temporal_kind",
        "num_layers",
        "num_heads",
        "mlp_ratio",
        "dropout",
        "max_len",
    ):
        if key in config:
            kwargs[key] = config[key]
    return kwargs


def _load_evaluator_motion_encoder(
    checkpoint_path: str,
    motion_key: str,
    device: torch.device,
) -> tuple[MotionEncoder, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Evaluator checkpoint must be a dict: {checkpoint_path}")
    for key in ("motion_encoder", "logit_scale"):
        if key not in checkpoint:
            raise KeyError(f"Evaluator checkpoint missing '{key}'")
    config = checkpoint.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError("Evaluator checkpoint config must be a dict")
    ckpt_motion_key = config.get("motion_key")
    if ckpt_motion_key is not None and ckpt_motion_key != motion_key:
        raise ValueError(f"Evaluator was trained with motion_key={ckpt_motion_key!r}, got {motion_key!r}")

    input_dim = motion_input_dim(
        motion_key,
        kinematics_path=str(config.get("kinematics_path", "assets/robots/g1/g1_kinematics.json")),
    )
    output_dim = int(config.get("embedding_dim", 512))
    motion_encoder = MotionEncoder(
        **_motion_encoder_kwargs(config, input_dim=input_dim, output_dim=output_dim)
    ).to(device)
    motion_encoder.load_state_dict(checkpoint["motion_encoder"], strict=True)
    motion_encoder.eval()
    print(f"[INFO] Loaded benchmark motion encoder from {checkpoint_path} (motion_key={motion_key}, dim={output_dim})")
    return motion_encoder, checkpoint


def _motion_for_evaluator(
    qpos_36: torch.Tensor,
    motion_key: str,
    *,
    kinematics: Any,
) -> torch.Tensor:
    if motion_key == "qpos_36":
        return qpos_36
    if motion_key in {"body_pos_local", "body_link_pos_local"}:
        body_pos_local = canonical_body_positions_from_qpos(qpos_36, kinematics)
        if motion_key == "body_link_pos_local":
            return body_pos_local[..., 1:, :]
        return body_pos_local
    raise ValueError(f"Text benchmark cannot derive evaluator motion_key={motion_key!r} from generated qpos_36")


def _encode_motion_embeddings(
    motion_encoder: MotionEncoder,
    motion: torch.Tensor,
    valid: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, motion.shape[0], batch_size),
            desc="Motion embeddings",
            unit="batch",
            leave=False,
        ):
            end = min(start + batch_size, motion.shape[0])
            z = motion_encoder(motion[start:end].to(device), valid_mask=valid[start:end].to(device))
            chunks.append(z.detach().cpu())
    return torch.cat(chunks, dim=0).numpy()


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] == 0:
        raise ValueError("summary stats require a non-empty 1D array")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _physical_values(
    qpos: torch.Tensor,
    fps: torch.Tensor,
    *,
    representation: Any,
    device: torch.device,
    contact_height_threshold: float,
    contact_penetration_tolerance: float,
) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {"contact_sliding_speed": [], "foot_ground_error": [], "body_jerk_mean": []}
    representation = representation.to(device).eval()
    with torch.inference_mode():
        for idx in tqdm(range(qpos.shape[0]), desc="Physical metrics", unit="sample", leave=False):
            metrics = physical_qpos_metrics(
                qpos_36=qpos[idx : idx + 1].to(device),
                representation=representation,
                fps=fps[idx : idx + 1].to(device),
                contact_height_threshold=float(contact_height_threshold),
                contact_penetration_tolerance=float(contact_penetration_tolerance),
            )
            for key in values:
                values[key].append(float(metrics[key].detach().cpu()))
    return {key: np.asarray(metric_values, dtype=np.float64) for key, metric_values in values.items()}


def _physical_summary_from_values(
    values: dict[str, np.ndarray],
    indices: np.ndarray | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, metric_values in values.items():
        selected = metric_values if indices is None else metric_values[indices]
        summary[key] = _summary_stats(selected)
    first = next(iter(values.values()))
    summary["num_samples"] = int(first.shape[0] if indices is None else len(indices))
    return summary


def _qpos_to_body_positions(
    model: Any,
    qpos_36: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, qpos_36.shape[0], int(batch_size)):
            end = min(start + int(batch_size), qpos_36.shape[0])
            fk = model.representation.kinematics.forward_kinematics(qpos_36[start:end].to(device))
            chunks.append(fk["body_pos_w"].detach().cpu())
    return torch.cat(chunks, dim=0)


def _embedding_distribution_metrics(
    reference_embeddings: np.ndarray,
    generated_embeddings: np.ndarray,
) -> dict[str, Any]:
    reference = np.asarray(reference_embeddings)
    generated = np.asarray(generated_embeddings)
    result: dict[str, Any] = {"num_samples": int(generated.shape[0])}
    if reference.shape[0] < 2 or generated.shape[0] < 2:
        result.update({"motion_fid": None, "motion_kid": None, "diversity_generated": None})
        return result
    result.update(
        {
            "motion_fid": motion_fid(reference, generated),
            "motion_kid": motion_kid(reference, generated),
            "diversity_generated": diversity(generated),
        }
    )
    return result


def _dataset_indices(dataset_names: list[str]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(dataset_names):
        groups.setdefault(str(name), []).append(idx)
    return {name: np.asarray(indices, dtype=np.int64) for name, indices in groups.items()}


def _finite_metrics(value: Any, *, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{path}.{key}" if path else str(key)
            _finite_metrics(item, path=current)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _finite_metrics(item, path=f"{path}[{idx}]")
    elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise ValueError(f"Metric is not finite: {path}={value}")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_condition_model(cfg: Any, ckpt_path: str, device: torch.device):
    model = instantiate(cfg.model)
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    if getattr(model, "text_encoder", None) is None:
        text_keys = [key for key in state_dict.keys() if str(key).startswith("text_encoder.")]
        if text_keys:
            state_dict = OrderedDict((key, value) for key, value in state_dict.items() if key not in text_keys)
            print(f"[INFO] Skipped {len(text_keys)} text_encoder parameters for condition-only benchmark")
    model.load_state_dict(state_dict, strict=True)
    if isinstance(payload, dict):
        print(
            f"[INFO] Loaded generation checkpoint path={ckpt_path} "
            f"global_step={payload.get('global_step')} epoch={payload.get('epoch')}"
        )
    return model.to(device).eval()


def output_dir(args: Any, name: str) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path("outputs") / "benchmark" / name / str(args.exp)


def select_condition_records(
    datasets: dict[str, Any],
    *,
    num_samples: int,
    seed: int,
    num_frames: int,
    condition: str,
) -> list[SampleRecord]:
    if int(num_samples) < 1:
        raise ValueError("--num_samples must be positive")
    print(f"[INFO] Selecting {condition} benchmark records with {num_frames} exact valid condition frames…")
    candidates: list[tuple[str, int, int]] = []
    per_dataset_counts: dict[str, int] = {}
    for name, dataset in datasets.items():
        count = 0
        if not hasattr(dataset, "sample_has_condition"):
            raise TypeError(f"Dataset {name!r} is not a LeRobot benchmark view")
        for index in tqdm(
            range(len(dataset)),
            desc=f"Scan {name}",
            unit="idx",
            leave=False,
        ):
            if dataset.sample_has_condition(index, condition, num_frames=num_frames):
                candidates.append((name, index, dataset.global_index(index)))
                count += 1
        per_dataset_counts[name] = count
    if not candidates:
        raise ValueError(f"No {condition} samples with {num_frames} exact valid condition frames found")
    if int(num_samples) > len(candidates):
        print(
            f"[WARN] Requested {num_samples} {condition} samples but only found {len(candidates)}; "
            "evaluating all available samples."
        )
        num_samples = len(candidates)
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(len(candidates), size=int(num_samples), replace=False))
    selected_counts: dict[str, int] = {}
    records: list[SampleRecord] = []
    for candidate_index in selected:
        dataset_name, index, candidate_global = candidates[int(candidate_index)]
        selected_counts[dataset_name] = selected_counts.get(dataset_name, 0) + 1
        records.append(SampleRecord(dataset=dataset_name, index=index, global_index=candidate_global))
    print(f"[INFO] {condition} samples by dataset: {per_dataset_counts}")
    print(f"[INFO] selected {condition} benchmark samples by dataset: {selected_counts}")
    return records


def qpos_to_body_positions(
    model: Any,
    qpos_36: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, qpos_36.shape[0], int(batch_size)),
            desc="Forward kinematics",
            unit="batch",
            leave=False,
        ):
            end = min(start + int(batch_size), qpos_36.shape[0])
            fk = model.representation.kinematics.forward_kinematics(qpos_36[start:end].to(device))
            chunks.append(fk["body_pos_w"].detach().cpu())
    return torch.cat(chunks, dim=0)


def summarize_values(values: dict[str, np.ndarray]) -> dict[str, Any]:
    first = next(iter(values.values()))
    summary = {key: _summary_stats(value) for key, value in values.items()}
    summary["num_samples"] = int(first.shape[0])
    return summary


def conditioned_dataset_metrics(
    *,
    dataset_names: list[str],
    reference_embeddings: np.ndarray,
    generated_embeddings: np.ndarray,
    physical_values: dict[str, np.ndarray],
    condition_values: dict[str, np.ndarray],
    condition_key: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, indices in _dataset_indices(dataset_names).items():
        metrics[name] = {
            "num_samples": int(len(indices)),
            "embedding": _embedding_distribution_metrics(reference_embeddings[indices], generated_embeddings[indices]),
            "physical": _physical_summary_from_values(physical_values, indices),
            condition_key: summarize_values({key: value[indices] for key, value in condition_values.items()}),
        }
    return metrics


def finite_valid(valid: torch.Tensor, num_frames: int) -> torch.Tensor:
    valid = valid[:, : int(num_frames)].detach().cpu().bool()
    if not valid.all():
        raise ValueError("conditioned benchmark requires all requested frames to be valid")
    return valid
