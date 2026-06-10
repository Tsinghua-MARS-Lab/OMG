from __future__ import annotations

from collections import OrderedDict
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from omg.benchmarks.evaluator.motion_encoder import MotionEncoder
from omg.benchmarks.evaluator.representation import (
    canonical_body_positions_from_qpos,
    motion_input_dim,
)
from omg.benchmarks.metrics import diversity, motion_fid, motion_kid
from omg.core.paths import resolve_repo_path
from omg.generation.metrics import physical_qpos_metrics


@dataclass(frozen=True)
class SampleRecord:
    dataset: str
    index: int
    global_index: int | None = None
    caption: str | None = None
    meta: dict[str, Any] | None = None


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
            records.append(
                SampleRecord(
                    dataset=str(payload["dataset"]),
                    index=int(payload["index"]),
                    global_index=None if payload.get("global_index") is None else int(payload["global_index"]),
                    caption=payload.get("caption"),
                    meta=payload.get("meta"),
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
            item = datasets[record.dataset][record.index]
            payload = asdict(record)
            payload["caption"] = str(item.get("caption", ""))
            payload["meta"] = _jsonable(item.get("meta", {}))
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _dataset_filter_tokens_from_records(records: list[SampleRecord]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for record in records:
        token = str(record.dataset)
        for suffix in ("_train", "_val", "_test"):
            if token.endswith(suffix):
                token = token[: -len(suffix)]
                break
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def _validate_sample_records(records: list[SampleRecord], datasets: dict[str, Any]) -> None:
    for record in records:
        if record.dataset not in datasets:
            raise KeyError(f"Sample references unknown dataset '{record.dataset}'")
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


def _with_representation_rotation(dataset_cfg: Any, rotation_representation: str | None) -> Any:
    if rotation_representation is None:
        return dataset_cfg
    if _dataset_target(dataset_cfg) != "omg.data.g1_motion.G1MotionDataset":
        return dataset_cfg
    patched = OmegaConf.to_container(dataset_cfg, resolve=True)
    if not isinstance(patched, dict):
        raise TypeError(f"Expected dataset config to convert to dict, got {type(patched)!r}")
    patched["rotation_representation"] = str(rotation_representation)
    return patched


def _build_datasets(cfg: Any, split: str, include: list[str] | None = None) -> dict[str, Any]:
    dataset_opts = cfg.data.dataset_opts
    if split not in dataset_opts:
        raise KeyError(f"Data config has no split {split!r}")
    rotation_representation = cfg.representation.get("rotation_representation") if "representation" in cfg else None
    datasets = {}
    for name, dataset_cfg in dataset_opts[split].items():
        if include is not None:
            name_text = str(name).lower()
            if not any(token.lower() in name_text for token in include):
                continue
        dataset = instantiate(_with_representation_rotation(dataset_cfg, rotation_representation))
        if len(dataset) <= 0:
            raise ValueError(f"Dataset {name!r} for split {split!r} is empty")
        datasets[str(name)] = dataset
    if not datasets:
        raise ValueError(f"No datasets selected for split {split!r}")
    return datasets


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


def item_has_condition(item: dict[str, Any], *, tensor_key: str, mask_key: str, num_frames: int) -> bool:
    value = item.get(tensor_key)
    mask = item.get("mask", {}).get(mask_key)
    if value is None or not torch.is_tensor(value):
        return False
    if mask is None or not torch.is_tensor(mask):
        return False
    if value.shape[0] < int(num_frames) or mask.shape[0] < int(num_frames):
        return False
    return bool(mask[: int(num_frames)].all().item())


def select_condition_records(
    datasets: dict[str, Any],
    *,
    num_samples: int,
    seed: int,
    num_frames: int,
    tensor_key: str,
    mask_key: str,
    label: str,
) -> list[SampleRecord]:
    if int(num_samples) < 1:
        raise ValueError("--num_samples must be positive")
    print(f"[INFO] Selecting {label} benchmark records (scanning datasets for valid {tensor_key} frames ≥{num_frames})…")
    candidates: list[tuple[str, int, int]] = []
    global_index = 0
    per_dataset_counts: dict[str, int] = {}
    for name, dataset in datasets.items():
        count = 0
        for index in tqdm(
            range(len(dataset)),
            desc=f"Scan {name}",
            unit="idx",
            leave=False,
        ):
            if item_has_condition(
                dataset[index],
                tensor_key=tensor_key,
                mask_key=mask_key,
                num_frames=num_frames,
            ):
                candidates.append((name, index, global_index))
                count += 1
            global_index += 1
        per_dataset_counts[name] = count
    if not candidates:
        raise ValueError(f"No {label} samples with at least {num_frames} valid condition frames found")
    if int(num_samples) > len(candidates):
        print(
            f"[WARN] Requested {num_samples} {label} samples but only found {len(candidates)}; "
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
    print(f"[INFO] {label} samples by dataset: {per_dataset_counts}")
    print(f"[INFO] selected {label} benchmark samples by dataset: {selected_counts}")
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
