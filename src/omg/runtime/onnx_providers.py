from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
from typing import Sequence

DEFAULT_DIFFUSION_ONNX_PROVIDERS = ("CUDAExecutionProvider",)
DEFAULT_TENSORRT_ONNX_PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider")
DEFAULT_TRACKER_ONNX_PROVIDERS = DEFAULT_TENSORRT_ONNX_PROVIDERS
DEFAULT_ONNX_PROVIDERS = DEFAULT_TENSORRT_ONNX_PROVIDERS
DEFAULT_DIFFUSION_ONNX_PROVIDERS_CSV = ",".join(DEFAULT_DIFFUSION_ONNX_PROVIDERS)
DEFAULT_TRACKER_ONNX_PROVIDERS_CSV = ",".join(DEFAULT_TRACKER_ONNX_PROVIDERS)
DEFAULT_ONNX_PROVIDERS_CSV = DEFAULT_TRACKER_ONNX_PROVIDERS_CSV


def parse_onnx_providers(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return list(DEFAULT_ONNX_PROVIDERS)
    if isinstance(value, str):
        providers = [item.strip() for item in value.split(",") if item.strip()]
    else:
        providers = [str(item).strip() for item in value if str(item).strip()]
    if not providers:
        raise ValueError("At least one ONNX Runtime provider is required")
    return providers


def validate_onnx_providers(provider_list: Sequence[str], available_providers: Sequence[str]) -> None:
    unavailable = sorted(set(provider_list).difference(available_providers))
    if unavailable:
        raise RuntimeError(
            "Requested ONNX Runtime providers are unavailable: "
            + ", ".join(unavailable)
            + f". Available providers: {list(available_providers)}"
        )


def uses_tensorrt_provider(provider_list: Sequence[str]) -> bool:
    return "TensorrtExecutionProvider" in set(provider_list)


def _tensorrt_lib_dir() -> Path:
    spec = importlib.util.find_spec("tensorrt_libs")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(
            "TensorrtExecutionProvider requires the tensorrt-cu12-libs package. "
            "Install it with: python -m pip install --extra-index-url https://pypi.nvidia.com tensorrt-cu12"
        )
    return Path(next(iter(spec.submodule_search_locations)))


def prepare_onnx_provider_runtime(provider_list: Sequence[str]) -> None:
    if not uses_tensorrt_provider(provider_list):
        return
    lib_dir = _tensorrt_lib_dir()
    current = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [item for item in current.split(os.pathsep) if item]
    if str(lib_dir) not in paths:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([str(lib_dir), *paths])
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for lib_name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
        lib_path = lib_dir / lib_name
        if not lib_path.exists():
            raise RuntimeError(f"TensorRT library is missing: {lib_path}")
        ctypes.CDLL(str(lib_path), mode=mode)


def validate_active_onnx_providers(requested_providers: Sequence[str], active_providers: Sequence[str]) -> None:
    missing = [provider for provider in requested_providers if provider not in active_providers]
    if missing:
        raise RuntimeError(
            "Requested ONNX Runtime providers are not active after session creation: "
            + ", ".join(missing)
            + f". Active providers: {list(active_providers)}"
        )
