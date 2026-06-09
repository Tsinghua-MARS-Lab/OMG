from __future__ import annotations

import argparse
import os
from pathlib import Path

from omg.realtime.planner_service import RealtimeDiffusionPlannerService, RealtimePlannerConfig
from omg.realtime.transport import ZmqPlanServer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 4090/A800 realtime OMG diffusion planner server.")
    parser.add_argument("--bind", default="tcp://0.0.0.0:5555")
    parser.add_argument("--diffusion-onnx", required=True)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Frames generated per request. Defaults to the exported ONNX sequence length.",
    )
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--cfg-text-scale", type=float, default=None)
    parser.add_argument("--cfg-audio-scale", type=float, default=None)
    parser.add_argument("--cfg-human-scale", type=float, default=None)
    parser.add_argument(
        "--providers",
        default=None,
        help="ONNX Runtime providers. Defaults to TensorRT then CUDA for TensorRT-compatible exports.",
    )
    parser.add_argument("--text-encoder-model", default=None)
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensorrt-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorrt-engine-cache-path", default="tensorrt_engine_cache/realtime_planner")
    parser.add_argument("--dit-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dit-cache-threshold", type=float, default=0.995)
    parser.add_argument("--dit-cache-warmup-steps", type=int, default=4)
    parser.add_argument("--dit-cache-max-consecutive", type=int, default=2)
    parser.add_argument("--include-motion-features", action="store_true")
    parser.add_argument("--log-jsonl", default=None)
    return parser.parse_args()


def _format_timing(timing_ms: dict) -> str:
    return (
        f"t5={float(timing_ms.get('text_encode_ms', 0.0)):.3f}ms "
        f"hist={float(timing_ms.get('canonicalization_ms', 0.0)):.3f}ms "
        f"sample={float(timing_ms.get('diffusion_sampling_total_ms', 0.0)):.3f}ms "
        f"onnx={float(timing_ms.get('diffusion_infer_ms', 0.0)):.3f}ms/"
        f"{int(timing_ms.get('onnx_run_count', 0))}runs "
        f"onnx_mean={float(timing_ms.get('onnx_run_mean_ms', 0.0)):.3f}ms "
        f"update={float(timing_ms.get('diffusion_update_ms', 0.0)):.3f}ms "
        f"decode={float(timing_ms.get('interpolation_ms', 0.0)):.3f}ms "
        f"hist2={float(timing_ms.get('rollout_history_ms', 0.0)):.3f}ms "
        f"concat={float(timing_ms.get('concat_output_ms', 0.0)):.3f}ms "
        f"total={float(timing_ms.get('total_ms', 0.0)):.3f}ms"
    )


def _format_audio(audio_metadata: object) -> str:
    if not isinstance(audio_metadata, dict):
        return ""
    start = audio_metadata.get("start_frame")
    end = audio_metadata.get("end_frame")
    if start is None or end is None:
        return ""
    return f" audio=[{int(start):05d},{int(end):05d})"


def _format_condition(condition_metadata: object) -> str:
    if not isinstance(condition_metadata, dict):
        return ""
    index = condition_metadata.get("index")
    chunk = condition_metadata.get("chunk")
    modality = chunk.get("modality") if isinstance(chunk, dict) else None
    if index is None or modality is None:
        return ""
    suffix = " held_last" if bool(condition_metadata.get("held_last_chunk", False)) else ""
    return f" cond={int(index):04d}:{modality}{suffix}"


def main() -> None:
    args = _parse_args()
    config = RealtimePlannerConfig(
        diffusion_onnx=args.diffusion_onnx,
        num_frames=args.num_frames,
        cfg_scale=args.cfg_scale,
        cfg_text_scale=args.cfg_text_scale,
        cfg_audio_scale=args.cfg_audio_scale,
        cfg_human_scale=args.cfg_human_scale,
        providers=args.providers,
        text_encoder_model=args.text_encoder_model,
        torch_device=args.torch_device,
        seed=args.seed,
        tensorrt_fp16=bool(args.tensorrt_fp16),
        tensorrt_engine_cache_path=args.tensorrt_engine_cache_path,
        dit_cache=bool(args.dit_cache),
        dit_cache_threshold=args.dit_cache_threshold,
        dit_cache_warmup_steps=args.dit_cache_warmup_steps,
        dit_cache_max_consecutive=args.dit_cache_max_consecutive,
        include_motion_features=bool(args.include_motion_features),
    )
    service = RealtimeDiffusionPlannerService(config)
    with ZmqPlanServer(args.bind) as server:
        print(
            f"[realtime-planner] bind={args.bind} onnx={Path(args.diffusion_onnx).expanduser()} "
            f"frames={service.plan_frames} tensorrt_fp16={bool(args.tensorrt_fp16)} dit_cache={bool(args.dit_cache)} "
            f"condition_source=request",
            flush=True,
        )
        while True:
            request = server.recv_request()
            response = service.plan(request)
            server.send_plan(response)
            timing_ms = response.metadata.get("timing_ms", {})
            print(
                f"[replan {response.plan_id:04d}] request={response.request_id} "
                f"frame={response.request_tracker_frame:05d} buffer={request.buffer_remaining_frames:04d} "
                f"prompt={response.prompt!r} latency={response.planning_latency_seconds * 1000.0:.3f}ms"
                f"{_format_condition(response.metadata.get('realtime_condition'))}"
                f"{_format_audio(response.metadata.get('realtime_audio'))}"
                f" {_format_timing(timing_ms)}",
                flush=True,
            )
            if args.log_jsonl is not None:
                service.append_jsonl(args.log_jsonl, request, response)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
