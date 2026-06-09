from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_datasets_root() -> Path:
    data_root = Path(os.environ.get("OMG_DATA_ROOT", "data/OMG-Data"))
    return Path(os.environ.get("OMG_DATASETS_ROOT", str(data_root / "datasets")))


def _command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "omg.cli.generation.benchmark",
        "--exp",
        args.exp,
        "--ckpts",
        *args.ckpts,
        "--evaluator_checkpoint",
        args.evaluator_checkpoint,
        "--split",
        args.split,
        "--num-texts",
        str(args.num_texts),
        "--batch-size",
        str(args.batch_size),
        "--num_frames",
        str(args.num_frames),
        "--output_dir",
        str(args.output_dir),
        "--device",
        args.device,
    ]
    if args.datasets:
        cmd.extend(["--datasets", *args.datasets])
    if args.cfg_scales:
        cmd.extend(["--cfg_scales", *args.cfg_scales])
    if args.enable_text_metrics:
        cmd.append("--enable_text_metrics")
        cmd.extend(["--t5_3b_model", args.t5_3b_model])
        cmd.extend(["--text_cache_dir", args.text_cache_dir])
    cmd.extend(args.overrides)
    return cmd


def _inspect_outputs(output_dir: Path) -> None:
    summary = output_dir / "summary.md"
    metrics = output_dir / "metrics.json"
    print(f"[INFO] expected summary={summary}")
    print(f"[INFO] expected metrics={metrics}")
    if not summary.exists():
        raise FileNotFoundError(f"summary.md not found: {summary}")
    if not metrics.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics}")
    summary_text = summary.read_text(encoding="utf-8")
    print("[INFO] summary.md first lines:")
    for line in summary_text.splitlines()[:12]:
        print(f"[INFO]   {line}")
    if "Real motions" not in summary_text:
        raise AssertionError("summary.md does not contain a Real motions row")
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    print(f"[INFO] metrics.json top_level_keys={sorted(payload.keys())}")
    run_keys = [key for key in payload.keys() if key not in {"real_motion_metrics", "samples_path"}]
    print(f"[INFO] metrics.json run_keys={run_keys}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build or run a small generation benchmark sanity command. Default mode is dry-run; pass --run "
            "to execute the official benchmark CLI and inspect summary.md / metrics.json."
        )
    )
    parser.add_argument("--exp", default="100m")
    parser.add_argument("--ckpts", nargs="+", required=True, help="One or more generation checkpoints.")
    parser.add_argument("--evaluator-checkpoint", dest="evaluator_checkpoint", required=True)
    parser.add_argument("--output-dir", dest="output_dir", default="outputs/sanity/generation_benchmark")
    parser.add_argument(
        "--dataset-root",
        default=str(_default_datasets_root()),
        help="Expected datasets root for logging. The benchmark Hydra config remains authoritative.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--num-texts", dest="num_texts", type=int, default=16)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=4)
    parser.add_argument("--num-frames", dest="num_frames", type=int, default=32)
    parser.add_argument("--cfg-scales", dest="cfg_scales", nargs="+", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--enable-text-metrics", action="store_true")
    parser.add_argument("--t5-3b-model", dest="t5_3b_model", default=None)
    parser.add_argument("--text-cache-dir", dest="text_cache_dir", default=None)
    parser.add_argument("--run", action="store_true", help="Actually run benchmark.py. Without this, only print the command.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra Hydra overrides for the benchmark CLI, e.g. model.text_encoder.model_name=/models/t5-base",
    )
    args = parser.parse_args()
    if args.enable_text_metrics and (args.t5_3b_model is None or args.text_cache_dir is None):
        raise ValueError("--enable-text-metrics requires --t5-3b-model and --text-cache-dir")

    root = _repo_root()
    cmd = _command(args)
    print("[INFO] Benchmark sanity command")
    print(f"[INFO] repo_root={root}")
    print(f"[INFO] expected_datasets_root={args.dataset_root}")
    print("[INFO] " + " ".join(str(part) for part in cmd))
    if not args.run:
        print("[INFO] dry_run=true; pass --run to execute on the remote machine")
        return 0

    env = dict(**__import__("os").environ)
    src = str(root / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else src + __import__("os").pathsep + env["PYTHONPATH"]
    subprocess.run(cmd, cwd=str(root), env=env, check=True)
    _inspect_outputs(Path(args.output_dir))
    print("[INFO] Generation benchmark sanity check finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
