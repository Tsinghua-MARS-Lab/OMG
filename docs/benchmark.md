# Benchmark

Benchmarks evaluate generated reference motion and tracker-executed motion.

## Main Entry

```bash
PYTHONPATH=src python -m omg.cli.generation.benchmark --help
```

The benchmark runners live under:

```text
src/omg/benchmarks/
```

They support text, audio, human-reference, artifact-based, and tracker-executed
evaluation paths.

## Evaluator Checkpoint

Evaluator-based distribution and retrieval metrics require the released OMG
evaluator checkpoint:

```text
https://huggingface.co/THU-MARS/OMG/blob/main/evaluator/step_004000.pt
```

Recommended local path:

```text
models/evaluator/pretrained.ckpt
```

Text retrieval metrics also use a [T5-3B text encoder](https://huggingface.co/google-t5/t5-3b). By default, the benchmark
loads `t5-3b`, or a local path from `OMG_T5_3B_MODEL`.

For offline runs:

```bash
hf download google-t5/t5-3b --local-dir models/t5-3b-local
export OMG_T5_3B_MODEL=models/t5-3b-local
```

## Sample Preparation

Prepare benchmark samples from the pinned public LeRobotDataset v3 release:

```bash
PYTHONPATH=src python -m omg.cli.evaluation.prepare_samples \
  --data omg_data_lerobot_omnimodal \
  --output_dir outputs/benchmark_samples/mixed_modalities_all_v2
```

The generated `omg.benchmark.sample.v2` rows contain the repository revision,
split, episode, exact window, and source identity. Runners resolve every field
against LeRobot metadata and reject stale or mismatched manifests. The same
fixed rows can therefore be shared by checkpoint and artifact evaluations
without relying on machine-local indices or private source paths.

The preparation command preserves the release cohort protocol (12 text
cohorts, 5 audio cohorts, and 11 human-reference cohorts) while resolving each
selected row to its canonical LeRobot source dataset.

Use the generated manifest as input to benchmark runners, for example with
`--samples_path .../text_test_1024.jsonl`. Dataset names passed through
`--datasets` are exact values from the `omg/dataset` episode column.
External baseline reproduction scripts live on the `repro/baselines` branch;
`main` keeps the benchmark artifact interface only.

## Physical Metrics

Run physical metrics on a motion artifact:

```bash
PYTHONPATH=src python -m omg.cli.generation.physical_benchmark \
  --motion outputs_pipeline/run/reference_motion.npz
```

The representative physical metrics are:

- `contact_sliding_speed`: average horizontal foot speed while a foot is in contact.
- `body_jerk_mean`: mean third finite difference magnitude of body positions.
- `foot_ground_error`: mean absolute signed distance from the lowest sole proxy
  point to the ground plane.

The default stats path is:

```text
assets/stats/g1_125d_stats.json
```

Generate this file with `omg.cli.generation.compute_stats` before running
benchmarks that load the motion representation.

## Tracker-Executed Evaluation

Tracker-executed metrics answer whether a generated reference can be followed by
the downstream tracker. Enable tracker execution in benchmark runners with the
tracker arguments exposed by each runner:

```text
--tracker_executed
--tracker_holomotion_onnx ...
```

The runner writes tracker-executed artifacts and metrics next to the benchmark
outputs.

## Output Files

Common benchmark outputs:

- `benchmark.json`
- `metrics.json`
- per-metric JSON files such as `physical_metrics.json`
- tracker-executed rollout artifacts when enabled

Use JSON outputs as the canonical source for tables. Markdown summaries should
be regenerated from JSON rather than edited by hand.
