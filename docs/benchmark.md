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

Evaluator-based distribution and retrieval metrics require a pretrained OMG
evaluator checkpoint. It will be released at:

```text
https://huggingface.co/<org>/OMG-Evaluator
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

Prepare benchmark samples with:

```bash
PYTHONPATH=src python -m omg.cli.evaluation.prepare_samples \
  --output_dir outputs_benchmark/samples
```

Use the generated manifest as input to artifact benchmark runners.
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
