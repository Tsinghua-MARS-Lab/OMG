# Development

## Package Layout

```text
src/omg/
  benchmarks/    Benchmark metrics, runners, reports, and evaluator inference.
  callbacks/      Lightning callbacks.
  cli/            User-facing command-line entry points.
  core/           Small shared logging/path/tensor utilities.
  data/           Unified OMG-Data and materialized dataset loaders.
  generation/     Training data, denoisers, diffusion, losses, export.
  motion/         G1 motion representation utilities.
  pipeline/       Offline diffusion and tracker orchestration.
  realtime/       ZMQ protocol, planner service, realtime buffers.
  render/         MuJoCo rendering.
  robots/         G1 kinematics and constants.
  runtime/        Runtime helpers such as ONNX provider setup.
  tracking/       HoloMotion tracker integration.
```

External baseline reproduction code is kept on the `repro/baselines` branch.
The release `main` branch keeps only the artifact benchmark interface needed to
evaluate generated `qpos_36` outputs.

## CLI Entry Points

Generation:

```text
omg.cli.generation.train
omg.cli.generation.generate
omg.cli.generation.export_onnx
omg.cli.generation.benchmark
omg.cli.generation.physical_benchmark
```

Pipeline:

```text
omg.cli.pipeline.main
```

Tracking:

```text
omg.cli.tracking.holomotion
omg.cli.tracking.export_holomotion_clip
```

Realtime:

```text
omg.cli.realtime.planner_server
omg.cli.realtime.holomotion_real_bridge
omg.cli.realtime.holomotion_dry_run
omg.cli.realtime.policy_node_smoke_driver
```

## Tests

Run the test suite with:

```bash
PYTHONPATH=src pytest
```

Lightweight syntax check:

```bash
python3 -m compileall -q src/omg tests
```

Before committing:

```bash
git diff --check
PYTHONPATH=src pytest
```
