# Data and Model Artifacts

## Hugging Face Placeholders

The public artifacts will be released on Hugging Face:

```text
OMG-Data:              https://huggingface.co/datasets/THU-MARS/OMG-Data
Materialized OMG-Data: https://huggingface.co/datasets/THU-MARS/OMG-Data-Materialized
OMG checkpoints:       https://huggingface.co/<org>/OMG
OMG evaluator:         https://huggingface.co/<org>/OMG-Evaluator
```

## Default Local Layout

The portable default layout is:

```text
OMG/
  data/
    OMG-Data/
      data/
      meta/
      materialized/
  models/
    generation/
    evaluator/
    holomotion/
```

Hydra configs read these paths from `configs/generation/paths/default.yaml`.
Override them with environment variables:

```bash
export OMG_DATA_ROOT=/path/to/OMG-Data
export OMG_MATERIALIZED_ROOT=/path/to/OMG-Data/materialized
export OMG_MODELS_ROOT=/path/to/OMG-models
```

## Required Runtime Artifacts

For generation and tracking demos, prepare:

- an exported OMG diffusion ONNX model and metadata sidecar;
- a HoloMotion G1 tracker ONNX model;
- a seed G1 motion file containing `qpos_36`;
- `assets/stats/g1_125d_stats.json`, generated with
  `omg.cli.generation.compute_stats`.

For evaluator-based benchmark metrics, prepare:

- `models/evaluator/pretrained.ckpt`, downloaded from the OMG evaluator release.

For training, prepare either the official LeRobot v3 dataset under `OMG-Data/`
or frame-level episode kinematics caches under `materialized/`.
