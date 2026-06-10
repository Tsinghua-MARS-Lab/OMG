<div align="center">
  <p>
    <img src="assets/title.svg" alt="OMG: Omni-Modal Motion Generation for Generalist Humanoid Control" width="900">
  </p>
  <p>
    Official repository for <strong>OMG: Omni-Modal Motion Generation for Generalist Humanoid Control</strong>.
  </p>
  <p>
    <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="arXiv"></a>
    <a href="https://tsinghua-mars-lab.github.io/OMG/"><img src="https://img.shields.io/badge/Website-Page-Green" alt="Website"></a>
  </p>
  <p>
    <img src="assets/teaser.png" alt="OMG teaser" width="900">
  </p>
</div>

## Pipeline

The usual end-to-end workflow is:

1. Install the environment.
2. Download OMG-Data, model checkpoints, and HoloMotion artifacts.
3. Materialize OMG-Data for faster training.
4. Compute normalization stats.
5. Train a diffusion model.
6. Export ONNX for TensorRT/CUDA inference.
7. Run generation or full pipeline modes.
8. Run benchmarks.
9. Deploy to a G1 robot when needed.

## 1. Install

```bash
cd /path/to/OMG
make venv
source .venv/bin/activate
make install

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
```

For China mainland networks:

```bash
make install-cn
```

See [Installation](docs/installation.md) for manual `uv` commands and optional
task-specific extras.

## 2. Download Data and Artifacts

OMG-Data and pretrained OMG checkpoints will be released on Hugging Face:

- [OMG-Data]() (coming soon)
- [Materialized OMG-Data]() (coming soon)
- [OMG checkpoints]() (coming soon)
- [OMG evaluator]() (coming soon)

Text-conditioned training and generation require the Hugging Face `t5-base`
text encoder. By default, configs load it from `${OMG_MODELS_ROOT}/t5-base-local`.
Download [t5-base](https://huggingface.co/google-t5/t5-base) for offline runs, or override
`model.text_encoder.model_name` with another local path or Hugging Face model id.

HoloMotion weights are not redistributed by OMG. Download HoloMotion models from the official [HoloMotion repository](https://github.com/HorizonRobotics/HoloMotion) or [HoloMotion Hugging Face artifacts](https://huggingface.co/HorizonRobotics/HoloMotion_models).

Recommended local layout:

```text
data/OMG-Data/
  omg_data/
  materialized/
models/
  generation/
  evaluator/
  t5-base-local/
  holomotion/
    motion_tracking/model.onnx
    velocity_tracking/model.onnx
```

Set explicit roots when using external storage:

```bash
export OMG_DATA_ROOT=/path/to/OMG-Data
export OMG_MATERIALIZED_ROOT=/path/to/OMG-Data/materialized
export OMG_MODELS_ROOT=/path/to/OMG-models
```

## 3. Materialize Data

Materialization precomputes fixed-window training shards. It is recommended for
full training because it removes repeated source parsing and feature assembly
from the training loop.


You can download precomputed [materialized OMG-Data](https://huggingface.co/datasets/<org>/OMG-Data-Materialized) into `OMG_MATERIALIZED_ROOT`, or generate the
same layout locally from source OMG-Data:

```bash
scripts/materialize_omg_data.sh --overwrite
```

Train with materialized data by using:

```bash
data=omg_data_materialized
```

For small debugging runs or custom tiny datasets, source data can be used
directly with:

```bash
data=omg_data
```

## 4. Compute Stats

Compute normalization statistics before training. The default representation
config expects the generated stats file at:

```text
assets/stats/g1_125d_stats.json
```

```bash
PYTHONPATH=src python -m omg.cli.generation.compute_stats \
  --data-config configs/generation/data/omg_data.yaml \
  --representation-config configs/generation/representation/125d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --output assets/stats/g1_125d_stats.json
```

Recompute this file whenever the training data, representation, sequence
length, or preprocessing changes.

## 5. Train

Example 50M training run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=src python -m omg.cli.generation.train \
  exp=50m \
  data=omg_data_materialized \
  trainer=4gpu \
  logger=wandb \
  exp_name=50m_release_train
```

Model-size configs are available under `configs/generation/exp/`:

```text
50m.yaml  100m.yaml  300m.yaml  500m.yaml  1b.yaml
```

See [Training](docs/training.md) for resume, initialization, and config details.

## 6. Export ONNX

Export a TensorRT-compatible denoiser step from a checkpoint:

```bash
PYTHONPATH=src python -m omg.cli.generation.export_onnx \
  --exp 50m \
  --ckpt_path outputs/50m_release_train/checkpoints/last.ckpt \
  --output models/generation/onnx/50m/last_denoiser_step.onnx \
  --batch_size 2 \
  --device cuda
```

The exporter writes a metadata sidecar next to the ONNX file. Runtime planners
use it to recover sequence length, condition dimensions, representation, and
diffusion settings.

## 7. Generate and Track

Offline async generation with HoloMotion tracking:

```bash
PYTHONPATH=src python -m omg.cli.pipeline.main \
  --mode async \
  --diffusion-onnx models/generation/onnx/50m/last_denoiser_step.onnx \
  --holomotion-onnx models/holomotion/motion_tracking/model.onnx \
  --seed-motion /path/to/seed_motion.npz \
  --condition-sequence "text: walk forward" \
  --num-frames 300 \
  --video \
  --output-root outputs_pipeline
```

Supported pipeline modes:

- `diffusion-only`
- `tracker-only`
- `sync`
- `async`
- `offline-track`

See [Generation](docs/generation.md) and [Tracking](docs/tracking.md).

## 8. Benchmark

Benchmarks that report evaluator-based distribution and retrieval metrics use a
pretrained evaluator checkpoint. It will be released at:

```text
https://huggingface.co/<org>/OMG-Evaluator
```

Recommended local path:

```text
models/evaluator/pretrained.ckpt
```

Prepare fixed benchmark sample manifests:

```bash
PYTHONPATH=src python -m omg.cli.evaluation.prepare_samples \
  --data omg_data \
  --exp 50m \
  --output_dir outputs/benchmark_samples
```

Run text, audio, human-reference, or artifact benchmarks:

```bash
PYTHONPATH=src python -m omg.cli.generation.benchmark text \
  --exp 50m \
  --ckpt_path outputs/50m_release_train/checkpoints/last.ckpt \
  --evaluator_checkpoint models/evaluator/pretrained.ckpt \
  --output_dir outputs/benchmarks/50m_text
```

See [Benchmark](docs/benchmark.md) for modality-specific commands and
tracker-executed evaluation.

## 9. Deploy

Realtime deployment uses:

- HoloMotion deployment on the G1 Orin.
- OMG realtime planner server on a GPU workstation.
- OMG real bridge on the G1 Orin.

For real-robot deployment, prefer the HoloMotion velocity-tracking model:

```text
models/holomotion/velocity_tracking/model.onnx
```

See [Realtime G1 Deployment](docs/realtime_g1.md) for the full launch sequence.

## Documentation

- [Installation](docs/installation.md)
- [Artifacts](docs/artifacts.md)
- [Data](docs/data.md)
- [Training](docs/training.md)
- [Generation](docs/generation.md)
- [Tracking](docs/tracking.md)
- [Realtime G1 Deployment](docs/realtime_g1.md)
- [Benchmark](docs/benchmark.md)
- [Configuration](docs/configuration.md)
- [Development](docs/development.md)

## License

This project is released under the [MIT License](LICENSE).
