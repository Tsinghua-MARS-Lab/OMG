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
    <a href="https://huggingface.co/THU-MARS/OMG"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-OMG-FFD21E" alt="Hugging Face model"></a>
    <a href="https://huggingface.co/datasets/THU-MARS/OMG-Data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-OMG--Data-FFD21E" alt="Hugging Face dataset"></a>
  </p>
  <p>
    <img src="assets/teaser.png" alt="OMG teaser" width="900">
  </p>
</div>

## News

🚩 **Jul. 2026**: OMG won **ExWBC@RSS 2026 Oral, RoboData@RSS 2026 Spotlight**, congrats!<br>
🚩 **Jun. 2026**: We release preprint, code and data for [OMG](https://arxiv.org/abs/2606.10340).

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

OMG-Data is released as an official LeRobotDataset v3 dataset. Pretrained model
checkpoints and the benchmark evaluator are available from the official
[THU-MARS/OMG Hugging Face repository](https://huggingface.co/THU-MARS/OMG):

- [OMG-Data](https://huggingface.co/datasets/THU-MARS/OMG-Data)
- [Materialized OMG-Data]() (coming soon)
- [OMG checkpoints and evaluator](https://huggingface.co/THU-MARS/OMG/tree/main)

| Model | Training step | Checkpoint |
| --- | ---: | --- |
| OMG 50M | 90,000 | [`checkpoints/50m/sstep=090000.ckpt`](https://huggingface.co/THU-MARS/OMG/blob/main/checkpoints/50m/sstep%3D090000.ckpt) |
| OMG 100M | 100,000 | [`checkpoints/100m/sstep=100000.ckpt`](https://huggingface.co/THU-MARS/OMG/blob/main/checkpoints/100m/sstep%3D100000.ckpt) |
| OMG 300M | 55,000 | [`checkpoints/300m/sstep=055000.ckpt`](https://huggingface.co/THU-MARS/OMG/blob/main/checkpoints/300m/sstep%3D055000.ckpt) |
| OMG 500M | 50,000 | [`checkpoints/500m/sstep=050000.ckpt`](https://huggingface.co/THU-MARS/OMG/blob/main/checkpoints/500m/sstep%3D050000.ckpt) |

The pretrained benchmark evaluator is available at
[`evaluator/step_004000.pt`](https://huggingface.co/THU-MARS/OMG/blob/main/evaluator/step_004000.pt).
Release checksums are listed in
[`SHA256SUMS`](https://huggingface.co/THU-MARS/OMG/blob/main/SHA256SUMS).

Text-conditioned training and generation require the Hugging Face `t5-base`
text encoder. By default, configs load it from `${OMG_MODELS_ROOT}/t5-base-local`.
Download [t5-base](https://huggingface.co/google-t5/t5-base) for offline runs, or override
`model.text_encoder.model_name` with another local path or Hugging Face model id.

HoloMotion weights are not redistributed by OMG. Download HoloMotion models from the official [HoloMotion repository](https://github.com/HorizonRobotics/HoloMotion) or [HoloMotion Hugging Face artifacts](https://huggingface.co/HorizonRobotics/HoloMotion_models).

Recommended local layout:

```text
data/OMG-Data/
  data/
  meta/
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

hf download THU-MARS/OMG-Data \
  --type dataset \
  --revision 6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7 \
  --local-dir "$OMG_DATA_ROOT"

hf cache verify THU-MARS/OMG-Data \
  --type dataset \
  --revision 6e0dfbc1c5298bff14d4e2b1459ad678af0a38e7 \
  --local-dir "$OMG_DATA_ROOT" \
  --fail-on-missing-files
```

The data configs pin the official Hub commit and release-manifest SHA-256.
`OMG_DATA_ROOT` must therefore contain that exact LeRobot v3 snapshot; an old
or partially substituted local dataset is rejected during initialization.

## 3. Materialize Data

Materialization precomputes frame-level episode kinematics. It is recommended
for full training because it removes repeated source parsing and FK while
preserving the exact exhaustive stride-1 window set without duplicating
overlapping window tensors.


Generate the cache locally from the pinned source OMG-Data revision. A
precomputed cache is valid only when its v2 source identity matches the active
config exactly:

```bash
scripts/materialize_omg_data.sh --overwrite
```

Validate every manifest, episode index, and frame tensor before use:

```bash
PYTHONPATH=src python -m omg.cli.data.validate_episode_cache \
  "$OMG_MATERIALIZED_ROOT/omg_episode_cache_v2_rot6d_seq60_hist10_k1"
```

Episode cache v2 pins the originating LeRobot repository revision. Unpinned v1
caches must be rebuilt; they are not accepted as release training input.

Train with materialized data by using:

```bash
data=omg_data_materialized
```

For small debugging runs or custom tiny datasets, source data can be used
directly with:

```bash
data=omg_data_lerobot
```

## 4. Compute Stats

Compute normalization statistics before training. The default representation
config expects the generated stats file at:

```text
assets/stats/g1_125d_stats.json
```

The materialized reader enumerates the same exhaustive windows as the source
reader while reusing cached FK tensors.

```bash
PYTHONPATH=src python -m omg.cli.generation.compute_stats \
  --data-config configs/generation/data/omg_data_materialized.yaml \
  --representation-config configs/generation/representation/125d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --device cuda \
  --output assets/stats/g1_125d_stats.json
```

For four-GPU exact statistics, use the same command through torchrun:

```bash
torchrun --standalone --nproc-per-node=4 -m omg.cli.generation.compute_stats \
  --data-config configs/generation/data/omg_data_materialized.yaml \
  --representation-config configs/generation/representation/125d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --device cuda \
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
diffusion settings. New checkpoints record their attention architecture and the
exporter rejects a mismatched config. For checkpoints created before this
contract existed, declare the training semantics explicitly; for example, a
checkpoint trained with cross-attention QK normalization only requires:

```bash
PYTHONPATH=src python -m omg.cli.generation.export_onnx \
  --exp 100m_omnimodal \
  --ckpt_path /path/to/legacy.ckpt \
  --legacy-attention-contract cross-only \
  denoiser.self_attention_qk_norm=false \
  denoiser.cross_attention_qk_norm=true
```

Export is fail-closed: it numerically compares the training denoiser, the
TensorRT-compatible wrapper, and the emitted ONNX graph before retaining the
artifact.

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
pretrained evaluator checkpoint:

```text
https://huggingface.co/THU-MARS/OMG/blob/main/evaluator/step_004000.pt
```

Recommended local path:

```text
models/evaluator/pretrained.ckpt
```

The validated release manifests are versioned under
`assets/benchmarks/mixed_modalities_all_v2`. Regenerate them from the pinned
dataset only when intentionally defining a new benchmark release:

```bash
PYTHONPATH=src python -m omg.cli.evaluation.prepare_samples \
  --data omg_data_lerobot_omnimodal \
  --output_dir outputs/benchmark_samples/mixed_modalities_all_v2
```

Run text, audio, human-reference, or artifact benchmarks:

```bash
PYTHONPATH=src python -m omg.cli.generation.benchmark text \
  --exp 50m \
  --ckpt_path outputs/50m_release_train/checkpoints/last.ckpt \
  --evaluator_checkpoint models/evaluator/pretrained.ckpt \
  --samples_path assets/benchmarks/mixed_modalities_all_v2/text_test_1024.jsonl \
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

## Citation

If you find our code useful, please consider citing our work:
```
@article{huang2026omg,
  title={OMG: Omni-Modal Motion Generation for Generalist Humanoid Control},
  author={Huang, Siqiao and Lee, Kun-Ying and Qiao, Dongming and He, Guanqi and Wang, Zhenyu and Li, Yitang and Zhu, Shaoting and Zhao, Hang},
  journal={arXiv preprint arXiv:2606.10340},
  year={2026}
}
```
