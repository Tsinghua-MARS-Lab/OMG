# Training

Training uses PyTorch Lightning with Hydra configs under `configs/generation`.
Download OMG-Data first and place it at `data/OMG-Data`, or set
`OMG_DATA_ROOT` and `OMG_MATERIALIZED_ROOT`.

Text-conditioned runs also need the Hugging Face `t5-base` text encoder. The
default config expects a local copy at:

```text
${OMG_MODELS_ROOT}/t5-base-local
```

Use a different local path or Hugging Face model id with:

```bash
model.text_encoder.model_name=/path/to/t5-base
```

## Minimal Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=src python -m omg.cli.generation.train \
  exp=50m \
  data=omg_data_materialized \
  trainer=4gpu \
  logger=wandb \
  exp_name=50m_release_train
```

The main config is:

```text
configs/generation/train.yaml
```

Common overrides:

```bash
trainer.max_steps=200000
trainer.val_check_interval=2000
callbacks.checkpoint.every_n_train_steps=2000
data.loader_opts.train.batch_size=64
```

## Model Sizes

Experiment presets:

```text
configs/generation/exp/50m.yaml
configs/generation/exp/100m.yaml
configs/generation/exp/300m.yaml
configs/generation/exp/500m.yaml
configs/generation/exp/1b.yaml
```

Each experiment selects a Transformer denoiser size and training hyperparameters.

## Resume Training

Use `ckpt_path` for a full Lightning resume:

```bash
PYTHONPATH=src python -m omg.cli.generation.train \
  exp=50m \
  data=omg_data_materialized \
  trainer=4gpu \
  logger=wandb \
  ckpt_path=outputs/50m_release_train/checkpoints/last.ckpt
```

Use `init_weights_only_ckpt` only when initializing model weights without
resuming optimizer, scheduler, dataloader, or global step state:

```bash
PYTHONPATH=src python -m omg.cli.generation.train \
  exp=50m \
  data=omg_data_materialized \
  trainer=4gpu \
  logger=wandb \
  init_weights_only_ckpt=outputs/source/checkpoints/last.ckpt
```

## W&B Logging

```bash
export WANDB_API_KEY="..."
export WANDB_MODE=online
```

Disable logging with:

```bash
logger=none
```

## Checkpoints

Find recent checkpoints:

```bash
find outputs/<exp_name>/checkpoints -maxdepth 1 -name "*.ckpt" | sort | tail -20
```

Export from a checkpoint with:

```bash
PYTHONPATH=src python -m omg.cli.generation.export_onnx \
  --exp 50m \
  --ckpt_path outputs/<exp_name>/checkpoints/last.ckpt \
  --output models/generation/onnx/50m/last_denoiser_step.onnx \
  --batch_size 2 \
  --device cuda
```

## Validation

Lightning validation runs according to `trainer.val_check_interval`. For quick
debugging, reduce both validation and checkpoint intervals:

```bash
trainer.val_check_interval=200 callbacks.checkpoint.every_n_train_steps=200
```

Use short debug runs only for code checks. Do not compare motion quality from
very short runs.
