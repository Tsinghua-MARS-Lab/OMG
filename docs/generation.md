# Generation

The main offline pipeline entry point is:

```bash
PYTHONPATH=src python -m omg.cli.pipeline.main
```

It supports five modes:

- `diffusion-only`: generate reference motion and optionally render it.
- `tracker-only`: track an existing reference through HoloMotion.
- `sync`: diffusion plans a chunk, tracker executes the whole chunk, then the
  next plan starts.
- `async`: tracker keeps executing a reference buffer while diffusion replans
  before the buffer runs out.
- `offline-track`: generate a reference once, then track it offline.

## Condition Sequence

Use `--condition-sequence` for chunk-level conditions:

```text
text: walk forward
text[5]: walk forward | text[3]: turn around
audio: inputs/audio/demo.wav
humanref: inputs/humanref/sample.npz
text+audio: wave arms+/path/to/audio.wav
text+humanref: imitate this+/path/to/ref.npz
```

`text[5]` repeats the same text condition for five diffusion chunks. Audio
chunks without `[N]` expand to the wav duration. When async runs with audio,
the audio timeline advances according to tracker execution time, not planner
latency.

## Diffusion Only

```bash
PYTHONPATH=src python -m omg.cli.pipeline.main \
  --mode diffusion-only \
  --diffusion-onnx models/generation/onnx/50m/last_denoiser_step.onnx \
  --seed-motion /path/to/seed_motion.npz \
  --condition-sequence "text: walk forward" \
  --num-frames 120 \
  --video \
  --output-root outputs_pipeline
```

The output directory contains generated reference motion and metadata.

## Sync Mode

```bash
PYTHONPATH=src python -m omg.cli.pipeline.main \
  --mode sync \
  --diffusion-onnx models/generation/onnx/50m/last_denoiser_step.onnx \
  --holomotion-onnx models/holomotion/motion_tracking/model.onnx \
  --seed-motion /path/to/seed_motion.npz \
  --condition-sequence "text[4]: walk forward | text[2]: turn around" \
  --num-frames 300 \
  --video \
  --output-root outputs_pipeline
```

Sync replans after each tracker-executed chunk.

## Async Mode

```bash
PYTHONPATH=src python -m omg.cli.pipeline.main \
  --mode async \
  --diffusion-onnx models/generation/onnx/50m/last_denoiser_step.onnx \
  --holomotion-onnx models/holomotion/motion_tracking/model.onnx \
  --seed-motion /path/to/seed_motion.npz \
  --condition-sequence "text: walk forward" \
  --num-frames 300 \
  --async-replan-remaining-frames 40 \
  --video \
  --output-root outputs_pipeline
```

Async mode starts replanning when the tracker reference buffer has at most
`--async-replan-remaining-frames` frames remaining. TensorRT FP16 and DiT cache
default to enabled in async mode.

## Audio

For wav-driven conditions:

```bash
--condition-sequence "audio: inputs/audio/demo.wav" --audio-type audio
```

For precomputed wav features at startup:

```bash
--condition-sequence "audio: inputs/audio/demo.wav" --audio-type feature
```

Both forms take a wav path in the condition string.

## Export ONNX

The default export path is TensorRT-compatible and uses fixed batch size 2 for
batched classifier-free guidance.

```bash
PYTHONPATH=src python -m omg.cli.generation.export_onnx \
  --exp 50m \
  --ckpt_path outputs/<run>/checkpoints/last.ckpt \
  --output models/generation/onnx/50m/last_denoiser_step.onnx \
  --batch_size 2 \
  --device cuda
```

The exporter writes a sidecar metadata file next to the ONNX model. The planner
uses that metadata to recover sequence length, feature dimension, text/audio
settings, representation, diffusion contract, and attention architecture.

New checkpoints carry an architecture contract. Legacy checkpoints do not, and
QK-normalization changes cannot be inferred from parameter names or shapes. A
legacy export must therefore declare one of `none`, `cross-only`, `self-only`,
or `self-and-cross` and instantiate the matching denoiser. Example:

```bash
PYTHONPATH=src python -m omg.cli.generation.export_onnx \
  --exp 100m_omnimodal \
  --ckpt_path /path/to/legacy.ckpt \
  --legacy-attention-contract cross-only \
  denoiser.self_attention_qk_norm=false \
  denoiser.cross_attention_qk_norm=true
```

The exporter validates training-denoiser-to-wrapper and wrapper-to-ONNX numeric
parity. It deletes the emitted graph and fails if either gate exceeds tolerance.

## TensorRT Runtime

The pipeline and realtime planner can run exported ONNX denoiser steps with
ONNX Runtime TensorRT providers. Async mode enables TensorRT FP16 and DiT cache
by default.

Common provider order:

```bash
--providers TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider
```

Realtime planner defaults:

- TensorRT FP16 enabled.
- DiT cache enabled.
- TensorRT engine cache under `tensorrt_engine_cache/realtime_planner`.

## Rendering

Common render flags:

```bash
--video
--camera-view iso
--follow-mode xy
--scene-preset studio
--video-width 1280
--video-height 720
```

`--follow-mode xy` is the default for pipeline rendering and is usually the
most useful view for walking motions.
