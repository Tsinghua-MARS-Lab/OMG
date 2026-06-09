# Tracking

OMG integrates HoloMotion as the downstream G1 motion tracker. Tracking
can be used directly with a reference clip or as part of sync/async generation.

## Tracker Only

```bash
PYTHONPATH=src python -m omg.cli.tracking.holomotion \
  --reference /path/to/reference_motion.npz \
  --holomotion-onnx models/holomotion/motion_tracking/model.onnx \
  --target-fps 50 \
  --video \
  --video-path outputs_tracking/reference_tracker.mp4 \
  --output outputs_tracking/reference_tracker.npz
```

The input reference must contain `qpos_36`. If the reference file does not carry
FPS metadata, pass `--reference-fps`.

## Pipeline Tracker Modes

The pipeline command also exposes tracker modes:

```bash
PYTHONPATH=src python -m omg.cli.pipeline.main \
  --mode tracker-only \
  --seed-motion /path/to/reference_motion.npz \
  --holomotion-onnx models/holomotion/motion_tracking/model.onnx \
  --num-frames 300 \
  --video
```

Use `sync`, `async`, or `offline-track` when the reference should come from the
diffusion planner.

## Export Deployment Clips

To convert a generated reference into HoloMotion deployment `motion_data` format:

```bash
PYTHONPATH=src python -m omg.cli.tracking.export_holomotion_clip \
  --reference outputs_pipeline/run/reference_motion.npz \
  --target-fps 50 \
  --output /home/unitree/holomotion/deployment/unitree_g1_ros2_29dof/src/motion_data/01_reference.npz
```

Restart the HoloMotion deployment process after changing deployment motion clips.

## Providers

Tracker provider default:

```text
TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider
```

Use CUDA-only when TensorRT is not available:

```bash
--providers CUDAExecutionProvider,CPUExecutionProvider
```

## Outputs

Tracker outputs include:

- tracker-executed `qpos_36`
- reference metadata
- optional video
- rollout timing and quality metadata when available

Use tracker-executed output to evaluate how much of a generated reference is
physically trackable by the downstream policy.
