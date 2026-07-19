# Data

OMG trains on Unitree G1 motion represented as `qpos_36` plus optional
conditioning streams. The public source release is an official LeRobotDataset
v3 dataset; OMG converts it to the 125D model representation at load or
materialization time.

## Motion Files

Maintainers may ingest `.npz` source files before producing the LeRobot release.
The most important key is:

```text
qpos_36: (T, 36) float array
```

The 36D G1 qpos layout is:

- Root position: 3
- Root quaternion in `wxyz`: 4
- G1 joint DOF positions: 29

Files may also include:

- `fps`: source frame rate.
- `caption`, `text`, or dataset metadata.
- `human_motion` or `human_joints` for human-reference conditioning.

## 125D Representation

The default release config is:

```text
configs/generation/representation/125d.yaml
```

It uses:

```text
assets/stats/g1_125d_stats.json
```

as the normalization statistics path. This file is not distributed with the
repository; generate it after downloading OMG-Data and before training.

## Default Data Location

OMG-Data is released on Hugging Face:

```text
https://huggingface.co/datasets/THU-MARS/OMG-Data
```

Place the dataset at:

```text
data/OMG-Data
```

or set:

```bash
export OMG_DATA_ROOT=/path/to/OMG-Data
export OMG_MATERIALIZED_ROOT=/path/to/OMG-Data/materialized
```

Dataset configs use these roots through `configs/generation/paths/default.yaml`.

## Dataset Configs

Dataset configs live in:

```text
configs/generation/data/
```

The public LeRobot source-data config is:

```text
configs/generation/data/omg_data_lerobot.yaml
```

It reads this official LeRobot v3 layout:

```text
OMG-Data/
  data/chunk-*/file-*.parquet
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-*/file-*.parquet
```

`observation.state` stores G1 `qpos_36`, `action` stores the next-frame target,
and text labels are represented through the standard `task_index`/tasks table.
Optional aligned features are `omg.audio.feature` and
`omg.humanref.motion`.

The unified `.npz + labels + info.yaml` config remains at
`configs/generation/data/omg_data.yaml` for source-data benchmarks and custom
datasets; it is not required for training on the public LeRobot release.

The materialized-data config is:

```text
configs/generation/data/omg_data_materialized.yaml
```

Use materialized data when training speed matters. The scalable format caches
frame-level qpos and FK results once per episode, then represents every exact
training window through prefix-sum metadata. It does not duplicate overlapping
stride-1 window tensors.

The release will include a precomputed materialized OMG-Data artifact:

```text
https://huggingface.co/datasets/THU-MARS/OMG-Data-Materialized
```

Place it under `OMG_MATERIALIZED_ROOT`, or generate it locally from source
OMG-Data.

To materialize the default OMG-Data source datasets, set the data roots and run:

```bash
export OMG_DATA_ROOT=/path/to/OMG-Data
export OMG_MATERIALIZED_ROOT=/path/to/OMG-Data/materialized

scripts/materialize_omg_data.sh --overwrite
```

Validate the published cache before computing stats or training:

```bash
PYTHONPATH=src python -m omg.cli.data.validate_episode_cache \
  "$OMG_MATERIALIZED_ROOT/omg_episode_cache_rot6d_seq60_hist10_k1"
```

The default command writes the layout expected by
`configs/generation/data/omg_data_materialized.yaml`:

```text
$OMG_MATERIALIZED_ROOT/
  omg_episode_cache_rot6d_seq60_hist10_k1/
    train/
      episodes.npz
      captions.json
      shards/shard_*/{qpos_36,body_pos_w,body_quat_w}.npy
    val/
    test/
```

Useful options:

```bash
scripts/materialize_omg_data.sh \
  --splits train val test \
  --max-frames-per-shard 262144 \
  --device cuda
```

## Text Conditions

Text labels are read by the unified source dataset adapter and passed to the T5 text encoder.
For release experiments, text-to-motion uses `text` chunks inside
`--condition-sequence`:

```text
text: walk forward
text[5]: walk forward
text+audio: wave arms+/path/to/audio.wav
```

## Audio Conditions

Audio conditions use 35D frame-level features. There are two runtime modes:

- `--audio-type audio`: slice waveform by time and extract features at request time.
- `--audio-type feature`: precompute features from the wav at startup and slice
  the feature timeline at request time.

Condition-sequence audio chunks always point to wav files:

```text
audio: inputs/audio/demo.wav
```

## Human Reference Conditions

Human-reference chunks point to `.npy` or `.npz` files containing a human motion
array:

```text
humanref: /path/to/human_reference.npz
text+humanref: imitate this+/path/to/human_reference.npz
```

Arrays may be flat `(T, D)` or joint-shaped `(T, J, 3)`, depending on the model
that was trained and exported.

## Materialize OMG-Data

Official LeRobot source data can be materialized with the standard materializer
through `omg.data.lerobot_dataset.LeRobotG1MotionDataset`:

```yaml
dataset_opts:
  train:
    omg_lerobot_train:
      _target_: omg.data.lerobot_dataset.LeRobotG1MotionDataset
      dataset_root: /path/to/OMG-Data
      repo_id: THU-MARS/OMG-Data
      split: train
      sequence_duration: 2.0
      fps: 30.0
      num_prev_states: ${representation.num_prev_states}
      canonical_frame_idx: ${representation.canonical_frame_idx}
      rotation_representation: ${representation.rotation_representation}
      train_window_policy: exhaustive
      train_window_stride: 1
      use_text: true
```

## Preparing New Data

Use the unified source dataset adapter as the source of truth for accepted fields.
For custom data, prefer building a small materialized dataset with:

- `qpos_36`
- frame rate metadata
- one clear condition label per clip
- optional audio or human-reference arrays aligned to the same timeline

Compute stats before training, and recompute them whenever the representation or
dataset distribution changes:

The episode-cache stats iterator is exact over all stride-1 windows and avoids
repeating source decoding and FK.

```bash
PYTHONPATH=src python -m omg.cli.generation.compute_stats \
  --data-config configs/generation/data/omg_data_materialized.yaml \
  --representation-config configs/generation/representation/125d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --output assets/stats/g1_125d_stats.json
```

The output path should match `stats_path` in the active representation config.
