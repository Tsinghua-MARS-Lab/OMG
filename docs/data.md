# Data

OMG trains on Unitree G1 motion represented as `qpos_36` plus optional
conditioning streams. The release default representation is 125D.

## Motion Files

Motion files are usually `.npz`, `.npy`, or `.pt` files containing G1 state.
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

OMG-Data will be released on Hugging Face:

```text
https://huggingface.co/datasets/<org>/OMG-Data
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

The main source-data config is:

```text
configs/generation/data/omg_data.yaml
```

Source datasets use the unified OMG-Data layout:

```text
<dataset>/
  g1/
  labels/       # optional for non-text datasets
  metadata/     # optional
  music_npy/    # optional audio features
  info.yaml
```

The `info.yaml` split entries are relative stems resolved against `g1/`,
`labels/`, and optional side-modality directories.

The materialized-data config is:

```text
configs/generation/data/omg_data_materialized.yaml
```

Use materialized data when training speed matters and the dataset has already
been converted into fixed-window shards.

The release will include a precomputed materialized OMG-Data artifact:

```text
https://huggingface.co/datasets/<org>/OMG-Data-Materialized
```

Place it under `OMG_MATERIALIZED_ROOT`, or generate it locally from source
OMG-Data.

To materialize the default OMG-Data source datasets, set the data roots and run:

```bash
export OMG_DATA_ROOT=/path/to/OMG-Data
export OMG_MATERIALIZED_ROOT=/path/to/OMG-Data/materialized

scripts/materialize_omg_data.sh --overwrite
```

The default command writes the layout expected by
`configs/generation/data/omg_data_materialized.yaml`:

```text
$OMG_MATERIALIZED_ROOT/
  filtered_original_mixed_modalities_all_rot6d_seq60_hist10_k1/
    val/
    test/
  filtered_original_mixed_modalities_all_rot6d_seq60_hist10_k1_by_dataset/
    <dataset_name>/train/
```

Useful options:

```bash
scripts/materialize_omg_data.sh \
  --splits train val test \
  --shard-size 8192 \
  --train-window-stride 1
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

## Preparing New Data

Use the unified source dataset adapter as the source of truth for accepted fields.
For custom data, prefer building a small materialized dataset with:

- `qpos_36`
- frame rate metadata
- one clear condition label per clip
- optional audio or human-reference arrays aligned to the same timeline

Compute stats before training, and recompute them whenever the representation or
dataset distribution changes:

```bash
PYTHONPATH=src python -m omg.cli.generation.compute_stats \
  --data-config configs/generation/data/omg_data.yaml \
  --representation-config configs/generation/representation/125d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --output assets/stats/g1_125d_stats.json
```

The output path should match `stats_path` in the active representation config.
