# Data

The only source dataset contract supported by OMG is the official
[THU-MARS/OMG-Data](https://huggingface.co/datasets/THU-MARS/OMG-Data)
LeRobotDataset v3 release. Training, statistics, generation sample selection,
and benchmarks all resolve samples from this contract.

## Canonical release

Place the dataset at `data/OMG-Data`, or set:

```bash
export OMG_DATA_ROOT=/path/to/OMG-Data

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

The expected layout is:

```text
OMG-Data/
  data/chunk-*/file-*.parquet
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-*/file-*.parquet
```

`observation.state` stores G1 `qpos_36`; text is represented by the standard
LeRobot task table. The aligned optional modalities are
`omg.audio.feature` and `omg.humanref.motion`, with explicit per-frame masks.
Episode metadata also carries the immutable source identity fields used by the
benchmark manifests.

Both public configs pin the same dataset revision:

```text
configs/generation/data/omg_data_lerobot.yaml
configs/generation/data/omg_data_lerobot_omnimodal.yaml
```

The first enables text only. The second enables text, audio, and human
reference conditioning. The loader verifies both the full 40-character Hub
revision and the SHA-256 of `meta/omg_manifest.json`; pointing
`OMG_DATA_ROOT` at an older or different local snapshot fails before any
training or benchmark samples are read.

## G1 representation

The source state has 36 values:

- root position: 3;
- root quaternion in `wxyz`: 4;
- G1 joint positions: 29.

The default model representation is defined by
`configs/generation/representation/125d.yaml`. It converts LeRobot windows to
the 125D model features and uses `assets/stats/g1_125d_stats.json` for
normalization.

Compute statistics directly from the canonical dataset with:

```bash
PYTHONPATH=src python -m omg.cli.generation.compute_stats \
  --data-config configs/generation/data/omg_data_lerobot.yaml \
  --representation-config configs/generation/representation/125d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --output assets/stats/g1_125d_stats.json
```

Recompute statistics whenever the pinned dataset revision, representation,
window length, or preprocessing changes.

## Optional episode cache

For large training runs, OMG can derive a frame-level episode cache from the
pinned LeRobot source. The cache is an optimization layer, not another source
dataset format: its manifest records the LeRobot identity and it must pass the
strict validator before use.

```bash
export OMG_MATERIALIZED_ROOT=/path/to/OMG-Data/materialized
scripts/materialize_omg_data.sh --overwrite
PYTHONPATH=src python -m omg.cli.data.validate_episode_cache \
  "$OMG_MATERIALIZED_ROOT/omg_episode_cache_v2_rot6d_seq60_hist10_k1"
```

`omg_data_materialized` and `omg_data_materialized_omnimodal` read only this
derived cache. Deleting the cache never removes canonical data; it can be
recreated from the pinned LeRobot release.
Legacy v1 caches have no verifiable source revision and are intentionally not
accepted; rebuild them into the v2 path.

## Benchmark sample identity

Benchmark manifests use `omg.benchmark.sample.v2`. Every row pins:

- `repo_id`, `revision`, and `split`;
- `episode_index`, `window_start`, and `num_frames`;
- source dataset, source id, segment index, and source frame interval.

The runner resolves that complete identity against LeRobot metadata and fails
if any field disagrees. Local list indices, private filesystem paths, and the
removed `.npz + labels + info.yaml` layout are not valid benchmark identities.

Prepare all three benchmark cohorts with:

```bash
PYTHONPATH=src python -m omg.cli.evaluation.prepare_samples \
  --data omg_data_lerobot_omnimodal \
  --output_dir outputs/benchmark_samples/mixed_modalities_all_v2
```

Condition eligibility follows the release protocol: text needs a non-empty
task and uses the motion-valid mask for short clips, while every requested
audio or human-reference frame must have its corresponding condition mask set.
Sampling remains balanced over the original release benchmark
cohorts; the four language-specific BEAT2 source groups form one human-reference
cohort so the protocol remains comparable to `mixed_modalities_all_v1`.

## External inference conditions

Standalone generation may accept explicit motion, audio, or human-reference
artifacts. These are inference inputs and outputs, not training datasets. OMG
does not infer sibling files from filenames or silently repair missing
conditions; callers must pass each artifact explicitly.

For new training data, publish it as a LeRobotDataset v3 revision containing
the same required state, task, modality, mask, split, and immutable identity
fields. Do not add another repository-specific source loader.
