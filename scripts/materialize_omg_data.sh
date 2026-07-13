#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"

"${PYTHON:-python3}" -m omg.cli.data.materialize_episode_cache \
  --data-config configs/generation/data/omg_data_lerobot.yaml \
  --representation-config configs/generation/representation/125d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  --output-root "${OMG_MATERIALIZED_ROOT:?set OMG_MATERIALIZED_ROOT}/omg_episode_cache_rot6d_seq60_hist10_k1" \
  "$@"
