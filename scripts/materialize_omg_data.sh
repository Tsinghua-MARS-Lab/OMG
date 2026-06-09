#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"

"${PYTHON:-python3}" -m omg.cli.data.materialize \
  --data-config configs/generation/data/omg_data.yaml \
  --representation-config configs/generation/representation/125d.yaml \
  --paths-config configs/generation/paths/default.yaml \
  "$@"
