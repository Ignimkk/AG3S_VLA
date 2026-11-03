#!/usr/bin/env bash
# score every finished arm of a baselines root that has no scores.jsonl yet (CPU, one at a time)
set -u
ROOT=$1; shift
cd "$(dirname "$0")/../.."
source scripts/setup/env.sh >/dev/null 2>&1
for d in "$ROOT"/*/; do
  [ -f "$d/summary_input.json" ] && [ ! -f "$d/scores.jsonl" ] || continue
  echo "scoring $(basename $d)"
  PYTHONPATH=$PWD .venv-ag3s/bin/python experiments/segmentation/score_safelibero_baseline.py --run "$d" --dataset "$@" | tail -1
done
