#!/usr/bin/env bash
# nb1104 P5: offline target / obstacle segmentation on the same-machine VLA-only SafeLIBERO LI dev run (20 episodes),
# same scripts as 11-03 (BASELINE_SAFELIBERO_TARGET_OBSTACLE_SEGMENTATION_*), source run switched by AG3S_SL_SOURCE_RUN.
# Runs inside nb1104_queue.sh (inherits the GPU lock; offline extraction = shared-lock category, no timing reported).
# usage: nb1104_offline_seg.sh noun | NAME PROVIDER MODE [YOLOE_CONF]
set -u
cd "$(dirname "$0")/../.."
B=/mnt/dev/AG3S_VLA-storage/results/benchmark
export AG3S_SL_SOURCE_RUN=$B/20261104T045210_fe3abf215dc4_nb1104-none-LI-LI-none
G=$B/20261104T061000_fe3abf215dc4_nb1104-safelibero-gt-LI-none
ROOT=$G/baselines
mkdir -p "$ROOT"
if [ "$1" = noun ]; then
  source scripts/setup/env.sh >/dev/null 2>&1
  exec .venv-openpi-live/bin/python experiments/segmentation/safelibero_noun_rows.py --out "$G/noun_rows"
fi
NAME=$1 PROV=$2 MODE=$3 CONF=${4:-0}
source Baseline/paper_baselines/setup/paper_baselines_env.sh >/dev/null 2>&1
[ -d "$ROOT/$NAME" ] && { echo "skip $NAME (exists)"; exit 0; }
.venv-baselines/bin/python experiments/segmentation/run_safelibero_baseline.py --provider "$PROV" --mode "$MODE" \
  --dataset "$G/d01" "$G/d23" --out "$ROOT/$NAME" --yoloe-conf "$CONF"
