#!/usr/bin/env bash
# Sequentially run every external segmenter arm on the SafeLIBERO dev rollouts (shared GPU lock per arm), then score on CPU.
# usage: run_safelibero_baselines_all.sh ROOT DATASET_DIR [DATASET_DIR...]
set -u
ROOT=$1; shift
DATASETS=("$@")
cd "$(dirname "$0")/../.."
source Baseline/paper_baselines/setup/paper_baselines_env.sh >/dev/null 2>&1
mkdir -p "$ROOT"
run() {  # name provider mode [yoloe_conf]
  local name=$1 prov=$2 mode=$3 conf=${4:-0}
  [ -d "$ROOT/$name" ] && { echo "skip $name (exists)"; return; }
  python3 scripts/setup/gpu_lock.py shared --est-gib 14 --tag "sl-base-$name" -- \
    .venv-baselines/bin/python experiments/segmentation/run_safelibero_baseline.py --provider "$prov" --mode "$mode" \
    --dataset "${DATASETS[@]}" --out "$ROOT/$name" --yoloe-conf "$conf" 2>&1 | grep -E "^(grounded|yoloe)|ERRORS|Traceback|Error" | tail -3
  echo "done $name rc=${PIPESTATUS[0]} $(date -u +%H:%M:%S)"
}
run gsam2_frame__phrase    grounded_sam2_frame phrase
run gsam2_frame__instruction grounded_sam2_frame instruction
run gsam2_track__phrase    grounded_sam2_track phrase
echo ALL-DONE
