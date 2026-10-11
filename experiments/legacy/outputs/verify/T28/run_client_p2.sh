#!/bin/bash
# usage: run_client_p2.sh E1|E2 EP K
C=$1; EP=$2; K=$3; OUT=${AG3S_PROJECT_ROOT}/outputs/verify/T28/$C
case $C in E1) PORT=8202;; E2) PORT=8203;; esac
date -Iseconds > $OUT/${C}_ep${EP}_r${K}.start
cd ${AG3S_PROJECT_ROOT}/pi05_TO_hybrid && MUJOCO_GL=osmesa ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -u rby1_bringup/pi05_infer.py \
  --model rby1_randomized_pick_place_16d --remote localhost:$PORT --episode-index $EP --max-steps 600 --headless \
  --record-frames $OUT/${C}_ep${EP}_r${K} \
  --safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators left --safe-gate off > $OUT/${C}_ep${EP}_r${K}.client.log 2>&1
echo "exit=$?" >> $OUT/${C}_ep${EP}_r${K}.client.log
