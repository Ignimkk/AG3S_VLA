#!/bin/bash
# usage: run_client.sh EP RUNK
EP=$1; K=$2; OUT=${AG3S_PROJECT_ROOT}/outputs/verify/T28/E0
cd ${AG3S_PROJECT_ROOT}/pi05_TO_hybrid && MUJOCO_GL=osmesa ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -u rby1_bringup/pi05_infer.py \
  --model rby1_randomized_pick_place_16d --remote localhost:8201 --episode-index $EP --max-steps 600 --headless \
  --record-frames $OUT/E0_ep${EP}_r${K} > $OUT/E0_ep${EP}_r${K}.client.log 2>&1
echo "exit=$?" >> $OUT/E0_ep${EP}_r${K}.client.log
