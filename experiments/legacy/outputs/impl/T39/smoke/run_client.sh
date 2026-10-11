#!/bin/bash
# T39 S1 smoke client. usage: run_client.sh E0|E3b PORT TAG SEED|none [EP]   — 6 chunks, fixed warmup 1000 steps
C=$1; PORT=$2; TAG=$3; SEED=$4; EP=${5:-1807}; OUT=${AG3S_PROJECT_ROOT}/outputs/impl/T39/smoke/${C}; RUN=$OUT/$TAG
mkdir -p $RUN
case $C in
  E0)  SAFE="";;
  E3b) SAFE="--safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators left --safe-gate reasons --safe-hold-mode fixed";;
esac
SEEDARG=""; [ "$SEED" != none ] && SEEDARG="--policy-seed $SEED"
cd ${AG3S_PROJECT_ROOT}/pi05_TO_hybrid && MUJOCO_GL=osmesa ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -u rby1_bringup/pi05_infer.py \
  --model rby1_randomized_pick_place_16d --remote localhost:$PORT --episode-index $EP --max-steps 48 --headless \
  --warmup-steps 1000 $SEEDARG \
  --record-frames $RUN --trajectory-out $RUN/traj.npz $SAFE > $RUN/client.log 2>&1
echo "exit=$?" >> $RUN/client.log
