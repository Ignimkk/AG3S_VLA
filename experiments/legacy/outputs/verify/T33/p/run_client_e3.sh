#!/bin/bash
# T33 (copied from T30/p, paths only) E3 client with third-person recording. usage: run_client_e3.sh COND PORT EP K
C=$1; PORT=$2; EP=$3; K=$4; OUT=${AG3S_PROJECT_ROOT}/outputs/verify/T33/${C}_rec; RUN=$OUT/${C}_ep${EP}_r${K}
case $C in
  E3a) GATE="--safe-gate off";;
  E3b) GATE="--safe-gate reasons --safe-hold-mode fixed";;
esac
date -Iseconds > $OUT/${C}_ep${EP}_r${K}.start
echo "port=$PORT" >> $OUT/${C}_ep${EP}_r${K}.start
cd ${AG3S_PROJECT_ROOT}/pi05_TO_hybrid && MUJOCO_GL=osmesa ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -u rby1_bringup/pi05_infer.py \
  --model rby1_randomized_pick_place_16d --remote localhost:$PORT --episode-index $EP --max-steps 600 --headless \
  --record-frames $RUN --record $RUN/third_person_front.mp4 --view front \
  --safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators left $GATE > $OUT/${C}_ep${EP}_r${K}.client.log 2>&1
echo "exit=$?" >> $OUT/${C}_ep${EP}_r${K}.client.log
date -Iseconds > $OUT/${C}_ep${EP}_r${K}.end
