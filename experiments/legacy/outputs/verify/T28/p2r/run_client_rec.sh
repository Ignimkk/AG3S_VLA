#!/bin/bash
# T28 Phase 2R client with third-person recording. usage: run_client_rec.sh E0|E1|E2 EP K
C=$1; EP=$2; K=$3; OUT=${AG3S_PROJECT_ROOT}/outputs/verify/T28/${C}_rec; RUN=$OUT/${C}_ep${EP}_r${K}
case $C in
  E0) PORT=8201; SAFE="";;
  E1|E2) PORT=$([ $C = E1 ] && echo 8202 || echo 8203); SAFE="--safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators left --safe-gate off";;
esac
date -Iseconds > $OUT/${C}_ep${EP}_r${K}.start
cd ${AG3S_PROJECT_ROOT}/pi05_TO_hybrid && MUJOCO_GL=osmesa ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -u rby1_bringup/pi05_infer.py \
  --model rby1_randomized_pick_place_16d --remote localhost:$PORT --episode-index $EP --max-steps 600 --headless \
  --record-frames $RUN --record $RUN/third_person_front.mp4 --view front $SAFE > $OUT/${C}_ep${EP}_r${K}.client.log 2>&1
echo "exit=$?" >> $OUT/${C}_ep${EP}_r${K}.client.log
date -Iseconds > $OUT/${C}_ep${EP}_r${K}.end
