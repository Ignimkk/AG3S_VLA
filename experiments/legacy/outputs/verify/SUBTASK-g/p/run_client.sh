#!/bin/bash
# SUBTASK-g client, one run (copied from SUBTASK-f p/run_client.sh: same client flags; snapshot ${AG3S_PROJECT_ROOT}-sg).
# usage: run_client.sh COND PORT OUTDIR EP SEED      run dir = OUTDIR/{COND}_ep{EP}_s{SEED}
source ${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-g/p/env.sh
C=$1; PORT=$2; OUT=$3; EP=$4; SEED=$5; NAME=${C}_ep${EP}_s${SEED}; RUN=$OUT/$NAME
port_ok $PORT || { echo "port $PORT not allowed"; exit 2; }
[ -f $OUT/$NAME.done ] && { echo "skip $NAME (done)"; exit 0; }
case $C in
  E3bD|E3bN) SAFE="--safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators left --safe-gate reasons --safe-hold-mode fixed";;
  *) echo "unknown COND $C"; exit 2;;
esac
mkdir -p $OUT
if [ -e $RUN ] || [ -e $OUT/$NAME.client.log ]; then
  TS=$(date +%Y%m%dT%H%M%S); mkdir -p $OUT/failed
  for f in $RUN $OUT/$NAME.client.log $OUT/$NAME.start $OUT/$NAME.end $OUT/$NAME.check.json; do [ -e $f ] && mv $f $OUT/failed/$(basename $f).$TS; done
fi
date -Iseconds > $OUT/$NAME.start; echo "port=$PORT" >> $OUT/$NAME.start
cd $SD/pi05_TO_hybrid && MUJOCO_GL=osmesa PYTHONPATH=$SD timeout $CLIENT_TIMEOUT_S $PY -u rby1_bringup/pi05_infer.py \
  --model rby1_randomized_pick_place_16d --remote localhost:$PORT --episode-index $EP --max-steps 600 --headless \
  --warmup-steps 1000 --policy-seed $SEED \
  --record-frames $RUN --record $RUN/third_person_front.mp4 --view front --trajectory-out $RUN/traj.npz \
  $SAFE > $OUT/$NAME.client.log 2>&1
echo "exit=$?" >> $OUT/$NAME.client.log
date -Iseconds > $OUT/$NAME.end
if $PY $P/check_run.py $RUN $C $EP $SEED > $OUT/$NAME.check.json 2>&1; then touch $OUT/$NAME.done; echo "ok $NAME"; else echo "FAILED $NAME"; exit 1; fi
