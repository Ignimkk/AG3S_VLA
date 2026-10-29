#!/bin/bash
# T43 LB client, one run (copy of T41/c/p/run_client.sh). usage: run_client.sh ARM PORT OUTDIR EP SEED [OBST_CONFIG OBST_PROFILE]
# run dir = OUTDIR/{ARM}_ep{EP}_s{SEED}; without OBST_* = no obstacle. Always: --obstacle-stop-distance 0 + recording (T42 client; no arrow, T43 LB).
#   E0: no safe flags.  V0: --safe-hold-mode fixed (T34 E3b).  V1 / V2: --safe-hold-mode measured (T41c).
# skips a run with OUTDIR/NAME.done; a partial run dir is moved to OUTDIR/failed/ first.
source ${AG3S_PROJECT_ROOT}/outputs/verify/T44/LB2/p/env.sh
C=$1; PORT=$2; OUT=$3; EP=$4; SEED=$5; OCFG=$6; OPROF=$7; NAME=${C}_ep${EP}_s${SEED}; RUN=$OUT/$NAME
port_ok $PORT || { echo "port $PORT not allowed"; exit 2; }
[ -f $OUT/$NAME.done ] && { echo "skip $NAME (done)"; exit 0; }
SIDE=$(side_of $EP)
SB="--safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators $SIDE --safe-gate reasons"
case $C in
  E0) SAFE="";;
  *) echo "unknown ARM $C"; exit 2;;
esac
OBST="--obstacle-stop-distance $OBST_STOP"
if [ -n "$OCFG" ]; then OBST="$OBST --obstacle-config $OCFG --obstacle-profile $OPROF"; fi
mkdir -p $OUT
if [ -e $RUN ] || [ -e $OUT/$NAME.client.log ]; then
  TS=$(date +%Y%m%dT%H%M%S); mkdir -p $OUT/failed
  for f in $RUN $OUT/$NAME.client.log $OUT/$NAME.start $OUT/$NAME.end $OUT/$NAME.check.json; do [ -e $f ] && mv $f $OUT/failed/$(basename $f).$TS; done
fi
date -Iseconds > $OUT/$NAME.start; echo "port=$PORT" >> $OUT/$NAME.start; echo "obstacle=$OCFG:$OPROF" >> $OUT/$NAME.start
cd $O1/pi05_TO_hybrid && MUJOCO_GL=osmesa PYTHONPATH=$O1 timeout $CLIENT_TIMEOUT_S $PY -u rby1_bringup/pi05_infer.py \
  --model rby1_randomized_pick_place_16d --remote localhost:$PORT --episode-index $EP --max-steps 600 --headless \
  --warmup-steps 1000 --policy-seed $SEED \
  --record-frames $RUN --record $RUN/third_person_front.mp4 --view front --trajectory-out $RUN/traj.npz \
  $OBST $SAFE > $OUT/$NAME.client.log 2>&1
echo "exit=$?" >> $OUT/$NAME.client.log
date -Iseconds > $OUT/$NAME.end
if $PY $P/check_run.py $RUN $C $EP $SEED $OPROF > $OUT/$NAME.check.json 2>&1; then touch $OUT/$NAME.done; echo "ok $NAME"; else echo "FAILED $NAME"; exit 1; fi
