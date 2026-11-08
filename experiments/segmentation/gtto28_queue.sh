#!/usr/bin/env bash
# lead G97 (2026-11-08 08:3x): the user asked for the non-GT AG3S on EVERY benchmark; gtto16's LIBERO gate (>= 25/27) would skip
# the safety benches for it -> SafeLIBERO 11 + LIBERO-Safety 10 (seed 7) run here regardless, after the seed-8 queue (gtto27).
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
until grep -q "queue end" outputs/2026-11-08_SafeLIBERO_seed8_B3_B5/queue.log 2>/dev/null; do sleep 20; done
PY=.venv-ag3s/bin/python
V=AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100; TAG=gt24att
OUT=outputs/2026-11-08_GT없는_AG3S_B5_F14h
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
if grep -q "safety benches not run" $LOG; then
  LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $V --record-constraints full"
  SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --seed 7"
  run "$TAG SafeLIBERO t1 (gate overridden: user asked for every benchmark)" $PY $LP $SL --task-ids 1 --episode-ids 1,4 --run-id gtto-$TAG-safelibero11-t1
  run "$TAG SafeLIBERO t2" $PY $LP $SL --task-ids 2 --episode-ids 0,1,2,3,4 --run-id gtto-$TAG-safelibero11-t2
  run "$TAG SafeLIBERO t3" $PY $LP $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-$TAG-safelibero11-t3
  LS="benchmark/runners/launch_libero_safety_paired.py --level 0 --task-ids 0 --policy-checkpoint pi05_libero --conversion C --seed 7 --modes vla_v11ab --ag3s-version $V --record-constraints full"
  run "$TAG LS human_safety" $PY $LS --task-suite-name human_safety --episode-ids 0,1,2,3 --run-id gtto-$TAG-ls10
  run "$TAG LS obstacle_avoidance" $PY $LS --task-suite-name obstacle_avoidance --episode-ids 0,1,2,3 --run-id gtto-$TAG-ls10
  run "$TAG LS affordance" $PY $LS --task-suite-name affordance --episode-ids 0,2 --run-id gtto-$TAG-ls10
fi
log "queue end (safety benches, G97)"
