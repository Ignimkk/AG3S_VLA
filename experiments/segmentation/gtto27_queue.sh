#!/usr/bin/env bash
# lead G95 (2026-11-08 07:5x): n = 11 per arm on SafeLIBERO -> the B5 vs B3 trade-off (B5 9 success / 2 collisions, B3 10 / 6) is
# checked on a second SafeLIBERO seed (8; VLA seed 8 exists: gtto-vla-safelibero11s8). After the non-GT run (gtto26 step 3).
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
until grep -q "queue end" outputs/2026-11-08_GT없는_AG3S_B5_F14h/queue.log 2>/dev/null; do sleep 20; done
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-08_SafeLIBERO_seed8_B3_B5
mkdir -p $OUT
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --seed 8"
for pair in "gt23t100:AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh-t100" "gt21pts:AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve"; do
  TAG=${pair%%:*}; V=${pair#*:}
  LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $V --record-constraints full"
  run "$TAG SafeLIBERO s8 t1" $PY $LP $SL --task-ids 1 --episode-ids 1,4 --run-id gtto-$TAG-safelibero11s8-t1
  run "$TAG SafeLIBERO s8 t2" $PY $LP $SL --task-ids 2 --episode-ids 0,1,2,3,4 --run-id gtto-$TAG-safelibero11s8-t2
  run "$TAG SafeLIBERO s8 t3" $PY $LP $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-$TAG-safelibero11s8-t3
done
log "queue end"
