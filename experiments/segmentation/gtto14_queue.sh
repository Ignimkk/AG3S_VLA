#!/usr/bin/env bash
# HVC hypothesis check (lead G61, 2026-11-07): after gtto12_queue.sh ends (reordered 07:5x: before the F9 seed-8 runs), F7 and F8 on SafeLIBERO LI t2 e0-4 (same env history as
# the scored runs) with --record-constraints full; the field at t2e4 seq 3 (where F7 intervened and F8 read 'clear') is compared.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-07_HVC가설_기록
mkdir -p $OUT
LOG=$OUT/queue.log
until grep -q "queue end" outputs/2026-11-07_후보_seed8확인/queue.log 2>/dev/null; do sleep 20; done
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
SL="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --task-ids 2 --episode-ids 0,1,2,3,4 --record-constraints full"
run "F7 SafeLIBERO t2 rec" $PY $SL --run-id gtto-gt10ee-sl2rec --ag3s-version AG3S-commonto-gt2-termsoff-pass-zr-ee
run "F8 SafeLIBERO t2 rec" $PY $SL --run-id gtto-gt11hdp-sl2rec --ag3s-version AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp
log "queue end"
