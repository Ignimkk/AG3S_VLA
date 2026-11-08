#!/usr/bin/env bash
# LIBERO parity calibration (lead G67, 2026-11-07 19:4x; user ruling 19:1x "a LIBERO episode the VLA alone succeeds must succeed
# with the module"): F8 in observe-only (GT12 `-hdp-obs`: the env executes the VLA chunk bit for bit, the server records what F8
# would have sent) on CALIBRATION seeds 10 · 11 (LIBERO 9 each; the evaluation seeds 7 · 8 · 9 are not used for tuning), then
# SafeLIBERO LI 11 at seed 10 (the VLA collides there: the true positives the loosened rule must keep). --record-constraints full.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-07_LIBERO_과민제약_진단
mkdir -p $OUT
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-obs
LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $V --record-constraints full"
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14"
for s in 10 11; do
  run "F8obs LIBERO9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 0,1,2,3,4,6,7,8,9 --episode-ids 0 --seed $s --run-id gtto-gt12obs-libero9s$s
done
run "F8obs SafeLIBERO t1 s10" $PY $LP $SL --seed 10 --task-ids 1 --episode-ids 1,4 --run-id gtto-gt12obs-safelibero11s10-t1
run "F8obs SafeLIBERO t2 s10" $PY $LP $SL --seed 10 --task-ids 2 --episode-ids 0,1,2,3,4 --run-id gtto-gt12obs-safelibero11s10-t2
run "F8obs SafeLIBERO t3 s10" $PY $LP $SL --seed 10 --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-gt12obs-safelibero11s10-t3
log "queue end"
