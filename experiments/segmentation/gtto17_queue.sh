#!/usr/bin/env bash
# LIBERO parity (lead G70, 2026-11-07): after the F8 observe queue (gtto15) ends, F11 observe-only (`-hdp-loose-obs`) on the
# CALIBRATION seeds 10 · 11 (LIBERO 9 each, --record-constraints full). Gate (fixed before results): no call that F11 would
# change or hold on a VLA-success episode. Gate open -> gtto16_queue.sh runs F11 closed loop on the evaluation protocol
# (LIBERO 27, then SafeLIBERO 11 · LIBERO-Safety 10 only if 27/27). Gate closed -> stop and report the remaining kinds.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-07_LIBERO_동률_F11
mkdir -p $OUT
LOG=$OUT/queue.log
until grep -q "queue end" outputs/2026-11-07_LIBERO_과민제약_진단/queue.log 2>/dev/null; do sleep 20; done
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
VO=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose-obs
LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $VO --record-constraints full"
for s in 10 11; do
  run "F11obs LIBERO9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 0,1,2,3,4,6,7,8,9 --episode-ids 0 --seed $s --run-id gtto-gt14obs-libero9s$s
done
if $PY -I experiments/segmentation/gtto_obs_fp.py --tag gt14obs --out $OUT/관찰_F11_seed10_11 --gate >> "$LOG" 2>&1; then
  log "gate open: F11 closed loop"
  log "queue end (observe part)"
  V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose TAG=gt14loose OUT=$OUT bash experiments/segmentation/gtto16_queue.sh
else
  log "gate closed: F11 still changes / holds the VLA path on calibration seeds -- not evaluated"
  log "queue end"
fi
