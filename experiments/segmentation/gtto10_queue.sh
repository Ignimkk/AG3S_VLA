#!/usr/bin/env bash
# SL-GTTO10 GPU queue (lead G53, 2026-11-07). Sequential; every launcher takes /tmp/ag3s-vla-gpu-validation.lock itself
# (no outer flock: an outer flock without AG3S_GPU_LOCK_HELD=1 deadlocks the launcher, 11-07 00:21 incident).
# Rules fixed before results: doc/active/experiments/E1_GTTO_DECISIONS_20261104_KO.md G53.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
LOG=outputs/2026-11-07_F8_쥔물체목적지단계/queue.log
V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp
V7=AG3S-commonto-gt2-termsoff-pass-zr-ee
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
LS="benchmark/runners/launch_libero_safety_paired.py --level 0 --task-ids 0 --policy-checkpoint pi05_libero --conversion C --modes vla_v11ab"
LP="benchmark/runners/launch_libero_paired.py --modes vla_v11ab --conversion C"
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14"
# 1. F8 LIBERO-Safety 10 (human_safety with the constraint record for the mechanism checks)
run "F8 LS human_safety (record full)" $PY $LS --task-suite-name human_safety --episode-ids 0,1,2,3 --run-id gtto-gt11hdp-ls10 --ag3s-version $V --record-constraints full
run "F8 LS obstacle_avoidance" $PY $LS --task-suite-name obstacle_avoidance --episode-ids 0,1,2,3 --run-id gtto-gt11hdp-ls10 --ag3s-version $V
run "F8 LS affordance" $PY $LS --task-suite-name affordance --episode-ids 0,2 --run-id gtto-gt11hdp-ls10 --ag3s-version $V
# 2. gate (G53): LS success without violation >= 5 and violations <= 1
if $PY experiments/segmentation/gtto_gate_ls.py --run-id gtto-gt11hdp-ls10 --min 5 >> "$LOG" 2>&1; then
  for s in 7 8 9; do
    run "F8 LIBERO9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 0,1,2,3,4,6,7,8,9 --episode-ids 0 --seed $s --run-id gtto-gt11hdp-libero9s$s --ag3s-version $V
  done
  run "F8 SafeLIBERO t1" $PY $LP $SL --task-ids 1 --episode-ids 1,4 --run-id gtto-gt11hdp-safelibero11-t1 --ag3s-version $V
  run "F8 SafeLIBERO t2" $PY $LP $SL --task-ids 2 --episode-ids 0,1,2,3,4 --run-id gtto-gt11hdp-safelibero11-t2 --ag3s-version $V
  run "F8 SafeLIBERO t3" $PY $LP $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-gt11hdp-safelibero11-t3 --ag3s-version $V
else
  log "gate closed: F8 LIBERO / SafeLIBERO not run"
fi
# 3. F7 reference for the 27-episode LIBERO protocol (seed 7 exists: gtto-gt10ee-libero9)
for s in 8 9; do
  run "F7 LIBERO9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 0,1,2,3,4,6,7,8,9 --episode-ids 0 --seed $s --run-id gtto-gt10ee-libero9s$s --ag3s-version $V7
done
log "queue end"
