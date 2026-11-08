#!/usr/bin/env bash
# SL-GTTO11 GPU queue (lead G54, 2026-11-07): starts after gtto10_queue.sh writes "queue end". Sequential; every launcher takes
# the shared GPU lock itself. Rules fixed before results: E1_GTTO_DECISIONS_20261104_KO.md G54.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-07_F9_스케일QP_닫힘확인
mkdir -p $OUT
LOG=$OUT/queue.log
until grep -q "queue end" outputs/2026-11-07_F8_쥔물체목적지단계/queue.log 2>/dev/null; do sleep 20; done
V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-qs
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
LS="benchmark/runners/launch_libero_safety_paired.py --level 0 --task-ids 0 --policy-checkpoint pi05_libero --conversion C --modes vla_v11ab"
LP="benchmark/runners/launch_libero_paired.py --modes vla_v11ab --conversion C"
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14"
run "F9 LS human_safety (record full)" $PY $LS --task-suite-name human_safety --episode-ids 0,1,2,3 --run-id gtto-gt12qs-ls10 --ag3s-version $V --record-constraints full
run "F9 LS obstacle_avoidance" $PY $LS --task-suite-name obstacle_avoidance --episode-ids 0,1,2,3 --run-id gtto-gt12qs-ls10 --ag3s-version $V
run "F9 LS affordance" $PY $LS --task-suite-name affordance --episode-ids 0,2 --run-id gtto-gt12qs-ls10 --ag3s-version $V
for s in 7 8 9; do
  run "F9 LIBERO9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 0,1,2,3,4,6,7,8,9 --episode-ids 0 --seed $s --run-id gtto-gt12qs-libero9s$s --ag3s-version $V
done
run "F9 SafeLIBERO t1" $PY $LP $SL --task-ids 1 --episode-ids 1,4 --run-id gtto-gt12qs-safelibero11-t1 --ag3s-version $V
run "F9 SafeLIBERO t2" $PY $LP $SL --task-ids 2 --episode-ids 0,1,2,3,4 --run-id gtto-gt12qs-safelibero11-t2 --ag3s-version $V
run "F9 SafeLIBERO t3" $PY $LP $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-gt12qs-safelibero11-t3 --ag3s-version $V
# VLA-only references for the 27-episode LIBERO protocol on this machine (seed 7 exists: gtto-vla-libero9)
for s in 8 9; do
  run "VLA LIBERO9 seed $s" $PY benchmark/runners/launch_libero_paired.py --modes vla_only --conversion C --task-suite-name libero_spatial --task-ids 0,1,2,3,4,6,7,8,9 --episode-ids 0 --seed $s --run-id gtto-vla-libero9s$s
done
log "queue end"
