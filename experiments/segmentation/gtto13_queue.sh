#!/usr/bin/env bash
# F9 seed-8 pair (information only, lead G60, 2026-11-07): after gtto14_queue.sh (HVC check, reordered 07:5x) writes "queue end", F9 runs LIBERO-Safety 10 and
# SafeLIBERO 11 at seed 8 (VLA-only seed 8 comes from gtto12). Skipped if gtto12 already ran F9 as the candidate.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-07_후보_seed8확인
LOG=$OUT/queue_f9.log
until grep -q "queue end" outputs/2026-11-07_HVC가설_기록/queue.log 2>/dev/null; do sleep 20; done
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
if grep -q "candidate AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-qs" $OUT/queue.log; then log "F9 already ran as the candidate"; log "queue end"; exit 0; fi
V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-qs; ID=gt12qs
LS="benchmark/runners/launch_libero_safety_paired.py --level 0 --task-ids 0 --policy-checkpoint pi05_libero --conversion C --seed 8 --modes vla_v11ab --ag3s-version $V"
LP="benchmark/runners/launch_libero_paired.py --conversion C --seed 8 --modes vla_v11ab --ag3s-version $V"
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14"
run "$ID LS human_safety s8" $PY $LS --task-suite-name human_safety --episode-ids 0,1,2,3 --run-id gtto-$ID-ls10s8
run "$ID LS obstacle_avoidance s8" $PY $LS --task-suite-name obstacle_avoidance --episode-ids 0,1,2,3 --run-id gtto-$ID-ls10s8
run "$ID LS affordance s8" $PY $LS --task-suite-name affordance --episode-ids 0,2 --run-id gtto-$ID-ls10s8
run "$ID SafeLIBERO t1 s8" $PY $LP $SL --task-ids 1 --episode-ids 1,4 --run-id gtto-$ID-safelibero11s8-t1
run "$ID SafeLIBERO t2 s8" $PY $LP $SL --task-ids 2 --episode-ids 0,1,2,3,4 --run-id gtto-$ID-safelibero11s8-t2
run "$ID SafeLIBERO t3 s8" $PY $LP $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-$ID-safelibero11s8-t3
log "queue end"
