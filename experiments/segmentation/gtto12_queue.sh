#!/usr/bin/env bash
# seed-8 check of the candidate (lead G56 rule, 2026-11-07): after gtto11_queue.sh writes "queue end", the G54 gate picks the
# candidate (F9 if it opens, else F8); then VLA only and the candidate run LIBERO-Safety 10 and SafeLIBERO 11 at seed 8.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-07_후보_seed8확인
mkdir -p $OUT
LOG=$OUT/queue.log
until grep -q "queue end" outputs/2026-11-07_F9_스케일QP_닫힘확인/queue.log 2>/dev/null; do sleep 20; done
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
if $PY experiments/segmentation/gtto_gate_f9.py >> "$LOG" 2>&1; then V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-qs; TAG=gt12qs; else V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp; TAG=gt11hdp; fi
log "candidate $V"
LS="benchmark/runners/launch_libero_safety_paired.py --level 0 --task-ids 0 --policy-checkpoint pi05_libero --conversion C --seed 8"
LP="benchmark/runners/launch_libero_paired.py --conversion C --seed 8"
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14"
for arm in vla cand; do
  if [ $arm = vla ]; then M="--modes vla_only"; A=""; ID=vla; else M="--modes vla_v11ab"; A="--ag3s-version $V"; ID=$TAG; fi
  run "$ID LS human_safety s8" $PY $LS $M $A --task-suite-name human_safety --episode-ids 0,1,2,3 --run-id gtto-$ID-ls10s8
  run "$ID LS obstacle_avoidance s8" $PY $LS $M $A --task-suite-name obstacle_avoidance --episode-ids 0,1,2,3 --run-id gtto-$ID-ls10s8
  run "$ID LS affordance s8" $PY $LS $M $A --task-suite-name affordance --episode-ids 0,2 --run-id gtto-$ID-ls10s8
  run "$ID SafeLIBERO t1 s8" $PY $LP $M $A $SL --task-ids 1 --episode-ids 1,4 --run-id gtto-$ID-safelibero11s8-t1
  run "$ID SafeLIBERO t2 s8" $PY $LP $M $A $SL --task-ids 2 --episode-ids 0,1,2,3,4 --run-id gtto-$ID-safelibero11s8-t2
  run "$ID SafeLIBERO t3 s8" $PY $LP $M $A $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-$ID-safelibero11s8-t3
done
log "queue end"
