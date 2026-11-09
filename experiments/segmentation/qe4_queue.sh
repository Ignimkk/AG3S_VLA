#!/usr/bin/env bash
# dev-38/dev-e5 (2026-11-09 05:xx KST, G112): after dev-4a's final evaluation queue ('queue k end'), run
#   (1) dev-4a's final SL-PE version on E1a SafeLIBERO Spatial LI (task 0-3 x test init 5-14, seed 7, main_aegis) -- SOTA table
#   (2) the SL-QE2 evaluation of B5-tro (passed calibration, G112) on the evaluation protocol of B5 (same as gtto16_queue.sh:
#       LIBERO 9 x seeds 7 · 8 · 9 = 27 -> SafeLIBERO 11 (seed 7, dev split, client default budget) -> LIBERO-Safety L0 10)
#       and then E1a 40 (main_aegis), so B5-tro pairs with every B5 number we have.
# usage: PE_V=<dev-4a final AG3S_VERSIONS key> PE_TAG=<short tag> bash experiments/segmentation/qe4_queue.sh
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
: "${PE_V:?}" "${PE_TAG:?}"
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-09_QE2_평가_E1a
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
log "qe4: waiting for dev-4a 'queue k end'"
until grep -qs "queue k end" outputs/2026-11-09_집기놓기_장애물취급/eval_*/queue.log; do sleep 30; done
log "qe4: dev-4a evaluation finished -> start"
E1A="--task-suite-name safelibero_spatial --safety-level I --episode-ids 5,6,7,8,9,10,11,12,13,14 --seed 7 --data-split test_SafeLIBERO_Spatial_ep5-49_user_2026-10-14 --safelibero-step-budget main_aegis"
LPB="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --record-constraints full"
for t in 0 1 2 3; do
  run "$PE_TAG E1a spatial LI task $t" $PY $LPB --ag3s-version $PE_V $E1A --task-ids $t --run-id qe-$PE_TAG-e1a-spatial-LI-t$t
done
V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh-t100-tro; TAG=gt23tro
for s in 7 8 9; do
  run "$TAG LIBERO9 seed $s" $PY $LPB --ag3s-version $V --task-suite-name libero_spatial --task-ids 0,1,2,3,4,6,7,8,9 --episode-ids 0 --seed $s --run-id gtto-$TAG-libero9s$s
done
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --seed 7"
run "$TAG SafeLIBERO t1" $PY $LPB --ag3s-version $V $SL --task-ids 1 --episode-ids 1,4 --run-id gtto-$TAG-safelibero11-t1
run "$TAG SafeLIBERO t2" $PY $LPB --ag3s-version $V $SL --task-ids 2 --episode-ids 0,1,2,3,4 --run-id gtto-$TAG-safelibero11-t2
run "$TAG SafeLIBERO t3" $PY $LPB --ag3s-version $V $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-$TAG-safelibero11-t3
LS="benchmark/runners/launch_libero_safety_paired.py --level 0 --task-ids 0 --policy-checkpoint pi05_libero --conversion C --seed 7 --modes vla_v11ab --ag3s-version $V --record-constraints full"
run "$TAG LS human_safety" $PY $LS --task-suite-name human_safety --episode-ids 0,1,2,3 --run-id gtto-$TAG-ls10
run "$TAG LS obstacle_avoidance" $PY $LS --task-suite-name obstacle_avoidance --episode-ids 0,1,2,3 --run-id gtto-$TAG-ls10
run "$TAG LS affordance" $PY $LS --task-suite-name affordance --episode-ids 0,2 --run-id gtto-$TAG-ls10
for t in 0 1 2 3; do
  run "$TAG E1a spatial LI task $t" $PY $LPB --ag3s-version $V $E1A --task-ids $t --run-id qe-$TAG-e1a-spatial-LI-t$t
done
log "queue end (qe4)"
