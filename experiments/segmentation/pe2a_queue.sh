#!/usr/bin/env bash
# SL-PE x SL-QE2 (dev-4a, 2026-11-09 12:xx KST, user-approved order):
#   (1) pe123458-tro = GT-free final pe123458 + --ee-action-qp-translation-only: calibration LIBERO s10 · s11 -> evaluation
#       protocol (gtto16_queue.sh via pe1k_eval_queue.sh) -> E1a 40 (SafeLIBERO Spatial LI, test init 5-14, seed 7, main_aegis)
#   (2) pe1234578911 (= pe12345789 + SL-PE11 --scene-obstacle-target-cut) mechanism check on the gt24att-failed tasks
#       (same set as pe1j2: LIBERO t3,4,7,9 x seeds 9 · 7 · 8 + SafeLIBERO t3 dev) -- does PE11 keep the obstacles PE9 dropped.
# usage: bash experiments/segmentation/pe2a_queue.sh
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
V=AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123458-tro; TAG=pe123458tro
OUT=outputs/2026-11-09_집기놓기_장애물취급/eval_$TAG
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
log "pe2a: HEAD $(git rev-parse --short HEAD)"
V=$V TAG=$TAG bash experiments/segmentation/pe1k_eval_queue.sh
E1A="--task-suite-name safelibero_spatial --safety-level I --episode-ids 5,6,7,8,9,10,11,12,13,14 --seed 7 --data-split test_SafeLIBERO_Spatial_ep5-49_user_2026-10-14 --safelibero-step-budget main_aegis"
LPB="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --record-constraints full"
for t in 0 1 2 3; do
  run "$TAG E1a spatial LI task $t" $PY $LPB --ag3s-version $V $E1A --task-ids $t --run-id qe-$TAG-e1a-spatial-LI-t$t
done
log "pe2a part 1 end"
V=AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe1234578911; TAG=pe1234578911
LP="$LPB --ag3s-version $V"
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --seed 7"
run "$TAG SafeLIBERO t3" $PY $LP $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-$TAG-safelibero11-t3
for s in 9 7 8; do
  run "$TAG LIBERO t3,4,7,9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 3,4,7,9 --episode-ids 0 --seed $s \
    --run-id gtto-$TAG-libero4s$s
done
log "queue pe2a end"
