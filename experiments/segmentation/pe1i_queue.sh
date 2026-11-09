#!/usr/bin/env bash
# SL-PE8 closed loop, step i (dev-4a, 2026-11-09 03:2x KST, G162): mechanism check of pe123458 (= pe12345 + PE8
# --scene-obstacle-held-motion) on the same gt24att-failed tasks. Waits for 'queue h end' (pe1234567b seed 9 rerun).
# Order: seed 9 and 7 first (pe12345 failed t9 there while carrying / placing), SafeLIBERO t3 (safety), then seed 8.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-08_PE_경로끝점_identity/closed_loop
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
until grep -q "queue h end" "$LOG"; do sleep 30; done
log "pe1i: pe123458 (PE8 held motion) mechanism check"
V=AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123458; TAG=pe123458
LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $V --record-constraints full"
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --seed 7"
for s in 9 7; do
  run "$TAG LIBERO t3,4,7,9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 3,4,7,9 --episode-ids 0 --seed $s \
    --run-id gtto-$TAG-libero4s$s
done
run "$TAG SafeLIBERO t3" $PY $LP $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-$TAG-safelibero11-t3   # same as pe1e
run "$TAG LIBERO t3,4,7,9 seed 8" $PY $LP --task-suite-name libero_spatial --task-ids 3,4,7,9 --episode-ids 0 --seed 8 \
  --run-id gtto-$TAG-libero4s8
log "queue i end"
