#!/usr/bin/env bash
# SL-PE closed loop, step e (dev-30, 2026-11-09 00:3x KST, G159): pe1234567 = pe12345 + PE6 (--no-target-certify) + PE7
# (--close-seed-attention-ratio 5) on the gt24att-failed tasks; SafeLIBERO t3 first (the e3 wine-bottle collision of pe12345).
# Replaces pe1d's pe123456 part (pe1d stopped after pe125cls). Waits for the pe125cls SafeLIBERO launcher (pid in $1) to end.
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
while kill -0 "${1:-0}" 2>/dev/null; do sleep 15; done
log "pe125cls SafeLIBERO t3 launcher (pid ${1:-?}) ended (pe1d bash stopped at 00:3x)"
V=AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe1234567; TAG=pe1234567
LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $V --record-constraints full"
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --seed 7"
run "$TAG SafeLIBERO t3" $PY $LP $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-$TAG-safelibero11-t3
for s in 8 9 7; do
  run "$TAG LIBERO t3,4,7,9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 3,4,7,9 --episode-ids 0 --seed $s \
    --run-id gtto-$TAG-libero4s$s
done
log "queue e end"
