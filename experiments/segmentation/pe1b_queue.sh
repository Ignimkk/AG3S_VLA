#!/usr/bin/env bash
# SL-PE closed loop, step 1b (dev-30, 2026-11-08 21:0x KST, decision log G154): pe123 + the cut's local support given to the
# T43 T 3(c) support test (v1 = pe1_queue.sh refused the bowl on the cabinet as not_supported, s8 t9 call 7).: B5 without GT + the VLA's grasp point as the
# target identity (…-t100-pe123 = gt24att + --close-seed-grounding --destination-after-held --scene-obstacle-held-exclude).
# Only the tasks gt24att failed on: LIBERO spatial t3 · t4 · t7 · t9 x seeds 8 · 9 · 7 (12 ep) + SafeLIBERO LI t3 (seed 7, 4 ep).
# Compared per episode with gt24att (same seeds / init states). The launcher takes the GPU lock per job (no outer flock).
# usage: bash experiments/segmentation/pe1_queue.sh
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
V=AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123
TAG=pe123b
OUT=outputs/2026-11-08_PE_경로끝점_identity/closed_loop
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $V --record-constraints full"
for s in 8 9 7; do
  run "$TAG LIBERO t3,4,7,9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 3,4,7,9 --episode-ids 0 --seed $s \
    --run-id gtto-$TAG-libero4s$s
done
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --seed 7"
run "$TAG SafeLIBERO t3" $PY $LP $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-$TAG-safelibero11-t3
log "queue end"
