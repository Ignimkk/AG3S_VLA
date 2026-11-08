#!/usr/bin/env bash
# SL-PE closed loop, step d (dev-30, 2026-11-08 23:0x KST, G158) — mechanism checks on the gt24att-failed tasks, after pe1c_queue.sh.
#  1. pe125cls  — the user-requested check of the earlier Panda obstacle-row method alone (G75 A': TO row classes + beneath rule,
#                 no scene obstacle set); PE1 · PE2 · PE5 as in pe12345. LIBERO t3 · 4 · 7 · 9 x seeds 8 · 9 + SafeLIBERO LI t3.
#  2. pe123456  — pe12345 + --no-target-certify (PE6: no HOLD on the first no-target calls). LIBERO seeds 8 · 9 · 7 + SafeLIBERO t3.
# Waits for pe1c_queue.sh's "queue end" (a running queue script is never edited). The launcher takes the GPU lock per job.
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
until grep -q "queue end" <(grep -A100 "start pe12345 LIBERO" "$LOG"); do sleep 30; done
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --seed 7"
for spec in "pe125cls:AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst2-pts-nocarve-fh-t100-pe125cls:8 9" \
            "pe123456:AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123456:8 9 7"; do
  TAG=${spec%%:*}; rest=${spec#*:}; V=${rest%%:*}; SEEDS=${rest#*:}
  LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $V --record-constraints full"
  for s in $SEEDS; do
    run "$TAG LIBERO t3,4,7,9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 3,4,7,9 --episode-ids 0 --seed $s \
      --run-id gtto-$TAG-libero4s$s
  done
  run "$TAG SafeLIBERO t3" $PY $LP $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-$TAG-safelibero11-t3
done
log "queue d end"
