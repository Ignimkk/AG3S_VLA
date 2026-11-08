#!/usr/bin/env bash
# dev-e5 SL-QE (2026-11-08 18:0x KST): put B5 (GT, gt23t100) and B5 without GT (gt24att) on the E1a SafeLIBERO protocol
# (paper/experiments/configs/E1_SAFELIBERO_LIBEROSAFETY_BASELINE_PROTOCOL_20261104_KO.md, sample A: Spatial Level I, task 0-3 x
# test init 5-14 = 40 episodes, seed 7, budget main_aegis = 300 policy + 20 wait) so they sit next to the E1a arms (π0.5 only,
# AEGIS, multi-link CBF, KNOWS; outputs/e1a/agg_2030/e1a_table.md) under the official TSR / CAR / SSR / ETS definitions.
# Evaluation only: both versions are fixed (B5 = user ruling G101, gt24att = G92); nothing is tuned on these episodes.
# One job per task (10 episodes) so peers can take the GPU lock between jobs (the launcher takes it per job; no outer flock).
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-08_E1a_SafeLIBERO_B5
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
SL="--task-suite-name safelibero_spatial --safety-level I --episode-ids 5,6,7,8,9,10,11,12,13,14 --seed 7 --data-split test_SafeLIBERO_Spatial_ep5-49_user_2026-10-14 --safelibero-step-budget main_aegis"
for pair in "gt23t100 AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh-t100" \
            "gt24att AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100"; do
  TAG=${pair%% *}; V=${pair#* }
  LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $V --record-constraints full"
  for t in 0 1 2 3; do
    run "$TAG E1a spatial LI task $t" $PY $LP $SL --task-ids $t --run-id qe-$TAG-e1a-spatial-LI-t$t
  done
done
log "queue end"
