#!/usr/bin/env bash
# dev-e5/dev-38 SL-QE (2026-11-08 19:3x KST, user: "이어서 돌리세요"): the rest of qe1_queue.sh after the session ended at
# B5 (gt23t100) task 2 (2/10 episodes; that run is renamed INVALID-interrupted_*). Same protocol: E1a SafeLIBERO Spatial LI,
# task x test init 5-14, seed 7, main_aegis budget, --record-constraints full. One job per task (peers can take the lock between).
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
log "qe1b: resume after the session stop (gt23t100 task 2 interrupted at 2/10 -> rerun)"
SL="--task-suite-name safelibero_spatial --safety-level I --episode-ids 5,6,7,8,9,10,11,12,13,14 --seed 7 --data-split test_SafeLIBERO_Spatial_ep5-49_user_2026-10-14 --safelibero-step-budget main_aegis"
jobs=("gt23t100 AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh-t100 2"
      "gt23t100 AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh-t100 3"
      "gt24att AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100 0"
      "gt24att AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100 1"
      "gt24att AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100 2"
      "gt24att AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100 3")
for j in "${jobs[@]}"; do
  read -r TAG V t <<< "$j"
  LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $V --record-constraints full"
  run "$TAG E1a spatial LI task $t" $PY $LP $SL --task-ids $t --run-id qe-$TAG-e1a-spatial-LI-t$t
done
log "queue end"
