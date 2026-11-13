#!/usr/bin/env bash
# §9.37 gate (sdm-f6, 2026-11-13; plan AG3S_PLAN_SCHEDULE_20261110_KO.md §9.37, criteria fixed before results): candidate
# SL-SOTA19 ``-gh156a-sw``, same run layouts as the frozen version (pkgFrssdm). Stage order changed from the plan's 1-2-3 to
# 1-3-2 so the adoption stages (E1a 40, guard) come first and stage 2 (in-sample mechanism record, not an adoption condition)
# runs only for a version that passed. Each adoption stage is judged here mechanically (gate37_judge.py) and the queue stops
# at the first failure; if -gh156a-sw fails stage 1 or 3, the plan's fallback -gh156a runs stages 1 and 3 the same way.
# usage: bash experiments/segmentation/sota_gate37_queue.sh   [CANDS="gh156a-sw gh156a"] [MS=1]
# The launchers take the GPU lock themselves. Do not edit while it runs.
set -u -o pipefail   # the judge's exit status must survive "| tee"
cd /mnt/dev/AG3S_VLA
P="AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-rs"
CANDS=${CANDS:-"gh156a-sw gh156a"}
MS=${MS:-1}
OUT=outputs/2026-11-11_SOTA_개선/gate37
mkdir -p "$OUT"
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$OUT/queue.log"; }
J=".venv-ag3s/bin/python experiments/segmentation/gate37_judge.py"
E1A40="0:5,6,7,8,9,10,11,12,13,14 1:5,6,7,8,9,10,11,12,13,14 2:5,6,7,8,9,10,11,12,13,14 3:5,6,7,8,9,10,11,12,13,14"
log "gate37: HEAD $(git rev-parse --short HEAD) CANDS='$CANDS' MS=$MS"
for c in $CANDS; do
  V="$P-$c-tro"; TAG="g37${c//-/}sdm"
  log "candidate $c: stage 1 E1a 40 (TAG=$TAG)"
  V="$V" TAG="$TAG" SUB="$E1A40" bash experiments/segmentation/sota_sub_queue.sh
  if ! $J e1a "$TAG" | tee -a "$OUT/queue.log"; then log "candidate $c: stage 1 FAIL"; continue; fi
  log "candidate $c: stage 3 guard"
  V="$V" TAG="$TAG" OUT="outputs/2026-11-11_SOTA_개선/eval_$TAG" MIN27=21 bash experiments/segmentation/gtto16_queue.sh
  if ! $J guard "$TAG" | tee -a "$OUT/queue.log"; then log "candidate $c: stage 3 FAIL"; continue; fi
  log "candidate $c: PASS (stages 1 and 3) -> Tab. 1 version"
  if [ "$MS" = 1 ]; then
    log "candidate $c: stage 2 mechanism record (56 ep, not an adoption condition)"
    V="$V" TAG="$TAG" INITS=15,16,17,18 CELLS="spatial:I" bash experiments/segmentation/sota_ms_queue.sh
    V="$V" TAG="$TAG" INITS=15,16 CELLS="spatial:II long:I long:II" bash experiments/segmentation/sota_ms_queue.sh
    V="$V" TAG="$TAG" INITS=17,18 CELLS="spatial:II long:I long:II" bash experiments/segmentation/sota_ms_queue.sh
  fi
  break
done
log "gate37 queue end"
