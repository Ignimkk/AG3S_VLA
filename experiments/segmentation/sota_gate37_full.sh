#!/usr/bin/env bash
# §9.37 gate + the L12H3 second candidate (sdm-f6, 2026-11-13; plan §9.37, §9.38 보충 8, criteria fixed before results).
# 1) sota_gate37_queue.sh (-gh156a-sw, fallback -gh156a; E1a 40 -> guard -> mechanism record, judged by gate37_judge.py)
# 2) -a12h3: E1a 40 = the existing sub-a12h3sdm-e1a-spatial-LI-t1 run (same run layout) + task 0, 2, 3; judge; guard; judge.
# Meant to run detached: bash experiments/segmentation/run_detached.sh gate37 'bash experiments/segmentation/sota_gate37_full.sh'
set -u -o pipefail
cd /mnt/dev/AG3S_VLA
L=outputs/2026-11-11_SOTA_개선/gate37/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$L"; }
J=".venv-ag3s/bin/python experiments/segmentation/gate37_judge.py"
bash experiments/segmentation/sota_gate37_queue.sh
V="AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-rs-a12h3-tro"
I="5,6,7,8,9,10,11,12,13,14"
log "candidate a12h3 (user 18:4x: SSR 7 -> 9): stage 1 E1a 40 = existing sub-a12h3sdm-e1a t1 (same layout) + t0, t2, t3"
V="$V" TAG=a12h3sdm SUB="0:$I 2:$I 3:$I" bash experiments/segmentation/sota_sub_queue.sh
if $J e1a a12h3sdm | tee -a "$L"; then
  log "candidate a12h3: stage 3 guard"
  V="$V" TAG=a12h3sdm OUT=outputs/2026-11-11_SOTA_개선/eval_a12h3sdm MIN27=21 bash experiments/segmentation/gtto16_queue.sh
  $J guard a12h3sdm | tee -a "$L" || log "candidate a12h3: stage 3 FAIL"
else
  log "candidate a12h3: stage 1 FAIL"
fi
log "gate37 full end"
