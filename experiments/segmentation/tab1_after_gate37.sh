#!/usr/bin/env bash
# (sdm-f6, 2026-11-14) Paper Tab. 1 SDM 1/10, AG3S + GT oracle arms, started as soon as the §9.37 gate (gate37b) ends.
# Version rule (plan §9.37 / §9.38 보충 8, fixed before the results): among the candidates that passed stages 1 (E1a 40) and 3
# (guard) -- -gh156a passed both on 11-14 00:00 -- take the higher E1a 40 SSR, then CAR; none -> the frozen version.
# The AG3S arm runs its FWDATT twin (<v>-fwd-tro, user decision 11-13 22:0x, no separate gate); the GT oracle arm is <frozen>-gto.
# Before the queue: main (/mnt/dev/AG3S_VLA) is fast-forwarded to wip/sota-hf (FWDATT code + versions); a launcher dry-run must
# show --attention-inpass in the server argv. Run detached:
#   bash experiments/segmentation/run_detached.sh tab1ag3s 'bash /mnt/dev/AG3S_VLA-wt-hf/experiments/segmentation/tab1_after_gate37.sh'
set -u -o pipefail
cd /mnt/dev/AG3S_VLA
L=outputs/2026-11-13_paper_tab1_1-10/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$L"; }
W=outputs/detached/gate37b.pid
while [ -f "$W" ] && kill -0 "$(cat "$W")" 2>/dev/null; do sleep 30; done
J=".venv-ag3s/bin/python experiments/segmentation/gate37_judge.py"
P="AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-rs"
best=""; best_key=""
for c in "gh156a:g37gh156asdm" "a12h3:a12h3sdm"; do
  s=${c%%:*}; tag=${c#*:}
  e=$($J e1a "$tag" 2>/dev/null) || { log "tab1: $s not passed stage 1"; continue; }
  $J guard "$tag" > /dev/null 2>&1 || { log "tab1: $s not passed stage 3"; continue; }
  key=$(.venv-ag3s/bin/python -c "import json,sys; c=json.loads(sys.argv[1].split('cand=')[1].split('  ref(')[0]); print(f\"{c['ssr']:03d}{c['car']:03d}\")" "$e")
  log "tab1: $s passed (E1a ssr/car key $key)"
  if [ -z "$best_key" ] || [ "$key" \> "$best_key" ]; then best=$s; best_key=$key; fi
done
if [ -n "$best" ]; then V="$P-$best-fwd-tro"; else V="$P-fwd-tro"; fi
VGT="$P-tro-gto"
log "tab1: chosen AG3S version $V (best=${best:-frozen}); GT oracle $VGT"
git merge --ff-only wip/sota-hf >> "$L" 2>&1 || { log "tab1: main fast-forward FAILED -- not starting"; exit 1; }
log "tab1: main HEAD $(git rev-parse --short HEAD)"
DRY=$(mktemp);
source scripts/setup/env.sh > /dev/null 2>&1; cd /mnt/dev/AG3S_VLA
.venv-ag3s/bin/python benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --record-constraints full \
  --ag3s-version "$V" --task-suite-name safelibero_spatial --safety-level I --seed 7 --task-ids 0 --episode-ids 7 \
  --safelibero-step-budget main_aegis --run-id dryrun-tab1fwd --dry-run > "$DRY" 2>&1
if ! grep -q "attention-inpass" "$DRY"; then log "tab1: dry-run has no --attention-inpass -- not starting (see $DRY)"; exit 1; fi
log "tab1: dry-run ok (--attention-inpass in server argv)"
V="$V" VGT="$VGT" ARMS="ag3s gto" bash experiments/segmentation/paper_tab1_tenth_queue.sh
log "tab1: AG3S + GT oracle arms end"
