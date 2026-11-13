#!/usr/bin/env bash
# SL-AUD before Tab. 1 (sdm-f6, 2026-11-14; hand-off doc/active/handoff_SL_AUD_20261113/2026-11-13_SL-AUD_test_plan_KO.md,
# plan AG3S_PLAN_SCHEDULE_20261110_KO.md §9.41 -- sets, versions, criteria fixed there before any result). Replaces
# tab1_after_gate37.sh (stopped while waiting, user OK 11-14 00:10). Order:
#   0) wait for gate37b; X = the gate's choice (same rule as tab1_after_gate37.sh)
#   1) main (/mnt/dev/AG3S_VLA) fast-forward to wip/sl-aud (SL-AUD code + versions on top of wip/sota-hf)
#   2) frozen version, first episode of the Spatial L-I init 15-18 run (task 0 init 15) again; calls must be identical
#      (count, max_abs_action_change_executed, to.passthrough_clear.reason) -- different -> STOP (no filter)
#   3) filter 1: <frozen>-rf-lp-mr8-af-tro on S32;  4) filter 2: <frozen>-ohr-dlo-pro-tro on S32
#   5) (only if filter 1 says md5) <frozen>-md5-tro on S32, filter 1's criteria
#   6) adoption gate (only if something passed): X + the passed segments, E1a 40 + guard (gate37_judge.py, as -gh156a)
#   7) Tab. 1 AG3S arm = the final version's -fwd twin + GT oracle arm (<frozen>-tro-gto), paper_tab1_tenth_queue.sh
# Run detached (the script lives in the wt-aud worktree; do not edit it while it runs):
#   bash experiments/segmentation/run_detached.sh audq 'bash /mnt/dev/AG3S_VLA-wt-aud/experiments/segmentation/aud_queue.sh'
set -u -o pipefail
cd /mnt/dev/AG3S_VLA
OUTD=outputs/2026-11-14_SL-AUD; mkdir -p "$OUTD"
L=$OUTD/queue.log
LT=outputs/2026-11-13_paper_tab1_1-10/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$L"; }
run() { log "start $1"; shift; "$@" >> "$L.out" 2>&1; log "done rc=$?"; }
PY=.venv-ag3s/bin/python
P="AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-rs"
FROZEN="$P-tro"
J="$PY experiments/segmentation/gate37_judge.py"
A="$PY /mnt/dev/AG3S_VLA-wt-aud/experiments/segmentation/aud_judge.py"
registered() { $PY -c "import sys; from AG3S.adapters.panda import profile as pf; sys.exit(0 if sys.argv[1] in pf.AG3S_VERSIONS else 1)" "$1"; }

# 0) wait for the gate
W=outputs/detached/gate37b.pid
log "aud: waiting for gate37b ($(cat $W))"
while [ -f "$W" ] && kill -0 "$(cat "$W")" 2>/dev/null; do sleep 30; done
best=""; best_key=""
for c in "gh156a:g37gh156asdm" "a12h3:a12h3sdm"; do
  s=${c%%:*}; tag=${c#*:}
  e=$($J e1a "$tag" 2>/dev/null) || { log "aud: gate $s not passed stage 1"; continue; }
  $J guard "$tag" > /dev/null 2>&1 || { log "aud: gate $s not passed stage 3"; continue; }
  key=$($PY -c "import json,sys; c=json.loads(sys.argv[1].split('cand=')[1].split('  ref(')[0]); print(f\"{c['ssr']:03d}{c['car']:03d}\")" "$e")
  log "aud: gate $s passed (E1a ssr/car key $key)"
  if [ -z "$best_key" ] || [ "$key" \> "$best_key" ]; then best=$s; best_key=$key; fi
done
if [ -n "$best" ]; then X="$P-$best-tro"; else X="$FROZEN"; fi
log "aud: gate chose X=$X (best=${best:-frozen})"

# 1) main fast-forward (queue boundary: gate37b has ended)
git merge --ff-only wip/sl-aud >> "$L" 2>&1 || { log "aud: main fast-forward to wip/sl-aud FAILED -- STOP"; exit 1; }
log "aud: main HEAD $(git rev-parse --short HEAD)"
source scripts/setup/env.sh > /dev/null 2>&1; cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LPB="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --record-constraints full"
label() { $PY -c "import sys; from benchmark.runners import suite_registry as r; print(r.split_label(sys.argv[1], sys.argv[2], 'test'))" "$1" "$2"; }

# 2) frozen version unchanged? (first episode of ms-pkgFrssdm-spatial-LI-i15_16_17_18, same launcher arguments)
run "aud check: frozen spatial LI task 0 init 15" $PY $LPB --ag3s-version "$FROZEN" --task-suite-name safelibero_spatial \
  --safety-level I --task-ids 0 --episode-ids 15 --seed 7 --data-split "$(label safelibero_spatial I)" \
  --safelibero-step-budget main_aegis --run-id ms-audchksdm-spatial-LI-i15
B=/mnt/dev/AG3S_VLA-storage/results/benchmark
DN=$(ls -d $B/2*_ms-audchksdm-spatial-LI-i15-* 2>/dev/null | tail -1)
DR=$(ls -d $B/2*_ms-pkgFrssdm-spatial-LI-i15_16_17_18-* | tail -1)
if [ -z "$DN" ] || ! $A same "$DN" "$DR" safelibero_spatial_t0_e15 2>&1 | tee -a "$L"; then
  log "aud: frozen version check DIFFERENT (or no run) -- STOP before the filters (plan §3: report)"; exit 1
fi
log "aud: frozen version check SAME"

s32run() {   # V TAG
  V="$1" TAG="$2" INITS=15,16,17,18 CELLS="spatial:I" bash experiments/segmentation/sota_ms_queue.sh
  V="$1" TAG="$2" INITS=17,18 CELLS="goal:I goal:II" bash experiments/segmentation/sota_ms_queue.sh
}
# 3) filter 1
log "aud: filter 1 (-rf-lp-mr8-af) S32"
s32run "$P-rf-lp-mr8-af-tro" aud1sdm
f1=$($A s32 aud1sdm f1 2>&1); f1rc=$?; log "$f1"
# 4) filter 2
log "aud: filter 2 (-ohr-dlo-pro) S32"
s32run "$P-ohr-dlo-pro-tro" aud2sdm
f2=$($A s32 aud2sdm f2 2>&1); f2rc=$?; log "$f2"
# 5) md5 (only when filter 1 says so)
md5rc=1
if grep -q "next=md5" <<< "$f1"; then
  log "aud: filter 1 -> md5 on S32"
  s32run "$P-md5-tro" aud1md5sdm
  m=$($A s32 aud1md5sdm f1 2>&1); md5rc=$?; log "$m"
fi
SEG=""
[ $f1rc -eq 0 ] && SEG="-rf-lp-mr8-af"
[ $md5rc -eq 0 ] && SEG="$SEG-md5"
[ $f2rc -eq 0 ] && SEG="$SEG-ohr-dlo-pro"
FINAL="$X"
if [ -n "$SEG" ]; then
  VA="${X%-tro}$SEG-tro"
  registered "$VA" || { log "aud: adoption version $VA is not registered -- STOP (register it, then resume from step 6)"; exit 1; }
  # 6) adoption gate, same stages and criteria as -gh156a (plan §9.2 / §9.37: E1a 40 then guard)
  I="5,6,7,8,9,10,11,12,13,14"
  log "aud: adoption gate $VA: stage 1 E1a 40"
  V="$VA" TAG=audgsdm SUB="0:$I 1:$I 2:$I 3:$I" bash experiments/segmentation/sota_sub_queue.sh
  if $J e1a audgsdm 2>&1 | tee -a "$L"; then
    log "aud: adoption gate: stage 3 guard"
    V="$VA" TAG=audgsdm OUT=outputs/2026-11-11_SOTA_개선/eval_audgsdm MIN27=21 bash experiments/segmentation/gtto16_queue.sh
    if $J guard audgsdm 2>&1 | tee -a "$L"; then FINAL="$VA"; log "aud: adoption gate PASS"; else log "aud: adoption gate stage 3 FAIL"; fi
  else
    log "aud: adoption gate stage 1 FAIL"
  fi
else
  log "aud: no filter passed -- no adoption gate"
fi
log "aud: FINAL frozen version = $FINAL"

# 7) Tab. 1: AG3S arm = FINAL's -fwd twin; GT oracle arm = <frozen>-tro-gto
if [ "$FINAL" = "$X" ]; then VT="${X%-tro}-fwd-tro"; else VT="${X%-tro}-fwd$SEG-tro"; fi
VGT="$P-tro-gto"
registered "$VT" || { log "aud: Tab. 1 version $VT is not registered -- STOP"; exit 1; }
DRY=$(mktemp)
$PY $LPB --ag3s-version "$VT" --task-suite-name safelibero_spatial --safety-level I --seed 7 --task-ids 0 --episode-ids 7 \
  --safelibero-step-budget main_aegis --run-id dryrun-tab1fwd --dry-run > "$DRY" 2>&1
grep -q "attention-inpass" "$DRY" || { log "aud: dry-run of $VT has no --attention-inpass -- STOP (see $DRY)"; exit 1; }
log "aud: Tab. 1 start: AG3S $VT, GT oracle $VGT (dry-run ok)"
echo "$(date +%Y-%m-%dT%H:%M:%S) tab1 (via aud_queue.sh): AG3S $VT, GT oracle $VGT" >> "$LT"
V="$VT" VGT="$VGT" ARMS="ag3s gto" bash experiments/segmentation/paper_tab1_tenth_queue.sh
log "aud: Tab. 1 AG3S + GT oracle arms end"
