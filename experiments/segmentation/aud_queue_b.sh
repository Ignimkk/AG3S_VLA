#!/usr/bin/env bash
# SL-AUD, filter 1 rerun with the fixed -rf (sdm-f6, 2026-11-14 03:4x; user decision; hand-off plan §10 = 0f1d90b4 / 6964b198,
# fixed before the result; AG3S_PLAN_SCHEDULE §9.41 보충 2). Takes over from aud_queue.sh (audq) at its boundary:
#   0) audq runs filter 2 and -md5 to the end; right after the -md5 judgement line (before audq's adoption gate / Tab. 1)
#      its process group is stopped. Any run dir audq created after that line is marked INTERRUPTED.
#   1) main fast-forward to wip/sl-aud (the -rf fix 0f1d90b4)
#   2) first-episode check (§10): the -rf version, Spatial L-I task 0 init 15 (= the S32 spatial run's first episode, same
#      arguments) -- every action-QP call must carry reference_floor.{n_rows_below_backoff, min_floor_m}, no why / error /
#      shape_mismatch; otherwise STOP
#   3) S32 with <frozen>-rf-lp-mr8-af-tro, TAG aud1bsdm; §5 criteria
#   4) adoption: segments = -rf-lp-mr8-af if the rerun passes (it fails -> the action-QP side is closed, -md5 included,
#      §10) + -ohr-dlo-pro if filter 2 passed; X = -gh156a (gate 01:09). Adoption gate = E1a 40 + guard as -gh156a.
#   5) Tab. 1 AG3S arm = the final version's -fwd twin, GT oracle arm = <frozen>-tro-gto
# Run detached: bash experiments/segmentation/run_detached.sh audqb 'bash /mnt/dev/AG3S_VLA-wt-aud/experiments/segmentation/aud_queue_b.sh'
set -u -o pipefail
cd /mnt/dev/AG3S_VLA
OUTD=outputs/2026-11-14_SL-AUD
L=$OUTD/queue.log
LT=outputs/2026-11-13_paper_tab1_1-10/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$L"; }
run() { log "start $1"; shift; "$@" >> "$L.out" 2>&1; log "done rc=$?"; }
PY=.venv-ag3s/bin/python
P="AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-rs"
X="$P-gh156a-tro"
V1="$P-rf-lp-mr8-af-tro"
WT=/mnt/dev/AG3S_VLA-wt-aud/experiments/segmentation
J="$PY experiments/segmentation/gate37_judge.py"
A="$PY $WT/aud_judge.py"
RC="$PY $WT/aud_rf_check.py"
B=/mnt/dev/AG3S_VLA-storage/results/benchmark
registered() { $PY -c "import sys; from AG3S.adapters.panda import profile as pf; sys.exit(0 if sys.argv[1] in pf.AG3S_VERSIONS else 1)" "$1"; }

# 0) take over from audq right after its -md5 judgement
QP=$(cat outputs/detached/audq.pid)
log "audb: waiting for audq ($QP) -- filter 2, then -md5; take over at the -md5 judgement (user: rerun filter 1 with the fixed -rf)"
n0=$(wc -l < "$L")
took=0
while kill -0 "$QP" 2>/dev/null; do
  new=$(tail -n +"$((n0 + 1))" "$L")
  if grep -qE "aud s32 aud1md5sdm|aud: (adoption gate|no filter passed|FINAL)|STOP" <<< "$new"; then
    stamp=$(mktemp); touch -d "-3 seconds" "$stamp"
    kill -TERM -- "-$QP" 2>/dev/null; sleep 20; kill -KILL -- "-$QP" 2>/dev/null
    for d in $(find "$B" -maxdepth 1 -mindepth 1 -type d -newer "$stamp" 2>/dev/null); do
      [ -f "$d/episodes.jsonl" ] || echo "$(date +%Y-%m-%dT%H:%M:%S) INTERRUPTED: audq stopped at its boundary (SL-AUD filter 1 rerun first). Do NOT score." > "$d/INTERRUPTED.txt"
    done
    log "audb: stopped audq at its boundary: $(grep -E 'aud s32 aud1md5sdm|aud: (adoption gate|no filter passed|FINAL)|STOP' <<< "$new" | tail -1 | cut -c1-160)"
    took=1; break
  fi
  sleep 1
done
[ $took = 1 ] || log "audb: audq ended by itself before the takeover point"
f2=$($A s32 aud2sdm f2 2>&1); f2rc=$?; log "audb: filter 2 (re-read) $f2"
[ $f2rc -le 1 ] || { log "audb: filter 2 INCOMPLETE -- STOP"; exit 1; }
if ! grep -q "aud s32 aud1md5sdm" "$L"; then log "audb: no -md5 judgement in the log -- STOP (check audq)"; exit 1; fi

# 1) main fast-forward (queue boundary)
git merge --ff-only wip/sl-aud >> "$L" 2>&1 || { log "audb: main fast-forward to wip/sl-aud FAILED -- STOP"; exit 1; }
log "audb: main HEAD $(git rev-parse --short HEAD) (has the -rf fix: $(git merge-base --is-ancestor 2a27e4c4 HEAD && echo yes || echo NO))"
git merge-base --is-ancestor 2a27e4c4 HEAD || { log "audb: the -rf fix is not in main -- STOP"; exit 1; }
source scripts/setup/env.sh > /dev/null 2>&1; cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LPB="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --record-constraints full"
label() { $PY -c "import sys; from benchmark.runners import suite_registry as r; print(r.split_label(sys.argv[1], sys.argv[2], 'test'))" "$1" "$2"; }

# 2) first-episode check of the fixed -rf (§10)
run "audb check: -rf-lp-mr8-af spatial LI task 0 init 15" $PY $LPB --ag3s-version "$V1" --task-suite-name safelibero_spatial \
  --safety-level I --task-ids 0 --episode-ids 15 --seed 7 --data-split "$(label safelibero_spatial I)" \
  --safelibero-step-budget main_aegis --run-id ms-aud1bchksdm-spatial-LI-i15
DN=$(ls -d $B/2*_ms-aud1bchksdm-spatial-LI-i15-* 2>/dev/null | tail -1)
if [ -z "$DN" ] || ! $RC "$DN" safelibero_spatial_t0_e15 2>&1 | tee -a "$L"; then
  log "audb: -rf first-episode check NOT OK (or no run) -- STOP before the rerun (§10: report the values)"; exit 1
fi
log "audb: -rf first-episode check OK"

# 3) filter 1 rerun on S32
log "audb: filter 1 rerun (-rf-lp-mr8-af, fixed -rf) S32, TAG aud1bsdm"
V="$V1" TAG=aud1bsdm INITS=15,16,17,18 CELLS="spatial:I" bash experiments/segmentation/sota_ms_queue.sh
V="$V1" TAG=aud1bsdm INITS=17,18 CELLS="goal:I goal:II" bash experiments/segmentation/sota_ms_queue.sh
f1=$($A s32 aud1bsdm f1 2>&1); f1rc=$?; log "$f1"
$RC "$(ls -d $B/2*_ms-aud1bsdm-spatial-LI-i15_16_17_18-* | tail -1)" 2>&1 | head -1 | tee -a "$L"

# 4) adoption
SEG=""
[ $f1rc -eq 0 ] && SEG="-rf-lp-mr8-af"
[ $f1rc -ne 0 ] && log "audb: filter 1 rerun FAIL -> the action-QP side is closed (-md5 included, plan §10)"
[ $f2rc -eq 0 ] && SEG="$SEG-ohr-dlo-pro"
FINAL="$X"
if [ -n "$SEG" ]; then
  VA="${X%-tro}$SEG-tro"
  registered "$VA" || { log "audb: adoption version $VA is not registered -- STOP"; exit 1; }
  I="5,6,7,8,9,10,11,12,13,14"
  log "audb: adoption gate $VA: stage 1 E1a 40"
  V="$VA" TAG=audgsdm SUB="0:$I 1:$I 2:$I 3:$I" bash experiments/segmentation/sota_sub_queue.sh
  if $J e1a audgsdm 2>&1 | tee -a "$L"; then
    log "audb: adoption gate: stage 3 guard"
    V="$VA" TAG=audgsdm OUT=outputs/2026-11-11_SOTA_개선/eval_audgsdm MIN27=21 bash experiments/segmentation/gtto16_queue.sh
    if $J guard audgsdm 2>&1 | tee -a "$L"; then FINAL="$VA"; log "audb: adoption gate PASS"; else log "audb: adoption gate stage 3 FAIL"; fi
  else
    log "audb: adoption gate stage 1 FAIL"
  fi
else
  log "audb: nothing passed -- no adoption gate"
fi
log "audb: FINAL frozen version = $FINAL"

# 5) Tab. 1
if [ "$FINAL" = "$X" ]; then VT="${X%-tro}-fwd-tro"; else VT="${X%-tro}-fwd$SEG-tro"; fi
VGT="$P-tro-gto"
registered "$VT" || { log "audb: Tab. 1 version $VT is not registered -- STOP"; exit 1; }
DRY=$(mktemp)
$PY $LPB --ag3s-version "$VT" --task-suite-name safelibero_spatial --safety-level I --seed 7 --task-ids 0 --episode-ids 7 \
  --safelibero-step-budget main_aegis --run-id dryrun-tab1fwd --dry-run > "$DRY" 2>&1
grep -q "attention-inpass" "$DRY" || { log "audb: dry-run of $VT has no --attention-inpass -- STOP (see $DRY)"; exit 1; }
log "audb: Tab. 1 start: AG3S $VT, GT oracle $VGT (dry-run ok)"
echo "$(date +%Y-%m-%dT%H:%M:%S) tab1 (via aud_queue_b.sh): AG3S $VT, GT oracle $VGT" >> "$LT"
V="$VT" VGT="$VGT" ARMS="ag3s gto" bash experiments/segmentation/paper_tab1_tenth_queue.sh
log "audb: Tab. 1 AG3S + GT oracle arms end"
