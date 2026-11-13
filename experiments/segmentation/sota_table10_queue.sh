#!/usr/bin/env bash
# One tenth of the episodes planned for the paper tables, with the frozen AG3S version (sdm-f6, 2026-11-12; user instruction
# 2026-11-12 21:08: "run 1/10 of the episodes planned for LIBERO / SafeLIBERO / LIBERO-Safety, show the result as the real
# table, then analyse the failure causes"). Planned episodes = experiments/segmentation/sdm_pi05_anchor_queue.sh P1-P3:
#   SafeLIBERO 4 suites x L-I/L-II x task 0-3 x init 19-28 = 320  -> init 19 only                      = 32 ep  (part A)
#   LIBERO     4 suites x task 0-9 x init 19-23            = 200  -> task 0,2,4,6,8 x init 19           = 20 ep  (part B)
#   LIBERO-Safety 4 suites x L0-L2 x task 0-4 x init 19    =  60  -> AAG/TSA/FSHOA x L2 x task 0,1      =  6 ep  (part C)
# Parts S + C run the AG3S arm on the official pi05_libero_safety checkpoint through SL-LS1 (controller conversion, 2026-11-12).
# Part D = pi0.5 alone on the same episodes where no SDM run exists yet (SafeLIBERO Long, LIBERO, LIBERO-Safety with both
# checkpoints); SafeLIBERO Spatial / Object / Goal init 19 are in the sdm-pi05-sl-* runs (sdm-c7).
# NOTE: init 19 belongs to the reporting split (init 19-49). After this run it has been looked at for failure analysis.
# usage: [PARTS="A B S C D M N G P F"] bash experiments/segmentation/sota_table10_queue.sh
# The launchers take the GPU lock themselves. Do not change this checkout's HEAD while it runs.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv-ag3s/bin/python
V=AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-rs-tro
TAG=tab10sdm
PARTS=${PARTS:-"A B C D"}
DRY=${DRY:-0}
OUT=outputs/2026-11-11_SOTA_개선/eval_$TAG
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() {
  local lab=$1; shift
  if [ "$DRY" = 1 ]; then "$@" --dry-run > /dev/null 2>> "$LOG.dry" && log "dry ok $lab" || log "dry FAILED $lab"; return; fi
  log "start $lab"; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"
}
label() { $PY -c "import sys; from benchmark.runners import suite_registry as r; print(r.split_label(sys.argv[1], None if sys.argv[2] == '-' else sys.argv[2], 'test'))" "$1" "$2"; }
LP="benchmark/runners/launch_libero_paired.py --conversion C"
LS="benchmark/runners/launch_libero_safety_paired.py --conversion C --seed 7 --data-split test --violation-termination official"
LB_TASKS=0,2,4,6,8
LS_SUITES="affordance obstacle_avoidance obstacle_avoidance_human"
log "table 1/10 queue: HEAD $(git rev-parse --short HEAD) parts='$PARTS' dry=$DRY V=$V"
if [[ " $PARTS " == *" A "* ]]; then
  if [ "$DRY" = 1 ]; then log "part A dry: skipped (sota_ms_queue.sh has no dry mode)"; else
    V=$V TAG=$TAG INITS=19 CELLS="spatial:I object:I goal:I long:I spatial:II object:II goal:II long:II" bash experiments/segmentation/sota_ms_queue.sh
  fi
  log "part A end"
fi
if [[ " $PARTS " == *" B "* ]]; then
  for suite in libero_spatial libero_object libero_goal libero_10; do
    run "B AG3S $suite task $LB_TASKS init 19" $PY $LP --modes vla_v11ab --record-constraints full --ag3s-version "$V" --task-suite-name $suite \
      --task-ids $LB_TASKS --episode-ids 19 --seed 7 --data-split "$(label $suite -)" --run-id "$TAG-lb-${suite#libero_}"
  done
  log "part B end"
fi
# Parts S / C (SL-LS1, 2026-11-12 23:5x): the AG3S arm of LIBERO-Safety on the OFFICIAL pi05_libero_safety checkpoint, version
# $V-lsnative (--response-model libero_safety_fork_native; the launcher sets the client adapter fork_native_unit).
# S = the smoke block (obstacle_avoidance = TSA, 2 ep; look at the conversion record before C), C = the other two suites.
# run-id tab10sdm-lsb: the first smoke (tab10sdm-ls, 2026-11-12 23:50) used the internal unit (1/16 m, 1/2 rad) and is superseded --
# a saturated action-QP row moved the hand 62.5 mm per step; the unit is now (1/32 m, 1/8 rad) (action.NATIVE_UNIT).
VN=$V-lsnative
ls_ag3s() {
  run "$1 AG3S-lsnative (pi05_libero_safety) LIBERO-Safety $2 L2 task 0,1 init 19" $PY $LS --modes vla_v11ab --record-constraints full --ag3s-version "$VN" \
    --policy-checkpoint pi05_libero_safety --task-suite-name $2 --level 2 --task-ids 0,1 --episode-ids 19 --run-id "$TAG-lsb"
}
if [[ " $PARTS " == *" S "* ]]; then
  ls_ag3s S obstacle_avoidance
  log "part S end"
fi
if [[ " $PARTS " == *" C "* ]]; then
  for suite in affordance obstacle_avoidance_human; do ls_ag3s C $suite; done
  log "part C end"
fi
# Part M (SL-SOTA18, 2026-11-13): the mug episodes of this run with $V + --graspable-hand-sized-mm 156 (version ...-gh156-tro).
# SafeLIBERO Long task 2,3 x L-I/L-II x init 19 (4 ep; frozen version: 0 safe successes) and LIBERO-10 task 4,6 x init 19 (2 ep; 0).
if [[ " $PARTS " == *" M "* ]]; then
  VG=${V%-tro}-gh156-tro
  for lvl in I II; do
    run "M gh156 safelibero_long L$lvl task 2,3 init 19" $PY $LP --modes vla_v11ab --record-constraints full --ag3s-version "$VG" \
      --task-suite-name safelibero_long --safety-level $lvl --task-ids 2,3 --episode-ids 19 --seed 7 \
      --data-split "$(label safelibero_long $lvl)" --safelibero-step-budget main_aegis --run-id "$TAG-gh156-long-L$lvl"
  done
  run "M gh156 libero_10 task 4,6 init 19" $PY $LP --modes vla_v11ab --record-constraints full --ag3s-version "$VG" --task-suite-name libero_10 \
    --task-ids 4,6 --episode-ids 19 --seed 7 --data-split "$(label libero_10 -)" --run-id "$TAG-gh156-lb-10"
  log "part M end"
fi
# Part N (2026-11-13): part M's six episodes with -hf30-gh156 (SL-SOTA17 held band 30 mm + SL-SOTA18).
if [[ " $PARTS " == *" N "* ]]; then
  VH=${V%-tro}-hf30-gh156-tro
  for lvl in I II; do
    run "N hf30-gh156 safelibero_long L$lvl task 2,3 init 19" $PY $LP --modes vla_v11ab --record-constraints full --ag3s-version "$VH" \
      --task-suite-name safelibero_long --safety-level $lvl --task-ids 2,3 --episode-ids 19 --seed 7 \
      --data-split "$(label safelibero_long $lvl)" --safelibero-step-budget main_aegis --run-id "$TAG-hf30gh156-long-L$lvl"
  done
  run "N hf30-gh156 libero_10 task 4,6 init 19" $PY $LP --modes vla_v11ab --record-constraints full --ag3s-version "$VH" --task-suite-name libero_10 \
    --task-ids 4,6 --episode-ids 19 --seed 7 --data-split "$(label libero_10 -)" --run-id "$TAG-hf30gh156-lb-10"
  log "part N end"
fi
# Part G (2026-11-13, plan §9.32 stage 2): the rest of the 1/10 set with -gh156 (part M covered Long task 2,3 and LIBERO-10 task 4,6):
# SafeLIBERO 28 ep (6 cells x task 0-3, Long x task 0,1) + LIBERO 18 ep (spatial/object/goal x task 0,2,4,6,8, libero_10 x task 0,2,8).
if [[ " $PARTS " == *" G "* ]]; then
  VG=${V%-tro}-gh156-tro
  for cell in spatial:I object:I goal:I long:I spatial:II object:II goal:II long:II; do
    suite=${cell%%:*}; lvl=${cell#*:}; tasks=0,1,2,3; [ "$suite" = long ] && tasks=0,1
    run "G gh156 safelibero_$suite L$lvl task $tasks init 19" $PY $LP --modes vla_v11ab --record-constraints full --ag3s-version "$VG" \
      --task-suite-name safelibero_$suite --safety-level $lvl --task-ids $tasks --episode-ids 19 --seed 7 \
      --data-split "$(label safelibero_$suite $lvl)" --safelibero-step-budget main_aegis --run-id "$TAG-gh156r-$suite-L$lvl"
  done
  for suite in libero_spatial libero_object libero_goal libero_10; do
    tasks=$LB_TASKS; [ "$suite" = libero_10 ] && tasks=0,2,8
    run "G gh156 $suite task $tasks init 19" $PY $LP --modes vla_v11ab --record-constraints full --ag3s-version "$VG" --task-suite-name $suite \
      --task-ids $tasks --episode-ids 19 --seed 7 --data-split "$(label $suite -)" --run-id "$TAG-gh156r-lb-${suite#libero_}"
  done
  log "part G end"
fi
# Part P (2026-11-13, SL-SOTA18b): part M's six mug episodes with -gh156a (hand-sized clusters must also lead on attention).
if [[ " $PARTS " == *" P "* ]]; then
  VA=${V%-tro}-gh156a-tro
  for lvl in I II; do
    run "P gh156a safelibero_long L$lvl task 2,3 init 19" $PY $LP --modes vla_v11ab --record-constraints full --ag3s-version "$VA" \
      --task-suite-name safelibero_long --safety-level $lvl --task-ids 2,3 --episode-ids 19 --seed 7 \
      --data-split "$(label safelibero_long $lvl)" --safelibero-step-budget main_aegis --run-id "$TAG-gh156a-long-L$lvl"
  done
  run "P gh156a libero_10 task 4,6 init 19" $PY $LP --modes vla_v11ab --record-constraints full --ag3s-version "$VA" --task-suite-name libero_10 \
    --task-ids 4,6 --episode-ids 19 --seed 7 --data-split "$(label libero_10 -)" --run-id "$TAG-gh156a-lb-10"
  log "part P end"
fi
# Part F (2026-11-13 11:0x): the FROZEN version in the run layout of parts M / N / P. An episode's outcome depends on its
# position in the run (same seed, same scene, different policy output at call 0 when the run holds other episodes before it:
# E1a task 0 init 10 alone vs inside init 5-14), so the mug-option runs are compared with the frozen version run the same way.
if [[ " $PARTS " == *" F "* ]]; then
  for lvl in I II; do
    run "F frozen safelibero_long L$lvl task 2,3 init 19" $PY $LP --modes vla_v11ab --record-constraints full --ag3s-version "$V" \
      --task-suite-name safelibero_long --safety-level $lvl --task-ids 2,3 --episode-ids 19 --seed 7 \
      --data-split "$(label safelibero_long $lvl)" --safelibero-step-budget main_aegis --run-id "$TAG-frz-long-L$lvl"
  done
  run "F frozen libero_10 task 4,6 init 19" $PY $LP --modes vla_v11ab --record-constraints full --ag3s-version "$V" --task-suite-name libero_10 \
    --task-ids 4,6 --episode-ids 19 --seed 7 --data-split "$(label libero_10 -)" --run-id "$TAG-frz-lb-10"
  log "part F end"
fi
if [[ " $PARTS " == *" D "* ]]; then
  for lvl in I II; do
    run "D pi05 safelibero_long L$lvl init 19" $PY $LP --modes vla_only --task-suite-name safelibero_long --safety-level $lvl --task-ids 0,1,2,3 \
      --episode-ids 19 --seed 7 --data-split "$(label safelibero_long $lvl)" --safelibero-step-budget main_aegis --run-id "$TAG-pi05-sl-long-L$lvl"
  done
  for suite in libero_spatial libero_object libero_goal libero_10; do
    run "D pi05 $suite task $LB_TASKS init 19" $PY $LP --modes vla_only --task-suite-name $suite --task-ids $LB_TASKS --episode-ids 19 --seed 7 \
      --data-split "$(label $suite -)" --run-id "$TAG-pi05-lb-${suite#libero_}"
  done
  for ck in pi05_libero_safety; do   # the pi05_libero pair was dropped (zero-shot pi0.5 is 0.7 % SR in the paper, Table A.10)
    for suite in $LS_SUITES; do
      run "D pi05 ($ck) LIBERO-Safety $suite L2 task 0,1 init 19" $PY $LS --modes vla_only --policy-checkpoint $ck \
        --task-suite-name $suite --level 2 --task-ids 0,1 --episode-ids 19 --run-id "$TAG-pi05-ls"
    done
  done
  log "part D end"
fi
log "table 1/10 queue end"
