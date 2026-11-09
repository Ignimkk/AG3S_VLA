#!/usr/bin/env bash
# Baseline tranche queue (dev-4a, 2026-11-09 17:xx KST; user rulings 2026-11-09 16:5x):
#   * these runs are REPORTED results (paper Tab. 1 rows other than AG3S), on a dedicated GPU, port versions for anything
#     that costs money (AEGIS official = GLM-4.5V API -> AEGIS port; rationale in
#     doc/active/experiments/BASELINE_TRANCHE_PLAN_PORT_RATIONALE_20261109_KO.md)
#   * run 1/10 of the planned size first, the other 9/10 later (same arms, same protocol, more init states).
# Tranche 1 = init states 15-18 (headline split 15-49 of G171; init 5-14 = development split, never used here):
#   A. SafeLIBERO 4 suites x task 0-3 x Level I/II x 4 init = 128 ep/arm, 7 arms (pi0.5, AEGIS port, AEGIS port privileged
#      names, multi-link CBF*, multi-link CBF* GT seg, KNOWS*, KNOWS* GT seg) = 896 ep
#   B. LIBERO (no obstacles) 4 suites x task 0-9 x 4 init = 160 ep/arm, 4 arms (no privileged variants: no obstacle GT) = 640 ep
# Order = scenario-major (all arms of one suite/level before the next), so a stop leaves complete paired blocks.
# Fixed protocol: checkpoint pi05_libero, seed 7, main_aegis step budget, official 1 mm collision rule, launcher
#   Baseline/paper_baselines/closed_loop/launch_method.py (takes the exclusive GPU lock itself -> never wrap in flock).
# usage: [INITS=15,16,17,18] [T=t1] [PARTS="A B"] bash experiments/segmentation/bl_tranche_queue.sh
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
INITS=${INITS:-15,16,17,18}
T=${T:-t1}
PARTS=${PARTS:-"A B"}
OUT=outputs/2026-11-09_baseline_tranche/$T
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
label() { $PY -c "import sys; from benchmark.runners import suite_registry as r; print(r.split_label(sys.argv[1], None if sys.argv[2] == '-' else sys.argv[2], 'test'))" "$1" "$2"; }
LM=Baseline/paper_baselines/closed_loop/launch_method.py
SAFE_ARMS=("none|" "aegis|" "aegis|--gt" "multilink_cbf|" "multilink_cbf|--gt" "knows|" "knows|--gt")
LIBERO_ARMS=("none|" "aegis|" "multilink_cbf|" "knows|")
log "baseline tranche $T: HEAD $(git rev-parse --short HEAD) inits $INITS parts $PARTS"
if [[ " $PARTS " == *" A "* ]]; then
  for suite in spatial object goal long; do
    for lvl in I II; do
      split=$(label safelibero_$suite $lvl)
      for a in "${SAFE_ARMS[@]}"; do
        m=${a%%|*}; g=${a#*|}; tag=$m${g:+-gt}
        run "$T $tag safelibero_$suite L$lvl" $PY $LM --method $m $g --task-suite-name safelibero_$suite --safety-level $lvl \
          --task-ids 0,1,2,3 --episode-ids $INITS --seed 7 --data-split "$split" --safelibero-step-budget main_aegis \
          --run-id bl$T-$tag-$suite-L$lvl
      done
    done
  done
  log "part A end"
fi
if [[ " $PARTS " == *" B "* ]]; then
  for suite in libero_spatial libero_object libero_goal libero_10; do
    split=$(label $suite -)
    for a in "${LIBERO_ARMS[@]}"; do
      m=${a%%|*}; tag=$m
      run "$T $tag $suite" $PY $LM --method $m --task-suite-name $suite --task-ids 0,1,2,3,4,5,6,7,8,9 \
        --episode-ids $INITS --seed 7 --data-split "$split" --run-id bl$T-$tag-${suite#libero_}
    done
  done
  log "part B end"
fi
log "queue baseline $T end"
