#!/usr/bin/env bash
# Baseline tranche: run only the given arms over every block, skipping blocks already finished (dev-4a, 2026-11-10 02:4x).
# Why: arms whose port version is settled keep the GPU busy while others are being fixed (2026-11-10: AEGIS / multi-link CBF
# detector arms -> VLM naming; KNOWS -> F3/F4a collapse diagnosis), instead of idling or producing runs to be discarded.
# A block counts as done when a run dir bl<T>-<tag>-<suite>-L<lvl>[-r2|-r3] with closed_loop_metrics.json exists
# (non-INVALID); new runs get SUFFIX (default empty) on the run-id. Same protocol as bl_tranche_queue.sh.
# usage: SAFE_ARMS_SEL="none| aegis|--gt multilink_cbf|--gt" LIBERO_ARMS_SEL="none|" [SUFFIX=-r2] [INITS=..] [T=t1] \
#        bash experiments/segmentation/bl_tranche_arms_queue.sh
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
INITS=${INITS:-15,16,17,18}
T=${T:-t1}
SUFFIX=${SUFFIX:-}
read -r -a SAFE_ARMS <<< "${SAFE_ARMS_SEL:-}"
read -r -a LIBERO_ARMS <<< "${LIBERO_ARMS_SEL:-}"
OUT=outputs/2026-11-09_baseline_tranche/$T
mkdir -p "$OUT"
LOG=$OUT/queue.log
STORE=/mnt/dev/AG3S_VLA-storage/results/benchmark
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
label() { $PY -c "import sys; from benchmark.runners import suite_registry as r; print(r.split_label(sys.argv[1], None if sys.argv[2] == '-' else sys.argv[2], 'test'))" "$1" "$2"; }
done_run() {  # run-id stem -> 0 if a finished, non-INVALID run exists (SUFFIX empty: any of "", -r2, -r3; else SUFFIX, -r3)
  local stem=$1 sfx d cands=("" -r2 -r3)
  [ -n "$SUFFIX" ] && cands=("$SUFFIX" -r3)
  for sfx in "${cands[@]}"; do
    for d in "$STORE"/*_"$stem$sfx"-*; do
      [[ "${2:-}" == libero && ( "$d" == *_"$stem$sfx"-LI-* || "$d" == *_"$stem$sfx"-LII-* ) ]] && continue  # SafeLIBERO run
      [[ -e "$d/closed_loop_metrics.json" && "$(basename "$d")" != INVALID* ]] && return 0
    done
  done
  return 1
}
LM=Baseline/paper_baselines/closed_loop/launch_method.py
log "baseline tranche $T ARMS queue: HEAD $(git rev-parse --short HEAD) safe=[${SAFE_ARMS[*]:-}] libero=[${LIBERO_ARMS[*]:-}] suffix='$SUFFIX'"
for suite in spatial object goal long; do
  for lvl in I II; do
    split=$(label safelibero_$suite $lvl)
    for a in "${SAFE_ARMS[@]}"; do
      m=${a%%|*}; g=${a#*|}; tag=$m${g:+-gt}; stem=bl$T-$tag-$suite-L$lvl
      if done_run "$stem"; then continue; fi
      run "$T $tag safelibero_$suite L$lvl$SUFFIX" $PY $LM --method $m $g --task-suite-name safelibero_$suite --safety-level $lvl \
        --task-ids 0,1,2,3 --episode-ids $INITS --seed 7 --data-split "$split" --safelibero-step-budget main_aegis \
        --run-id $stem$SUFFIX
    done
  done
done
for suite in libero_spatial libero_object libero_goal libero_10; do
  split=$(label $suite -)
  for a in "${LIBERO_ARMS[@]}"; do
    m=${a%%|*}; tag=$m; stem=bl$T-$tag-${suite#libero_}
    if done_run "$stem" libero; then continue; fi
    run "$T $tag $suite$SUFFIX" $PY $LM --method $m --task-suite-name $suite --task-ids 0,1,2,3,4,5,6,7,8,9 \
      --episode-ids $INITS --seed 7 --data-split "$split" --run-id $stem$SUFFIX
  done
done
log "queue baseline $T arms [${SAFE_ARMS[*]:-} / ${LIBERO_ARMS[*]:-}] end"
