#!/usr/bin/env bash
# Resume of baseline tranche 1 (dev-4a, 2026-11-10 01:2x KST; user decision 01:1x "F2 철회, 즉시 재개"):
#   * F2 (per-env-step filter) withdrawn: every filter arm stays chunk-level (--filter-rate chunk = launcher default), a
#     declared deviation shared by all arms -> the finished runs of bl_tranche_queue.sh (17:39-00:11, HEAD c027b00d + the
#     F1 / naming patches recorded in each source.patch) are reused where the method code they ran is still the reported one.
#   * Reruns (suffix -r2): AEGIS privileged names spatial LI (ran before the prefer_list naming fix) and the four KNOWS spatial
#     runs (ran before F3 track freeze + F4a attention target exclusion, now the KNOWS defaults, HEAD bfa9bdb0).
#   * Then the not-yet-run blocks in the same scenario-major order and protocol as bl_tranche_queue.sh.
# Reused (not rerun): none spatial LI / LII, object LI; aegis spatial LI / LII, object LI; aegis-gt spatial LII, object LI;
#   multilink_cbf (+gt) spatial LI / LII.  (Doc: BASELINE_TRANCHE_PLAN_PORT_RATIONALE_20261109_KO.md §4.7)
# usage: bash experiments/segmentation/bl_tranche_resume_queue.sh      (INITS / T / PARTS as bl_tranche_queue.sh)
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
# finished + still valid ("<tag> <suite> L<lvl>")
DONE=" none:spatial:LI aegis:spatial:LI multilink_cbf:spatial:LI multilink_cbf-gt:spatial:LI
 none:spatial:LII aegis:spatial:LII aegis-gt:spatial:LII multilink_cbf:spatial:LII multilink_cbf-gt:spatial:LII
 none:object:LI aegis:object:LI aegis-gt:object:LI "
# rerun with suffix -r2 (the first run stays on disk as the pre-fix record)
RERUN=" aegis-gt:spatial:LI knows:spatial:LI knows-gt:spatial:LI knows:spatial:LII knows-gt:spatial:LII "
DONE=" $(echo $DONE) "; RERUN=" $(echo $RERUN) "   # one line, single spaces (membership test below)
log "baseline tranche $T RESUME: HEAD $(git rev-parse --short HEAD) inits $INITS parts $PARTS filter-rate chunk (F2 withdrawn)"
safe_run() {  # suite lvl arm-spec
  local suite=$1 lvl=$2 a=$3 split m g tag key sfx=""
  m=${a%%|*}; g=${a#*|}; tag=$m${g:+-gt}; key="$tag:$suite:L$lvl"
  [[ "$DONE" == *" $key "* ]] && return 0
  [[ "$RERUN" == *" $key "* ]] && sfx="-r2"
  split=$(label safelibero_$suite $lvl)
  run "$T $tag safelibero_$suite L$lvl$sfx" $PY $LM --method $m $g --task-suite-name safelibero_$suite --safety-level $lvl \
    --task-ids 0,1,2,3 --episode-ids $INITS --seed 7 --data-split "$split" --safelibero-step-budget main_aegis \
    --run-id bl$T-$tag-$suite-L$lvl$sfx
}
if [[ " $PARTS " == *" A "* ]]; then
  # 1) reruns first, so the spatial blocks are complete paired blocks again
  safe_run spatial I "aegis|--gt"
  for lvl in I II; do safe_run spatial $lvl "knows|"; safe_run spatial $lvl "knows|--gt"; done
  # 2) the remaining blocks, scenario-major
  for suite in spatial object goal long; do
    for lvl in I II; do
      for a in "${SAFE_ARMS[@]}"; do
        key="$(m=${a%%|*}; g=${a#*|}; echo "$m${g:+-gt}:$suite:L$lvl")"
        [[ "$RERUN" == *" $key "* ]] && continue   # done in 1)
        safe_run $suite $lvl "$a"
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
