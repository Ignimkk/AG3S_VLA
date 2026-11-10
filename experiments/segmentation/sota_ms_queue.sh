#!/usr/bin/env bash
# SOTA other-suite check "MS" (dev-57, 2026-11-11, AG3S_PLAN_SCHEDULE_20261110_KO.md §9.2): the headline is the mean over
# SafeLIBERO 4 suites x Level I / II, but AG3S was developed on Spatial LI only. One candidate on the other cells, task 0-3,
# seed 7, main_aegis, init 15-18 (= the baseline-port validation split, excluded from the headline; dev-4a tranche 1 has
# pi0.5 / AEGIS / multi-link port runs on the same episodes and machine -> paired).
# usage: V=<AG3S_VERSIONS key> TAG=<tag without hyphen> [INITS=15,16] [CELLS="object:I object:II ..."] bash experiments/segmentation/sota_ms_queue.sh
# run-id ms-$TAG-<suite>-L<level>-i<inits with _>. The launchers take the GPU lock themselves. Do not edit while it runs.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv-ag3s/bin/python
: "${V:?}" "${TAG:?}"
INITS=${INITS:-15,16}
CELLS=${CELLS:-"object:I goal:I long:I object:II goal:II long:II spatial:II"}
OUT=outputs/2026-11-11_SOTA_개선/eval_$TAG
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
label() { $PY -c "import sys; from benchmark.runners import suite_registry as r; print(r.split_label(sys.argv[1], sys.argv[2], 'test'))" "$1" "$2"; }
LPB="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --record-constraints full"
log "sota ms: HEAD $(git rev-parse --short HEAD) V=$V inits=$INITS cells=[$CELLS]"
for cell in $CELLS; do
  suite=${cell%%:*}; lvl=${cell#*:}
  split=$(label safelibero_$suite $lvl)
  run "$TAG MS safelibero_$suite L$lvl init $INITS" $PY $LPB --ag3s-version "$V" --task-suite-name safelibero_$suite --safety-level $lvl \
    --task-ids 0,1,2,3 --episode-ids $INITS --seed 7 --data-split "$split" --safelibero-step-budget main_aegis \
    --run-id "ms-$TAG-$suite-L$lvl-i${INITS//,/_}"
done
log "queue sota ms end ($TAG)"
