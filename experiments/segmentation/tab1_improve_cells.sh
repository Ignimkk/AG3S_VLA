#!/usr/bin/env bash
# Improvement experiment on paper Tab. 1 cells (sdm-f6 SL-AUD session, 2026-11-15 14:1x; user: "L12H3 + 팔 link strict
# (link 7 까지) 로 1 시간 분량, Goal L-I + Spatial L-II"). Same launcher arguments, inits (PAPER_TENTH_SPLIT_20261113.json),
# seed and run layout as paper_tab1_tenth_queue.sh, so episodes pair with the Tab. 1 AG3S / GT oracle / pi0.5 arms;
# run-id tab1sdm-$TAG-sl-<suite>-L<lvl>.
# A cell starts only if now + EST <= CUTOFF (epoch s; empty = none): the GPU must be free for the real-robot session.
# usage: V=<version> TAG=<tag> CELLS="goal:I spatial:II" [CUTOFF=...] [EST=2700] bash tab1_improve_cells.sh
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv-ag3s/bin/python
: "${V:?}" "${TAG:?}" "${CELLS:?}"
CUTOFF=${CUTOFF:-}
EST=${EST:-2700}
SPLIT=doc/active/experiments/PAPER_TENTH_SPLIT_20261113.json
OUT=outputs/2026-11-15_tab1_improve
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
label() { $PY -c "import sys; from benchmark.runners import suite_registry as r; print(r.split_label(sys.argv[1], None if sys.argv[2] == '-' else sys.argv[2], 'test'))" "$1" "$2"; }
inits() { $PY -c "import json,sys; d=json.load(open('$SPLIT')); print(','.join(map(str, d[sys.argv[1]][sys.argv[2]]['inits'])))" "$1" "$2"; }
LP="benchmark/runners/launch_libero_paired.py --conversion C"
log "tab1 improve: HEAD $(git rev-parse --short HEAD) TAG=$TAG V=$V cells=[$CELLS] cutoff=${CUTOFF:+$(date -d @$CUTOFF +%H:%M)}"
for cell in $CELLS; do
  suite=${cell%%:*}; lvl=${cell#*:}
  if [ -n "$CUTOFF" ] && [ $(( $(date +%s) + EST )) -gt "$CUTOFF" ]; then
    log "tab1 improve: CUTOFF -- cell $suite L$lvl (est ${EST}s) would end after $(date -d @$CUTOFF +%H:%M); stopped, GPU free"; exit 3
  fi
  ids=$(inits safelibero "$suite L-$lvl")
  log "start $TAG safelibero_$suite L$lvl task 0-3 init $ids"
  $PY $LP --modes vla_v11ab --record-constraints full --ag3s-version "$V" --task-suite-name safelibero_$suite --safety-level $lvl \
    --task-ids 0,1,2,3 --episode-ids $ids --seed 7 --data-split "$(label safelibero_$suite $lvl)" --safelibero-step-budget main_aegis \
    --run-id "tab1sdm-$TAG-sl-$suite-L$lvl" >> "$LOG.out" 2>&1
  log "done rc=$?"
done
log "tab1 improve end"
