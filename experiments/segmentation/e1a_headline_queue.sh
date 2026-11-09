#!/usr/bin/env bash
# E1a headline split (G171, user ruling 2026-11-09): SafeLIBERO Spatial Level I, task 0-3 x init 15-49 (never used for any
# configuration choice), seed 7, main_aegis budget -- the FROZEN AG3S version and its pi0.5-only pair (same launcher).
# Run only after the configuration is frozen (LOCK). usage: V=<frozen version> TAG=<tag> [VLA=1] bash e1a_headline_queue.sh
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
: "${V:?}" "${TAG:?}"
OUT=outputs/2026-11-09_집기놓기_장애물취급/headline_$TAG
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
log "e1a headline 15-49: HEAD $(git rev-parse --short HEAD) V=$V"
IDS=$(seq -s, 15 49)
E1A="--task-suite-name safelibero_spatial --safety-level I --episode-ids $IDS --seed 7 --data-split test_SafeLIBERO_Spatial_ep5-49_user_2026-10-14 --safelibero-step-budget main_aegis"
for t in 0 1 2 3; do
  run "$TAG E1a(headline 15-49) task $t" $PY benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab \
    --record-constraints full --ag3s-version $V $E1A --task-ids $t --run-id qe-$TAG-e1aH-spatial-LI-t$t
done
if [ "${VLA:-1}" = 1 ]; then
  for t in 0 1 2 3; do
    run "vla E1a(headline 15-49) task $t" $PY benchmark/runners/launch_libero_paired.py --conversion C --modes vla_only \
      $E1A --task-ids $t --run-id qe-vla-e1aH-spatial-LI-t$t
  done
fi
log "queue e1a headline end"
