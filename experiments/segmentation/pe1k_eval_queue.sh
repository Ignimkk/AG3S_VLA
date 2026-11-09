#!/usr/bin/env bash
# SL-PE evaluation protocol (dev-4a, 2026-11-09, G163-): the version chosen from the mechanism checks.
# 1. calibration LIBERO 9 x seeds 10 · 11 (not the evaluation seeds), 2. gtto16_queue.sh (LIBERO 27 at seeds 7 · 8 · 9 ->
# SafeLIBERO 11 -> LIBERO-Safety 10; MIN27=0 = the safety benches always run: this GT-free version is not expected at 27/27).
# usage: V=<version> TAG=<tag> WAIT=<queue.log line to wait for> bash pe1k_eval_queue.sh
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
: "${V:?}" "${TAG:?}"
OUT=outputs/2026-11-09_집기놓기_장애물취급/eval_$TAG
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
if [ -n "${WAIT:-}" ]; then until grep -q "$WAIT" outputs/2026-11-08_PE_경로끝점_identity/closed_loop/queue.log; do sleep 30; done; fi
log "pe1k: evaluation protocol of $V"
LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $V --record-constraints full"
for s in 10 11; do
  run "$TAG calibration LIBERO9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 0,1,2,3,4,6,7,8,9 --episode-ids 0 \
    --seed $s --run-id gtto-$TAG-libero9s$s
done
log "calibration end"
V=$V TAG=$TAG OUT=$OUT MIN27=0 EXTRA="--record-constraints full" EXTRA_LS="--record-constraints full" bash experiments/segmentation/gtto16_queue.sh
log "queue k end"
