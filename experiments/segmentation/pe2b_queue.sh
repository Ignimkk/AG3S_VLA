#!/usr/bin/env bash
# SL-PE11 alone (dev-4a, 2026-11-09, G170): V = ...-pe12345811 or ...-pe12345811-tro (chosen from pe2a part 1).
# (WAIT_PE2A=1: wait for 'queue pe2a end' first.) Calibration LIBERO s10 · s11 -> evaluation protocol (pe1k_eval_queue.sh) -> E1a 40 on
# init 5-14 (= the DEVELOPMENT split since G171; headline numbers come from init 15-49, e1a_headline_queue.sh).
# usage: V=<version> TAG=<run-id tag without hyphen, e.g. pe12345811tro> bash experiments/segmentation/pe2b_queue.sh
#   On a new server use a new TAG (e.g. pe12345811n2): the comparison scripts take the latest run per run-id.
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
# pe2a finished 2026-11-09 15:36 on the old server; wait for it only when asked (a new server has no pe2a log -> would hang)
if [ "${WAIT_PE2A:-0}" = 1 ]; then
  log "pe2b: waiting for 'queue pe2a end'"
  until grep -qs "queue pe2a end" outputs/2026-11-09_집기놓기_장애물취급/eval_pe123458tro/queue.log; do sleep 30; done
fi
log "pe2b: HEAD $(git rev-parse --short HEAD) V=$V"
V=$V TAG=$TAG bash experiments/segmentation/pe1k_eval_queue.sh
E1A="--task-suite-name safelibero_spatial --safety-level I --episode-ids 5,6,7,8,9,10,11,12,13,14 --seed 7 --data-split test_SafeLIBERO_Spatial_ep5-49_user_2026-10-14 --safelibero-step-budget main_aegis"
LPB="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --record-constraints full"
for t in 0 1 2 3; do
  run "$TAG E1a(dev 5-14) spatial LI task $t" $PY $LPB --ag3s-version $V $E1A --task-ids $t --run-id qe-$TAG-e1a-spatial-LI-t$t
done
log "queue pe2b end"
