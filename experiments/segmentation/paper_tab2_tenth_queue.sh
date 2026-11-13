#!/usr/bin/env bash
# Paper Tab. 2 (LIBERO-Safety), this server's 1/10 (sdm-f6, 2026-11-13; user: 1/10 of the planned episodes, tables 1 -> 3).
# Plan: 4 suites x L0-L2 x task 0-4 x 10 trials = 600 ep / row; 1/10 = one trial (init 19) per (suite, level, task) = 60 ep / row.
# init 19 is the only init with measurements (L2, task 0-1: pi0.5 official 4 / 6 success); the other 9 trials go to the other
# environment (doc/active/experiments/PAPER_TENTH_SPLIT_20261113.json "libero_safety").
# Rows: pi05 = pi0.5 alone on the official checkpoint pi05_libero_safety (--modes vla_only, adapter none);
#       ag3s = $VLS (an -lsnative version: SL-LS1 controller conversion; the launcher sets the client adapter fork_native_unit).
# Termination on the official info['cost'] (B1: official as is; the client records the auxiliary contact columns).
# usage: VLS=<...-lsnative version> [ARMS="pi05 ag3s"] [DRY=1] bash experiments/segmentation/paper_tab2_tenth_queue.sh
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv-ag3s/bin/python
: "${VLS:?}"
ARMS=${ARMS:-"pi05 ag3s"}
DRY=${DRY:-0}
OUT=outputs/2026-11-13_paper_tab2_1-10
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() {
  local lab=$1; shift
  if [ "$DRY" = 1 ]; then "$@" --dry-run > /dev/null 2>> "$LOG.dry" && log "dry ok $lab" || log "dry FAILED $lab"; return; fi
  log "start $lab"; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"
}
LS="benchmark/runners/launch_libero_safety_paired.py --conversion C --seed 7 --data-split test --violation-termination official --policy-checkpoint pi05_libero_safety"
log "paper Tab. 2 1/10: HEAD $(git rev-parse --short HEAD) arms='$ARMS' VLS=$VLS"
for arm in $ARMS; do
  for lvl in 2 1 0; do
    for suite in affordance obstacle_avoidance obstacle_avoidance_human human_safety; do
      if [ "$arm" = pi05 ]; then margs="--modes vla_only"; else margs="--modes vla_v11ab --record-constraints full --ag3s-version $VLS"; fi
      run "$arm LIBERO-Safety $suite L$lvl task 0-4 init 19" $PY $LS $margs --task-suite-name $suite --level $lvl --task-ids 0,1,2,3,4 \
        --episode-ids 19 --run-id "tab2sdm-$arm"
    done
  done
done
log "paper Tab. 2 1/10 queue end"
