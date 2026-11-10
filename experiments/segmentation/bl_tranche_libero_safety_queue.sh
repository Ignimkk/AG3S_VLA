#!/usr/bin/env bash
# Baseline tranche C: LIBERO-Safety, pi0.5 alone (official pi05_libero_safety weight), 1/10 of plan E1b (2026-11-10 KST).
# Plan: paper/evidence/AG3S-VLA_CVPR2027_FINAL_EXPERIMENT_PLAN_20261108_KO.md §1.3 / §3.3 / §10.2 (physical 4 suite x L0-L2
# x task 0-4 x 10 trial = 600 ep/arm). Tranche 1 = 1 trial per task = 60 ep/arm. Pre-check and choices:
# outputs/2026-11-10_LIBERO-Safety_사전판정/README.md (H1 / H2 / B1 verdicts, init choice, time estimate).
#
# Fixed protocol (pre-registered, not chosen from results):
#   checkpoint pi05_libero_safety (HF @e66d84bf, openpi TrainConfig pi05_libero_safety, norm stats assets/lerobot),
#   action adapter none (actions go unchanged into the fork OSC: kp 750, output +-2, input clip +-1 = the controller the
#   official dataset / weight were made with), image rot180, replan 5, wait 10, max_steps 600 (launcher defaults),
#   termination on the official info['cost'] (B1: official as is; auxiliary geom-id robot contact + per-constraint
#   columns are recorded by the client and reported separately), seed 7 (episode seed = 7*100000 + task*1000 + init),
#   init states INITS (default 15 = first index of the reporting split 15-49; full plan = init 15-24 = 10 trials).
# Order: L2 first (main-text cells), suites AAG, TSA, FSHOA, HRI (HRI is supplement only).
# The launcher (benchmark/runners/launch_libero_safety_paired.py) takes /tmp/ag3s-vla-gpu-validation.lock itself:
# NO outer flock here (outer flock + launcher flock = self-deadlock, 2026-11-07 G53).
# A block is skipped when a run dir with the same run-id has client_returncode 0 and the expected episode count.
# run-id: bl<T>-none-ls[SUFFIX] -> launcher appends -<suite>-L<lvl>-pi05_libero_safety, run dir ends in -vla_only.
#
# usage:  [INITS=15] [T=t1] [SUFFIX=] [SUITES="affordance ..."] [LEVELS="2 1 0"] [DRY=1] \
#         bash experiments/segmentation/bl_tranche_libero_safety_queue.sh
#   DRY=1: launcher --dry-run for every block (CPU only, no lock, no server) -- pre-flight.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
INITS=${INITS:-15}
T=${T:-t1}
SUFFIX=${SUFFIX:-}
DRY=${DRY:-0}
TASKS=0,1,2,3,4
CKPT=pi05_libero_safety
ARM=none
OUT=outputs/2026-11-09_baseline_tranche/$T
mkdir -p "$OUT"
LOG=$OUT/queue.log
STORE=/mnt/dev/AG3S_VLA-storage/results/benchmark
LAUNCH=benchmark/runners/launch_libero_safety_paired.py
n_inits=$(awk -F, '{print NF}' <<< "$INITS")
expected=$((5 * n_inits))
log() { echo "$(date +%Y-%m-%dT%H:%M:%S%z) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
done_run() {  # $1 = full run-id (without -vla_only) -> 0 if a finished run with the expected episode count exists
  local d
  for d in "$STORE"/*_"$1"-vla_only; do
    [ -e "$d/episodes.jsonl" ] || continue
    [[ "$(basename "$d")" == INVALID* ]] && continue
    [ "$(grep -c . "$d/episodes.jsonl")" -ge "$expected" ] || continue
    $PY -c "import json,sys; m=json.load(open(sys.argv[1])); r=[e['settings'] for e in m['executions'] if e['settings'].get('role')=='launcher_result']; sys.exit(0 if r and r[-1].get('client_returncode')==0 else 1)" "$d/run_manifest.json" && return 0
  done
  return 1
}
log "baseline tranche $T LIBERO-Safety queue (C): HEAD $(git rev-parse --short HEAD) arm=$ARM ckpt=$CKPT inits=$INITS tasks=$TASKS suffix='$SUFFIX' dry=$DRY"
for lvl in ${LEVELS-2 1 0}; do
  for suite in ${SUITES-affordance obstacle_avoidance obstacle_avoidance_human human_safety}; do
    rid=bl$T-$ARM-ls$SUFFIX
    full=$rid-$suite-L$lvl-$CKPT
    if [ "$DRY" != 1 ] && done_run "$full"; then log "skip $full (done)"; continue; fi
    argv=($PY $LAUNCH --task-suite-name "$suite" --level "$lvl" --task-ids $TASKS --episode-ids "$INITS" --seed 7
          --data-split test --run-id "$rid" --modes vla_only --policy-checkpoint $CKPT --conversion C
          --violation-termination official)
    if [ "$DRY" = 1 ]; then
      "${argv[@]}" --dry-run > /dev/null 2>> "$LOG.dry" && log "dry ok $full" || log "dry FAILED $full (see $LOG.dry)"
    else
      run "$T $ARM ls $suite L$lvl init $INITS" "${argv[@]}"
    fi
  done
done
log "queue baseline $T LIBERO-Safety (C) end"
