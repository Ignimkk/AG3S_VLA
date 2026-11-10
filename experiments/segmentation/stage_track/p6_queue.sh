#!/usr/bin/env bash
# P6 (제약 단계 추종, 2026-11-10): one version on the reduced evaluation protocol (48 ep) — evaluation LIBERO 27 (seed 7 · 8 · 9),
# SafeLIBERO 11 (Spatial LI dev, seed 7), LIBERO-Safety 10 (L0, seed 7); same sets and flags as gtto16_queue.sh, without the
# LIBERO-27 gate (all three always run) and without calibration / E1a (plan doc/active/experiments/AG3S_PLAN_SCHEDULE_20261110_KO.md §6).
# Runs from THIS checkout (the stage-track worktree); every launcher takes the exclusive GPU lock per run (no outer lock).
# usage: V=<version> TAG=<run tag> bash experiments/segmentation/stage_track/p6_queue.sh
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export AG3S_STORAGE_ROOT=${AG3S_STORAGE_ROOT:-/mnt/dev/AG3S_VLA-storage}
source "$W/scripts/setup/env.sh" >/dev/null 2>&1
cd "$W"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$W/.runtime/magickwand/lib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv-ag3s/bin/python
: "${V:?}" "${TAG:?}"
OUT=/mnt/dev/AG3S_VLA/outputs/2026-11-10_제약단계_추종/p6/$TAG
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
log "P6 queue $TAG: checkout $W HEAD $(git -C "$W" rev-parse --short HEAD) V=$V storage=$AG3S_STORAGE_ROOT"
LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $V"
for s in 7 8 9; do
  run "$TAG LIBERO9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 0,1,2,3,4,6,7,8,9 --episode-ids 0 --seed $s --run-id gtto-$TAG-libero9s$s
done
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --seed 7"
run "$TAG SafeLIBERO t1" $PY $LP $SL --task-ids 1 --episode-ids 1,4 --run-id gtto-$TAG-safelibero11-t1
run "$TAG SafeLIBERO t2" $PY $LP $SL --task-ids 2 --episode-ids 0,1,2,3,4 --run-id gtto-$TAG-safelibero11-t2
run "$TAG SafeLIBERO t3" $PY $LP $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-$TAG-safelibero11-t3
LS="benchmark/runners/launch_libero_safety_paired.py --level 0 --task-ids 0 --policy-checkpoint pi05_libero --conversion C --seed 7 --modes vla_v11ab --ag3s-version $V"
run "$TAG LS human_safety" $PY $LS --task-suite-name human_safety --episode-ids 0,1,2,3 --run-id gtto-$TAG-ls10
run "$TAG LS obstacle_avoidance" $PY $LS --task-suite-name obstacle_avoidance --episode-ids 0,1,2,3 --run-id gtto-$TAG-ls10
run "$TAG LS affordance" $PY $LS --task-suite-name affordance --episode-ids 0,2 --run-id gtto-$TAG-ls10
log "queue end"
