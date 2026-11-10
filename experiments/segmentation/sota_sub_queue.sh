#!/usr/bin/env bash
# SOTA mechanism check on an E1a episode subset (dev-57, 2026-11-11, AG3S_PLAN_SCHEDULE_20261110_KO.md §9): same-machine,
# same-seed reruns reproduce (pe123458 vs pe12345811: 31 / 40 episodes identical to the step), so a flag's effect can be
# read on the episodes it targets before a full E1a 40. NOT an adoption run (adoption = sota_queue.sh STAGE=e1a).
# usage: V=<AG3S_VERSIONS key> TAG=<tag without hyphen> SUB="0:7,10,11 3:12" bash experiments/segmentation/sota_sub_queue.sh
#   SUB = space-separated <task>:<comma-separated init ids> (development split init 5-14 only)
# run-id sub-$TAG-e1a-spatial-LI-t<k>. The launchers take the GPU lock themselves. Do not edit while it runs.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv-ag3s/bin/python
: "${V:?}" "${TAG:?}" "${SUB:?}"
OUT=outputs/2026-11-11_SOTA_개선/eval_$TAG
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
E1A="--task-suite-name safelibero_spatial --safety-level I --seed 7 --data-split test_SafeLIBERO_Spatial_ep5-49_user_2026-10-14 --safelibero-step-budget main_aegis"
LPB="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --record-constraints full"
log "sota sub: HEAD $(git rev-parse --short HEAD) V=$V SUB=[$SUB]"
for cell in $SUB; do
  t=${cell%%:*}; ids=${cell#*:}
  for i in ${ids//,/ }; do
    if [ "$i" -lt 5 ] || [ "$i" -gt 14 ]; then log "init $i is outside the development split 5-14"; exit 2; fi
  done
  run "$TAG E1a subset spatial LI task $t init $ids" $PY $LPB --ag3s-version "$V" $E1A --task-ids $t --episode-ids $ids --run-id "sub-$TAG-e1a-spatial-LI-t$t"
done
log "queue sota sub end ($TAG)"
