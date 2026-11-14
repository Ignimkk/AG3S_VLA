#!/usr/bin/env bash
# Paper Tab. 2 (LIBERO-Safety 1/10) cell by cell with a wall-clock cutoff (sdm-f6 SL-AUD session, 2026-11-15; plan §9.42).
# Same launcher arguments, cell order and run-id as paper_tab2_tenth_queue.sh (arms in $ARMS order; per arm L2 -> L1 -> L0,
# suites affordance, obstacle_avoidance, obstacle_avoidance_human, human_safety; task 0-4, init 19; run-id tab2sdm-<arm>).
# Additions:
#  - a cell is started only if  now + estimated duration <= CUTOFF  (epoch seconds; empty = no cutoff). The estimate is the
#    longest measured duration of that arm's cells so far, else EST_<arm> (s). The GPU must be free at the cutoff
#    (RB-Y1 real-robot test 13:30-15:15, plan §9.42).
#  - finished cells (rc 0 and a complete episodes.jsonl with 5 rows) are appended to $OUT/cells_done.txt; a rerun skips them,
#    so the queue resumes where it stopped. A cell interrupted mid-way has no complete run and is redone from its start.
# usage: VLS=<-lsnative version> ARMS="pi05 ag3s" [CUTOFF=$(date -d '2026-11-15 13:30' +%s)] bash paper_tab2_cells.sh
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv-ag3s/bin/python
: "${ARMS:?}"
VLS=${VLS:-}
CUTOFF=${CUTOFF:-}
EST_pi05=${EST_pi05:-900}
EST_ag3s=${EST_ag3s:-1800}
OUT=outputs/2026-11-13_paper_tab2_1-10
mkdir -p "$OUT"
LOG=$OUT/queue.log
DONE=$OUT/cells_done.txt
touch "$DONE"
B=/mnt/dev/AG3S_VLA-storage/results/benchmark
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
LS="benchmark/runners/launch_libero_safety_paired.py --conversion C --seed 7 --data-split test --violation-termination official --policy-checkpoint pi05_libero_safety"
log "paper Tab. 2 cells: HEAD $(git rev-parse --short HEAD) arms='$ARMS' VLS=$VLS cutoff=${CUTOFF:+$(date -d @$CUTOFF +%H:%M)}"
for arm in $ARMS; do
  if [ "$arm" = ag3s ] && [ -z "$VLS" ]; then log "tab2 cells: VLS not set -- STOP before the ag3s arm"; exit 1; fi
  longest=0
  for lvl in 2 1 0; do
    for suite in affordance obstacle_avoidance obstacle_avoidance_human human_safety; do
      key="$arm $suite L$lvl${VLS:+ $([ "$arm" = ag3s ] && echo "$VLS")}"
      if grep -qxF "$key" "$DONE"; then continue; fi
      est_var="EST_$arm"; est=${!est_var}; [ "$longest" -gt "$est" ] && est=$longest
      if [ -n "$CUTOFF" ] && [ $(( $(date +%s) + est )) -gt "$CUTOFF" ]; then
        log "tab2 cells: CUTOFF -- next cell '$arm $suite L$lvl' (est ${est}s) would end after $(date -d @$CUTOFF +%H:%M); stopped, GPU free"
        exit 3
      fi
      if [ "$arm" = pi05 ]; then margs="--modes vla_only"; else margs="--modes vla_v11ab --record-constraints full --ag3s-version $VLS"; fi
      t0=$(date +%s)
      log "start $arm LIBERO-Safety $suite L$lvl task 0-4 init 19"
      $PY $LS $margs --task-suite-name $suite --level $lvl --task-ids 0,1,2,3,4 --episode-ids 19 --run-id "tab2sdm-$arm" >> "$LOG.out" 2>&1
      rc=$?; dt=$(( $(date +%s) - t0 )); [ "$dt" -gt "$longest" ] && longest=$dt
      d=$(ls -d $B/2*_tab2sdm-$arm-* 2>/dev/null | tail -1)
      n=$([ -n "$d" ] && [ -f "$d/episodes.jsonl" ] && wc -l < "$d/episodes.jsonl" || echo 0)
      log "done rc=$rc (${dt}s, $n ep, $(basename "${d:-none}" | cut -c1-40))"
      if [ "$rc" -eq 0 ] && [ "$n" -eq 5 ]; then echo "$key" >> "$DONE"; else log "tab2 cells: cell '$arm $suite L$lvl' incomplete -- STOP"; exit 1; fi
    done
  done
done
log "paper Tab. 2 cells end"
