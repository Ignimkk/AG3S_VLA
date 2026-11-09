#!/usr/bin/env bash
# Follow-up to bl_tranche_queue.sh T=t1 (dev-4a, 2026-11-09 20:0x KST): rerun the tranche-1 runs that started before a
# user-approved port fidelity fix. Waits for 'queue baseline t1 end'.
#   * AEGIS privileged-names arm, safelibero_spatial Level I (started 18:23, before the prefer_list naming fix applied
#     ~20:0x). The first run stays on disk as the pre-fix record; run-id gets suffix -rerun.
#   (KNOWS F1 was applied at 18:57, before the first KNOWS run -> nothing to rerun.)
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-09_baseline_tranche/t1
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
until grep -qs "queue baseline t1 end" "$LOG"; do sleep 60; done
split=$($PY -c "from benchmark.runners import suite_registry as r; print(r.split_label('safelibero_spatial', 'I', 'test'))")
run "t1 fix: aegis-gt safelibero_spatial LI (prefer_list naming)" $PY Baseline/paper_baselines/closed_loop/launch_method.py \
  --method aegis --gt --task-suite-name safelibero_spatial --safety-level I --task-ids 0,1,2,3 --episode-ids 15,16,17,18 \
  --seed 7 --data-split "$split" --safelibero-step-budget main_aegis --run-id blt1-aegis-gt-spatial-LI-rerun
log "queue baseline t1 fix end"
