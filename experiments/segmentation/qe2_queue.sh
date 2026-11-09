#!/usr/bin/env bash
# dev-38 SL-QE (2026-11-08 23:1x KST): the paired VLA-only control for the E1a B5 runs (qe1/qe1b). The E1a "π0.5 only" arm ran
# through Baseline/.../launch_method.py (another client), so its episodes are not paired with the B5 runs. This runs the same
# launcher, client, per-episode policy_seed and protocol as qe1b with `--modes vla_only` (--no-safe server): SafeLIBERO Spatial
# LI, task 0-3 x test init 5-14, seed 7, main_aegis budget. Separates "AG3S intervened" from "the policy / execution differed".
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-08_E1a_SafeLIBERO_B5
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
log "qe2: paired VLA-only control (same launcher / client / policy_seed)"
SL="--task-suite-name safelibero_spatial --safety-level I --episode-ids 5,6,7,8,9,10,11,12,13,14 --seed 7 --data-split test_SafeLIBERO_Spatial_ep5-49_user_2026-10-14 --safelibero-step-budget main_aegis"
for t in 0 1 2 3; do
  run "vla_only E1a spatial LI task $t" $PY benchmark/runners/launch_libero_paired.py --conversion C --modes vla_only $SL --task-ids $t --run-id qe-vla-e1a-spatial-LI-t$t
done
log "queue end (qe2)"
