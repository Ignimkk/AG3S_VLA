#!/usr/bin/env bash
# LIBERO-Safety VLA-only rollouts with geometry capture (level 0, dev init 0-4), exclusive GPU lock via gpu_lock.py (the launcher refuses a shared holder).
# usage: run_safety_vla_rollouts.sh PREFIX "TASK_IDS"
set -u
PREFIX=$1; TASKS=$2
cd "$(dirname "$0")/../.."
source scripts/setup/env.sh >/dev/null 2>&1
for suite in affordance human_safety obstacle_avoidance; do
  echo "start $suite $(date -u +%H:%M:%S)"
  python3 scripts/setup/gpu_lock.py exclusive --est-gib 25 --tag "safety-vla-$suite" -- \
    .venv-ag3s/bin/python benchmark/runners/launch_libero_safety_paired.py --task-suite-name $suite --level 0 \
    --task-ids "$TASKS" --episode-ids 0,1,2,3,4 --run-id "$PREFIX-$suite" --policy-checkpoint pi05_libero --conversion B \
    --modes vla_only --capture-segmentation 2>&1 | tail -3
  echo "done $suite rc=${PIPESTATUS[0]} $(date -u +%H:%M:%S)"
done
echo ALL-DONE
