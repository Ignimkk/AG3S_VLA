#!/usr/bin/env bash
# SL7 LIBERO-Safety VLA-only collection (protocol doc/active/experiments/SL7_P0M_PANDA_3BENCH_PROTOCOL_20261103_KO.md
# §1 / §5-5): affordance · human_safety · obstacle_avoidance, level 0, task 0-1 x init 0-4 (dev split) = 30 episodes,
# seed 7, policy pi05_libero + libero_to_safety action adapter (the launcher default for that checkpoint; NOT official
# comparable -- pi05_libero_safety has no openpi TrainConfig, hook H1). The launcher takes the exclusive GPU flock itself.
set -u
cd "$(dirname "$0")/../.."
source scripts/setup/env.sh >/dev/null 2>&1
for suite in affordance human_safety obstacle_avoidance; do
  echo "start $suite $(date -u +%H:%M:%S)"
  .venv-ag3s/bin/python benchmark/runners/launch_libero_safety_paired.py --task-suite-name "$suite" --level 0 \
    --task-ids 0,1 --episode-ids 0,1,2,3,4 --run-id "sl7-liberosafety-vla30-$suite" --policy-checkpoint pi05_libero \
    --conversion C --modes vla_only "$@" 2>&1 | tail -2
  echo "done $suite rc=${PIPESTATUS[0]} $(date -u +%H:%M:%S)"
done
echo ALL-DONE
