#!/usr/bin/env bash
# nb1104: attention audit (kv_L4 feature) of the re-made SL4b-protocol rollout, for the re-made HSR6-Panda probe (C6).
# Runs inside nb1104_queue.sh (inherits the GPU lock). Waits until the rollout launcher has written 40 episodes and exited.
set -u
cd "$(dirname "$0")/../.."
source scripts/setup/env.sh >/dev/null 2>&1
DEV=$(ls -d "$AG3S_STORAGE_ROOT"/results/benchmark/*_nb1104-probe-dev-pi05only-t0-9e1-4 | tail -1)
until [ "$(grep -c . "$DEV/episodes.jsonl" 2>/dev/null)" = 40 ] && ! pgrep -f 'launch_libero_pi05_only.py.*nb1104-probe' >/dev/null; do sleep 20; done
echo "source run $DEV complete $(date +%T)"
.venv-openpi-live/bin/python benchmark/runners/attention_audit.py --source-run "$(basename "$DEV")" --run-id nb1104-probe-attention-audit-dev-e1-4
