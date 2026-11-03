#!/usr/bin/env bash
# Closed-loop simple CBF-QP safety filters on SafeLIBERO spatial Level I dev (task 0-3 x init 0-4, seed 7), one arm after another.
# Each arm takes an EXCLUSIVE GPU lock through gpu_lock.py: launch_method.py only accepts a caller-held exclusive flock (a shared holder is refused, 2026-11-02 18:15 UTC).
# usage: run_cbf_arms_safelibero.sh PREFIX ARM[:gt] [ARM[:gt] ...]
set -u
PREFIX=$1; shift
cd "$(dirname "$0")/../.."
source scripts/setup/env.sh >/dev/null 2>&1
REF=20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only
for spec in "$@"; do
  arm=${spec%%:*}; gt=""; tag="$arm"
  if [ "$spec" != "$arm" ]; then gt="--gt"; tag="$arm-gt"; fi
  rid="$PREFIX-$tag"
  echo "start $rid $(date -u +%H:%M:%S)"
  python3 scripts/setup/gpu_lock.py exclusive --est-gib 30 --tag "cbf-$tag" -- \
    .venv-ag3s/bin/python Baseline/paper_baselines/closed_loop/launch_method.py --method "$arm" $gt \
    --safety-level I --task-ids 0,1,2,3 --episode-ids 0,1,2,3,4 --run-id "$rid" --sidecar-device cuda --ref-run "$REF" 2>&1 | tail -4
  echo "done $rid rc=${PIPESTATUS[0]} $(date -u +%H:%M:%S)"
done
echo ALL-DONE
