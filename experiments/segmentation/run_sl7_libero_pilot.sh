#!/usr/bin/env bash
# SL7 LIBERO pilot (protocol doc/active/experiments/SL7_P0M_PANDA_3BENCH_PROTOCOL_20261103_KO.md §2-§4):
# libero_spatial task 0-9 minus 5 (VLA-only failure) x init 0, seed 7, conversion C, one AG3S version after another.
# The launcher takes the exclusive GPU flock itself (TrajOpt closed loop).
set -u
cd "$(dirname "$0")/../.."
source scripts/setup/env.sh >/dev/null 2>&1
for v in "$@"; do
  echo "start $v $(date -u +%H:%M:%S)"
  .venv-ag3s/bin/python benchmark/runners/launch_libero_paired.py --task-suite-name libero_spatial \
    --task-ids 0,1,2,3,4,6,7,8,9 --episode-ids 0 --run-id "sl7-libero9-$v" --modes vla_v11ab --ag3s-version "$v" \
    --conversion C 2>&1 | tail -3
  echo "done $v rc=${PIPESTATUS[0]} $(date -u +%H:%M:%S)"
done
echo ALL-DONE
