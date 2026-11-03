#!/usr/bin/env bash
# SL7 SafeLIBERO closed loop (protocol doc/active/experiments/SL7_P0M_PANDA_3BENCH_PROTOCOL_20261103_KO.md §2):
# spatial Level I dev, the 11 VLA-only official-success episodes (t1e1, t1e4, t2e0-e4, t3e0, t3e2-e4), seed 7,
# conversion C; the same split / scenes as the SL6-H pc판 smoke. One AG3S version per argument; the launcher takes the
# exclusive GPU flock itself (TrajOpt closed loop).
set -u
cd "$(dirname "$0")/../.."
source scripts/setup/env.sh >/dev/null 2>&1
SPLIT=dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14
for v in "$@"; do
  for spec in "1:1,4" "2:0,1,2,3,4" "3:0,2,3,4"; do
    t=${spec%%:*}; eps=${spec#*:}
    echo "start $v t$t $(date -u +%H:%M:%S)"
    .venv-ag3s/bin/python -m benchmark.runners.launch_libero_paired --task-suite-name safelibero_spatial --safety-level I \
      --data-split "$SPLIT" --task-ids "$t" --episode-ids "$eps" --run-id "sl7-safelibero11-$v-t$t" \
      --conversion C --modes vla_v11ab --ag3s-version "$v" 2>&1 | tail -2
    echo "done $v t$t rc=${PIPESTATUS[0]} $(date -u +%H:%M:%S)"
  done
done
echo ALL-DONE
