#!/bin/bash
# T43W blind round launcher (snapshot ${AG3S_PROJECT_ROOT}-t43b = benchmark c265623 + pi05_TO_hybrid f62ab81). Starts after the V5 rerun
# ('all end V5' in runs/launch_v5.out). Two chains, <= 2 of our servers; same guards (pb/run_chain.sh, esdf_stats check for V arms); resumable.
#   chain G (port 8282): V5B (V5 + client --obstacle-policy-hidden) 14
#   chain H (port 8283): V5S (V5 visible, same snapshot) 14
# (lead 2026-10-06: blind round shrunk to V5-blind vs V5-visible; E0B / V4B are not run — the pb scripts still accept them)
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/pb/env.sh
H="$(cat $P/lists/hurdle.txt)"; mkdir -p $RB
echo "launch blind $(date -Iseconds)"
until grep -q '^all end V5' $V/runs/launch_v5.out 2>/dev/null; do sleep 30; done
echo "V5 rerun ended, start $(date -Iseconds)"
( EST_GB=28 $P/run_chain.sh V5B 8282 $RB/V5B "$H"; echo "G V5B rc=$? $(date -Iseconds)" ) > $RB/chainG.out 2>&1 &
( sleep 40; EST_GB=28 $P/run_chain.sh V5S 8283 $RB/V5S "$H"; echo "H V5S rc=$? $(date -Iseconds)" ) > $RB/chainH.out 2>&1 &
wait
echo "all end blind $(date -Iseconds)"
