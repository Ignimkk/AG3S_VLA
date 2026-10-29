#!/bin/bash
# T43V launcher: two chains (at most 2 of our servers at once), each arm = one server; resumable (.done files).
#   chain A (port 8280): V2 hurdle 8  -> V0 hurdle 8
#   chain B (port 8281): V1 hurdle 8  -> E0 hurdle 8 -> V2 no-obstacle 5
# Start guard per server (run_chain.sh, under a shared lock): GPU used + 28 <= 130 GB, working set + 8 GiB <= 64 GiB,
# youngest serve_safe (anyone's) >= 120 s; plus 120 s after our own server start before the first client; working set + 4 GiB before every client.
# usage: setsid nohup launch_v.sh > $V/runs/launch.out 2>&1 &
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V/p/env.sh
R=$V/runs; mkdir -p $R
H="$(cat $P/lists/hurdle.txt)"; B="$(cat $P/lists/clear.txt)"
chainA() { for a in V2 V0; do EST_GB=28 $P/run_chain.sh $a 8280 $R/$a "$H"; echo "A $a rc=$? $(date -Iseconds)"; sleep 5; done; echo "A end $(date -Iseconds)"; }
chainB() { sleep 30; for a in V1 E0; do EST_GB=28 $P/run_chain.sh $a 8281 $R/$a "$H"; echo "B $a rc=$? $(date -Iseconds)"; sleep 5; done
           EST_GB=28 $P/run_chain.sh V2 8281 $R/V2 "$B"; echo "B V2clear rc=$? $(date -Iseconds)"; echo "B end $(date -Iseconds)"; }
echo "launch $(date -Iseconds)"
chainA > $R/chainA.out 2>&1 &
chainB > $R/chainB.out 2>&1 &
wait
echo "all end $(date -Iseconds)"
