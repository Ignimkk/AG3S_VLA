#!/bin/bash
# T43W launcher: two chains (<= 2 of our servers at once), each arm = one server per list; resumable (.done files).
#   chain A (port 8282): V3 hurdle 14 -> V3 no-obstacle 5 -> V1 hurdle 14
#   chain B (port 8283): V4 hurdle 14 -> V4 no-obstacle 5 -> E0 hurdle 14
# (a later V5 arm gets its own launcher on the same ports, after these chains.)
# Start guard per server (run_chain.sh, shared lock): GPU used + 28 <= 130 GB, working set + 8 GiB <= 64 GiB, youngest serve_safe
# (anyone's) >= 120 s; 120 s after our own server start before the first client; working set + 4 GiB before every client.
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/p/env.sh
R=$V/runs; mkdir -p $R
H="$(cat $P/lists/hurdle.txt)"; B="$(cat $P/lists/clear.txt)"
chainA() { EST_GB=28 $P/run_chain.sh V3 8282 $R/V3 "$H"; echo "A V3H rc=$? $(date -Iseconds)"; sleep 5
           EST_GB=28 $P/run_chain.sh V3 8282 $R/V3 "$B"; echo "A V3B rc=$? $(date -Iseconds)"; sleep 5
           EST_GB=28 $P/run_chain.sh V1 8282 $R/V1 "$H"; echo "A V1H rc=$? $(date -Iseconds)"; echo "A end $(date -Iseconds)"; }
chainB() { sleep 30
           EST_GB=28 $P/run_chain.sh V4 8283 $R/V4 "$H"; echo "B V4H rc=$? $(date -Iseconds)"; sleep 5
           EST_GB=28 $P/run_chain.sh V4 8283 $R/V4 "$B"; echo "B V4B rc=$? $(date -Iseconds)"; sleep 5
           EST_GB=28 $P/run_chain.sh E0 8283 $R/E0 "$H"; echo "B E0H rc=$? $(date -Iseconds)"; echo "B end $(date -Iseconds)"; }
echo "launch $(date -Iseconds)"
chainA > $R/chainA.out 2>&1 &
chainB > $R/chainB.out 2>&1 &
wait
echo "all end $(date -Iseconds)"
