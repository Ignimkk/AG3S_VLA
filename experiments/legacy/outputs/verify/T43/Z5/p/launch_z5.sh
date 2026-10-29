#!/bin/bash
# T43 Z5 launcher: E0 (--no-safe), no obstacle, 14 ep:seed of the P2 hurdle list, snapshot ${AG3S_PROJECT_ROOT}-t43w3.
# Two chains of 7 (<= 2 of our servers): port 8284 items 1-7, port 8285 items 8-14.
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/Z5/p/env.sh
R=$V/runs; L=($(cat $P/lists/e0.txt)); LA="${L[*]:0:7}"; LB="${L[*]:7:7}"
echo "launch Z5 $(date -Iseconds)"
( EST_GB=28 $P/run_chain.sh E0 8284 $R/E0 "$LA"; echo "A E0 rc=$? $(date -Iseconds)" ) > $R/chainA.out 2>&1 &
( sleep 30; EST_GB=28 $P/run_chain.sh E0 8285 $R/E0 "$LB"; echo "B E0 rc=$? $(date -Iseconds)" ) > $R/chainB.out 2>&1 &
wait
echo "all end Z5 $(date -Iseconds)"
