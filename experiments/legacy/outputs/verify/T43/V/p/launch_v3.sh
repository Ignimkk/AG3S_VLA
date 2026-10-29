#!/bin/bash
# T43V V3 launcher (lead 2026-10-05: V3 = V2 without --gripper-cover): hurdle 4 ep x 2 seeds split over two chains (<= 2 of our servers),
# same guards as launch_v.sh / run_chain.sh; resumable (.done).
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V/p/env.sh
R=$V/runs; mkdir -p $R
L=($(cat $P/lists/hurdle.txt)); LA="${L[0]} ${L[1]} ${L[2]} ${L[3]}"; LB="${L[4]} ${L[5]} ${L[6]} ${L[7]}"
echo "launch V3 $(date -Iseconds)"
( EST_GB=28 $P/run_chain.sh V3 8280 $R/V3 "$LA"; echo "A V3 rc=$? $(date -Iseconds)" ) > $R/chainA_v3.out 2>&1 &
( sleep 30; EST_GB=28 $P/run_chain.sh V3 8281 $R/V3 "$LB"; echo "B V3 rc=$? $(date -Iseconds)" ) > $R/chainB_v3.out 2>&1 &
wait
echo "all end V3 $(date -Iseconds)"
