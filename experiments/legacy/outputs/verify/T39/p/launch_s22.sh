#!/bin/bash
# T39 S2-2: 4 chains (E0 A/B on 8224/8225, E3b A/B on 8226/8227), staggered; each chain refuses to start if the GPU budget
# would be exceeded (then re-run this script later: done runs are skipped). usage: nohup launch_s22.sh [N_CHAINS] > runs/launch.out 2>&1 &
# N_CHAINS (default 4): 2 = one chain per condition (A lists then B lists sequentially).
source ${AG3S_PROJECT_ROOT}/outputs/verify/T39/p/env.sh
N=${1:-4}; R=$T39/runs; mkdir -p $R
echo "s22 start $(date -Iseconds) chains=$N"
if [ "$N" = 4 ]; then
  EST_GB=${EST_GB_E3:-22} $P/run_chain.sh E3b 8226 $R/E3b "$(cat $P/lists/E3b_A.txt)" & p1=$!; sleep 240
  EST_GB=${EST_GB_E3:-22} $P/run_chain.sh E3b 8227 $R/E3b "$(cat $P/lists/E3b_B.txt)" & p2=$!; sleep 240
  EST_GB=${EST_GB_E0:-10} $P/run_chain.sh E0 8224 $R/E0 "$(cat $P/lists/E0_A.txt)" & p3=$!; sleep 240
  EST_GB=${EST_GB_E0:-10} $P/run_chain.sh E0 8225 $R/E0 "$(cat $P/lists/E0_B.txt)" & p4=$!
  for p in $p1 $p2 $p3 $p4; do wait $p; echo "chain pid=$p rc=$? $(date -Iseconds)"; done
else
  ( EST_GB=${EST_GB_E3:-22} $P/run_chain.sh E3b 8226 $R/E3b "$(cat $P/lists/E3b_A.txt) $(cat $P/lists/E3b_B.txt)" ) & p1=$!; sleep 240
  ( EST_GB=${EST_GB_E0:-10} $P/run_chain.sh E0 8224 $R/E0 "$(cat $P/lists/E0_A.txt) $(cat $P/lists/E0_B.txt)" ) & p2=$!
  for p in $p1 $p2; do wait $p; echo "chain pid=$p rc=$? $(date -Iseconds)"; done
fi
echo "s22 end $(date -Iseconds)"
