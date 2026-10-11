#!/bin/bash
# SUBTASK-f V1: gate on (E3bG), 48 runs, 4 chains on ports 8231-8234, staggered 240 s (first-request JIT before next GPU check).
# Resumable (NAME.done skipped). usage: nohup setsid launch_v1.sh [N_CHAINS=4] > runs/launch.out 2>&1 &
source ${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-f/p/env.sh
N=${1:-4}; R=$SD_OUT/runs; mkdir -p $R
echo "v1 start $(date -Iseconds) chains=$N"
pids=""
if [ "$N" = 4 ]; then
  for i in 1 2 3 4; do EST_GB=22 $P/run_chain.sh E3bG $((8230 + i)) $R/E3bG "$(cat $P/lists/E3bG_$i.txt)" & pids="$pids $!"; [ $i -lt 4 ] && sleep 240; done
else
  EST_GB=22 $P/run_chain.sh E3bG 8231 $R/E3bG "$(cat $P/lists/E3bG_1.txt) $(cat $P/lists/E3bG_3.txt)" & pids="$pids $!"; sleep 240
  EST_GB=22 $P/run_chain.sh E3bG 8232 $R/E3bG "$(cat $P/lists/E3bG_2.txt) $(cat $P/lists/E3bG_4.txt)" & pids="$pids $!"
fi
for p in $pids; do wait $p; echo "chain pid=$p rc=$? $(date -Iseconds)"; done
echo "v1 end $(date -Iseconds)"
