#!/bin/bash
# T43 LB launcher. usage: nohup setsid launch_lb.sh TAG &   (TAG = s0_5, c2, c3, ... from plan_lb.py; lists/TAG_0.txt -> port 8280, TAG_1.txt -> port 8281)
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/LB/LB3/p/env.sh
TAG=$1; R=$V/runs; mkdir -p $R/E0
LA="$(cat $P/lists/${TAG}_0.txt)"; LB_="$(cat $P/lists/${TAG}_1.txt)"
echo "launch $TAG $(date -Iseconds) loadavg=$(cat /proc/loadavg)"
( EST_GB=28 $P/run_chain.sh E0 8280 $R/E0 "$LA"; echo "A $TAG rc=$? $(date -Iseconds)" ) > $R/chainA_$TAG.out 2>&1 &
( sleep 20; EST_GB=28 $P/run_chain.sh E0 8281 $R/E0 "$LB_"; echo "B $TAG rc=$? $(date -Iseconds)" ) > $R/chainB_$TAG.out 2>&1 &
wait
echo "all end $TAG $(date -Iseconds) loadavg=$(cat /proc/loadavg)"
