#!/bin/bash
# T43Z3V launcher (snapshot ${AG3S_PROJECT_ROOT}-t43z3 = benchmark cf6b410). Two chains, <= 2 of our servers, ports 8284 / 8285:
#   chain E (port 8284): V5D hurdle items 1-7 -> V5D no-obstacle 5
#   chain F (port 8285): V5D hurdle items 8-14
# Same guards as p/run_chain.sh (incl. esdf_stats check after every run); resumable (.done). usage: nohup setsid launch_z3v.sh &
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/Z3V/p/env.sh
R=$V/runs; L=($(cat $P/lists/hurdle.txt)); B="$(cat $P/lists/clear.txt)"
LA="${L[*]:0:7}"; LB="${L[*]:7:7}"
echo "launch V5D $(date -Iseconds)"
( EST_GB=28 $P/run_chain.sh V5D 8284 $R/V5D "$LA"; echo "E V5DH rc=$? $(date -Iseconds)"; sleep 5
  EST_GB=28 $P/run_chain.sh V5D 8284 $R/V5D "$B"; echo "E V5DB rc=$? $(date -Iseconds)" ) > $R/chainE.out 2>&1 &
( sleep 30; EST_GB=28 $P/run_chain.sh V5D 8285 $R/V5D "$LB"; echo "F V5DH rc=$? $(date -Iseconds)" ) > $R/chainF.out 2>&1 &
wait
echo "all end V5D $(date -Iseconds)"
