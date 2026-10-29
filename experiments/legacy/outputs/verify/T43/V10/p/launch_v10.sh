#!/bin/bash
# T43V10 launcher (snapshot ${AG3S_PROJECT_ROOT}-t43v10 = benchmark 308ddfd). No other GPU round of ours runs (lead); the chains' start guard staggers servers. Two chains (ports 8286 / 8287):
#   chain E (8286): V10 hurdle items 1-7 -> V10 no-obstacle 5
#   chain F (8287): V10 hurdle items 8-14
# Same guards as p/run_chain.sh; resumable (.done). usage: nohup setsid launch_v10.sh &
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V10/p/env.sh
R=$V/runs; L=($(cat $P/lists/hurdle.txt)); B="$(cat $P/lists/clear.txt)"
LA="${L[*]:0:7}"; LB="${L[*]:7:7}"
echo "launch wait V10 $(date -Iseconds)"
echo "serve_safe procs now (read-only): $(pgrep -fc AG3S.trajopt.serve_safe)"
echo "launch V10 $(date -Iseconds)"
mkdir -p $R/V10
( EST_GB=28 $P/run_chain.sh V10 8286 $R/V10 "$LA"; echo "E V10H rc=$? $(date -Iseconds)"; sleep 5
  EST_GB=28 $P/run_chain.sh V10 8286 $R/V10 "$B"; echo "E V10B rc=$? $(date -Iseconds)" ) > $R/chainE.out 2>&1 &
( sleep 240; EST_GB=28 $P/run_chain.sh V10 8287 $R/V10 "$LB"; echo "F V10H rc=$? $(date -Iseconds)" ) > $R/chainF.out 2>&1 &
wait
echo "all end V10 $(date -Iseconds)"
