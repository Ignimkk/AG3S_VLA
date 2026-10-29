#!/bin/bash
# T43V8 launcher (snapshot ${AG3S_PROJECT_ROOT}-t43v8 = benchmark 786cf1b). No other GPU round of ours runs (lead); the chains' start guard staggers servers. Two chains (ports 8286 / 8287):
#   chain E (8286): V8 hurdle items 1-7 -> V8 no-obstacle 5
#   chain F (8287): V8 hurdle items 8-14
# Same guards as p/run_chain.sh; resumable (.done). usage: nohup setsid launch_v8.sh &
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V8/p/env.sh
R=$V/runs; L=($(cat $P/lists/hurdle.txt)); B="$(cat $P/lists/clear.txt)"
LA="${L[*]:0:7}"; LB="${L[*]:7:7}"
echo "launch wait V8 $(date -Iseconds)"
echo "serve_safe procs now (read-only): $(pgrep -fc AG3S.trajopt.serve_safe)"
echo "launch V8 $(date -Iseconds)"
mkdir -p $R/V8
( EST_GB=28 $P/run_chain.sh V8 8286 $R/V8 "$LA"; echo "E V8H rc=$? $(date -Iseconds)"; sleep 5
  EST_GB=28 $P/run_chain.sh V8 8286 $R/V8 "$B"; echo "E V8B rc=$? $(date -Iseconds)" ) > $R/chainE.out 2>&1 &
( sleep 240; EST_GB=28 $P/run_chain.sh V8 8287 $R/V8 "$LB"; echo "F V8H rc=$? $(date -Iseconds)" ) > $R/chainF.out 2>&1 &
wait
echo "all end V8 $(date -Iseconds)"
