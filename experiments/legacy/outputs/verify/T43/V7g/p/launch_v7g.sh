#!/bin/bash
# T43V7g launcher (snapshot ${AG3S_PROJECT_ROOT}-t43v7 = benchmark f2f335b). Waits until the Z3V (V5D) chain's serve_safe is gone (read-only pgrep), then two chains (ports 8286 / 8287):
#   chain E (8286): V7g hurdle items 1-7 -> V7g no-obstacle 5
#   chain F (8287): V7g hurdle items 8-14
# Same guards as p/run_chain.sh; resumable (.done). usage: nohup setsid launch_v7g.sh &
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V7g/p/env.sh
R=$V/runs; L=($(cat $P/lists/hurdle.txt)); B="$(cat $P/lists/clear.txt)"
LA="${L[*]:0:7}"; LB="${L[*]:7:7}"
echo "launch wait V7g $(date -Iseconds)"
while pgrep -f "serve_safe.*outputs/verify/T43/Z3V" > /dev/null; do sleep 30; done
echo "Z3V serve_safe gone $(date -Iseconds); Z3V launcher tail: $(tail -qn 1 ${AG3S_PROJECT_ROOT}/outputs/verify/T43/Z3V/runs/*.out | tr '\n' '|')"
echo "launch V7g $(date -Iseconds)"
mkdir -p $R/V7g
( EST_GB=28 $P/run_chain.sh V7g 8286 $R/V7g "$LA"; echo "E V7gH rc=$? $(date -Iseconds)"; sleep 5
  EST_GB=28 $P/run_chain.sh V7g 8286 $R/V7g "$B"; echo "E V7gB rc=$? $(date -Iseconds)" ) > $R/chainE.out 2>&1 &
( sleep 240; EST_GB=28 $P/run_chain.sh V7g 8287 $R/V7g "$LB"; echo "F V7gH rc=$? $(date -Iseconds)" ) > $R/chainF.out 2>&1 &
wait
echo "all end V7g $(date -Iseconds)"
