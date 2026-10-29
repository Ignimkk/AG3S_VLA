#!/bin/bash
# T43GD launcher (snapshot ${AG3S_PROJECT_ROOT}-t43v7). Waits until the V7g round is over (launch.out has 'all end V7g' AND no serve_safe from outputs/verify/T43/V7g; read-only), then two chains
# (ports 8284 / 8285, arm V7 = V7g server flags + --no-subtask-gate):
#   chain E (8284): hurdle items 1-7 + clear items 1-3
#   chain F (8285): hurdle items 8-14 + clear items 4-5   (started 240 s later)
# Same guards as p/run_chain.sh (V7g's, plus the gate-OFF check); resumable (.done). usage: nohup setsid launch_gd.sh &
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/GD/p/env.sh
R=$V/runs; L=($(cat $P/lists/hurdle.txt)); B=($(cat $P/lists/clear.txt))
LA="${L[*]:0:7} ${B[*]:0:3}"; LB="${L[*]:7:7} ${B[*]:3:2}"
echo "launch wait GD $(date -Iseconds)"
while ! grep -q "all end V7g" ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V7g/runs/launch.out || pgrep -f "serve_safe.*outputs/verify/T43/V7g" > /dev/null; do sleep 30; done
echo "V7g over $(date -Iseconds); V7g launch tail: $(tail -qn 1 ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V7g/runs/launch.out)"
echo "launch V7 $(date -Iseconds)"
mkdir -p $R/V7
( EST_GB=28 $P/run_chain.sh V7 8284 $R/V7 "$LA"; echo "E V7 rc=$? $(date -Iseconds)" ) > $R/chainE.out 2>&1 &
( sleep 240; EST_GB=28 $P/run_chain.sh V7 8285 $R/V7 "$LB"; echo "F V7 rc=$? $(date -Iseconds)" ) > $R/chainF.out 2>&1 &
wait
echo "all end V7 $(date -Iseconds)"
