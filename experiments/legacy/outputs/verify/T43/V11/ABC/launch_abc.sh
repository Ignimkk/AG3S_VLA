#!/bin/bash
# T43V11ABC launcher (copy of p/launch_v11.sh with ARM=V11ABC, run_chain_abc.sh). Two chains (ports 8286 / 8287), 240 s stagger.
#   chain E (8286): hurdle items 1-7 -> no-obstacle 5     chain F (8287): hurdle items 8-14
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/p/env.sh
ABC=$V/ABC; ARM=V11ABC; CH=$ABC/run_chain_abc.sh
R=$V/runs/$ARM; L=($(cat $P/lists/hurdle.txt)); B="$(cat $P/lists/clear.txt)"
LA="${L[*]:0:7}"; LB="${L[*]:7:7}"
echo "launch wait $ARM $(date -Iseconds)"
echo "serve_safe procs now (read-only): $(pgrep -fc AG3S.trajopt.serve_safe)"
echo "launch $ARM $(date -Iseconds)"
mkdir -p $R/$ARM
( EST_GB=28 $CH $ARM 8286 $R "$LA"; echo "E ${ARM}H rc=$? $(date -Iseconds)"; sleep 5
  EST_GB=28 $CH $ARM 8286 $R "$B"; echo "E ${ARM}B rc=$? $(date -Iseconds)" ) > $V/runs/chainE_$ARM.out 2>&1 &
( sleep 240; EST_GB=28 $CH $ARM 8287 $R "$LB"; echo "F ${ARM}H rc=$? $(date -Iseconds)" ) > $V/runs/chainF_$ARM.out 2>&1 &
wait
echo "all end $ARM $(date -Iseconds)"
