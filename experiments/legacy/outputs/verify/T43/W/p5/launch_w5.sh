#!/bin/bash
# T43W V5 rerun launcher (snapshot ${AG3S_PROJECT_ROOT}-t43w3 = benchmark 72ed927). Two chains, <= 2 of our servers:
#   chain E (port 8282): V5 hurdle items 1-7 -> V5 no-obstacle 5
#   chain F (port 8283): V5 hurdle items 8-14
# Same guards as p5/run_chain.sh (incl. the esdf_stats check after every run); resumable (.done).
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/p5/env.sh
R=$V/runs; L=($(cat $P/lists/hurdle.txt)); B="$(cat $P/lists/clear.txt)"
LA="${L[*]:0:7}"; LB="${L[*]:7:7}"
echo "launch V5 $(date -Iseconds)"
( EST_GB=28 $P/run_chain.sh V5 8282 $R/V5 "$LA"; echo "E V5H rc=$? $(date -Iseconds)"; sleep 5
  EST_GB=28 $P/run_chain.sh V5 8282 $R/V5 "$B"; echo "E V5B rc=$? $(date -Iseconds)" ) > $R/chainE.out 2>&1 &
( sleep 30; EST_GB=28 $P/run_chain.sh V5 8283 $R/V5 "$LB"; echo "F V5H rc=$? $(date -Iseconds)" ) > $R/chainF.out 2>&1 &
wait
echo "all end V5 $(date -Iseconds)"
