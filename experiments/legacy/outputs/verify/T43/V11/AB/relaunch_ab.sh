#!/bin/bash
# V11AB relaunch of the 8 runs lost to server connection errors (H 1995:19953, B 1967:19672; first launch: chain F server idled 35 min on the memory guard -> keepalive ping timeout; B server 'no close frame').
# Same run_chain.sh / guards / flags as launch_v11.sh; two chains on 8286 / 8287 (8287 staggered by 240 s).
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/p/env.sh
ARM=V11AB; R=$V/runs/$ARM; L=($(cat $P/lists/hurdle.txt)); B="$(cat $P/lists/clear.txt)"
# hurdle items order: 1807a 1807c 1968a 1968c 1982a 1982c 1995a | 1995c 1967a 1967c 1925a 1925c 1834a 1834c
LA="${L[7]} ${L[8]} ${L[9]} ${L[10]}"; LB="${L[11]} ${L[12]} ${L[13]}"; BB=$(echo $B | tr ' ' '\n' | grep '^1967:19672')
echo "relaunch $(date -Iseconds) A=[$LA] B=[$LB] clear=[$BB]"
( EST_GB=28 $P/run_chain.sh $ARM 8286 $R "$LA"; echo "E2 ${ARM}H rc=$? $(date -Iseconds)" ) > $V/runs/chainE2_$ARM.out 2>&1 &
( sleep 240; EST_GB=28 $P/run_chain.sh $ARM 8287 $R "$LB"; echo "F2 ${ARM}H rc=$? $(date -Iseconds)"; sleep 5; EST_GB=28 $P/run_chain.sh $ARM 8287 $R "$BB"; echo "F2 ${ARM}B rc=$? $(date -Iseconds)" ) > $V/runs/chainF2_$ARM.out 2>&1 &
wait
echo "all end relaunch $(date -Iseconds)"
