#!/bin/bash
# T43V11 launcher (snapshot ${AG3S_PROJECT_ROOT}-t43v11 = benchmark 3ca1b21). usage: nohup setsid launch_v11.sh ARM &   ARM = V11A | V11AB | V11ABC
# Two chains (ports 8286 / 8287), no other GPU round of ours runs; the chains' start guard staggers servers.
#   chain E (8286): hurdle items 1-7 -> no-obstacle 5     chain F (8287): hurdle items 8-14
# Same guards as p/run_chain.sh; resumable (.done).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
ARM=$1; is_arm $ARM || { echo "unknown ARM $ARM"; exit 2; }
R=$V/runs; L=($(sed "s|storage://|$AG3S_STORAGE_ROOT/|g" "$P/lists/hurdle.txt")); B="$(cat $P/lists/clear.txt)"
LA="${L[*]:0:7}"; LB="${L[*]:7:7}"
echo "launch wait $ARM $(date -Iseconds)"
echo "serve_safe procs now (read-only): $(pgrep -fc AG3S.trajopt.serve_safe)"
echo "launch $ARM $(date -Iseconds)"
mkdir -p $R/$ARM
( EST_GB=28 $P/run_chain.sh $ARM 8286 $R/$ARM "$LA"; echo "E ${ARM}H rc=$? $(date -Iseconds)"; sleep 5
  EST_GB=28 $P/run_chain.sh $ARM 8286 $R/$ARM "$B"; echo "E ${ARM}B rc=$? $(date -Iseconds)" ) > $R/chainE_$ARM.out 2>&1 &
( sleep 240; EST_GB=28 $P/run_chain.sh $ARM 8287 $R/$ARM "$LB"; echo "F ${ARM}H rc=$? $(date -Iseconds)" ) > $R/chainF_$ARM.out 2>&1 &
wait
echo "all end $ARM $(date -Iseconds)"
