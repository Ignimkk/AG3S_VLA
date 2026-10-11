#!/bin/bash
# T40 D launcher (copy of launch_g.sh): round-2 hurdle items per lists/<COND>_D.txt; E3b on 8242, E0 on 8243.
# Memory guard (lead rule): before each chain, /sys/fs/cgroup/memory.current + EST_RSS <= 64 GiB, re-check every 60 s (max 1 h).
# usage: nohup launch_g.sh [SERIAL] > ${AG3S_PROJECT_ROOT}/outputs/verify/T40/G/runs/launch_d.out 2>&1 &     SERIAL=1 -> one chain at a time
source ${AG3S_PROJECT_ROOT}/outputs/verify/T40/p/env.sh
R=$T40/G/runs; mkdir -p $R; SERIAL=${1:-0}
memok() { local need=$1 t0=$(date +%s); while :; do cur=$(cat /sys/fs/cgroup/memory.current); \
  [ $(( cur + need )) -le $(( 64 * 1024 * 1024 * 1024 )) ] && { echo "mem ok current=$cur need=$need $(date -Iseconds)"; return 0; }; \
  echo "mem wait current=$cur need=$need $(date -Iseconds)"; [ $(( $(date +%s) - t0 )) -ge 3600 ] && return 1; sleep 60; done; }
echo "g start $(date -Iseconds) serial=$SERIAL"
EST_RSS_E3B=${EST_RSS_E3B:-12884901888}; EST_RSS_E0=${EST_RSS_E0:-10737418240}   # server + client RSS estimate (bytes), updated after the first runs
if [ "$SERIAL" = 1 ]; then
  memok $EST_RSS_E3B && EST_GB=22 $P/run_chain.sh E3b 8242 $R/E3b "$(cat $P/lists/E3b_D.txt)"; echo "E3b rc=$? $(date -Iseconds)"
  memok $EST_RSS_E0 && EST_GB=10 $P/run_chain.sh E0 8243 $R/E0 "$(cat $P/lists/E0_D.txt)"; echo "E0 rc=$? $(date -Iseconds)"
else
  ( memok $EST_RSS_E3B && EST_GB=22 $P/run_chain.sh E3b 8242 $R/E3b "$(cat $P/lists/E3b_D.txt)"; echo "E3b rc=$? $(date -Iseconds)" ) & p1=$!; sleep 240
  ( memok $EST_RSS_E0 && EST_GB=10 $P/run_chain.sh E0 8243 $R/E0 "$(cat $P/lists/E0_D.txt)"; echo "E0 rc=$? $(date -Iseconds)" ) & p2=$!
  wait $p1; wait $p2
fi
echo "g end $(date -Iseconds)"
