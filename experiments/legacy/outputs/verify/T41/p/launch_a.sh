#!/bin/bash
# T41a launcher: ONE E3b chain on port 8260 (B 6 + C 5 + D 4 runs, lists/E3b_a.txt; snapshot ${AG3S_PROJECT_ROOT}-t41a).
# Memory guard (lead rule 2026-10-05): before the server, working set (memory.current - inactive_file) + EST_RSS_SERVER <= 64 GiB and
# no serve_safe started < 120 s ago (re-check every 60 s, max 2 h);
# run_chain.sh re-checks working set + 4 GiB before every client and GPU used + EST_GB <= 130 GB before the server.
# Estimates from T40 G (rss_samples / gpu_samples): server RSS 6.9 GiB, client RSS 3.0 GiB, server GPU +26.6 GiB.
# usage: nohup launch_a.sh > ${AG3S_PROJECT_ROOT}/outputs/verify/T41/a/runs/launch.out 2>&1 &
source ${AG3S_PROJECT_ROOT}/outputs/verify/T41/p/env.sh
R=$T41/a/runs; mkdir -p $R
memok() { local need=$1 t0=$(date +%s); while :; do cur=$(wset); rs=$(recent_server_s); \
  if [ $(( cur + need )) -le $(( 64 * 1024 * 1024 * 1024 )) ] && { [ -z "$rs" ] || [ "$rs" -ge 120 ]; }; then echo "mem ok working_set=$cur need=$need youngest_server_s=${rs:-none} $(date -Iseconds)"; return 0; fi; \
  echo "mem wait working_set=$cur need=$need youngest_server_s=${rs:-none} $(date -Iseconds)"; [ $(( $(date +%s) - t0 )) -ge 7200 ] && return 1; sleep 60; done; }
echo "a start $(date -Iseconds)"
EST_RSS_SERVER=${EST_RSS_SERVER:-8589934592}
memok $EST_RSS_SERVER && EST_GB=28 $P/run_chain.sh E3b 8260 $R/E3b "$(cat $P/lists/E3b_a.txt)"; echo "E3b rc=$? $(date -Iseconds)"
echo "a end $(date -Iseconds)"
