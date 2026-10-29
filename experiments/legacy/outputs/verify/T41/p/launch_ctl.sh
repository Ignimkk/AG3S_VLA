#!/bin/bash
# T41a control (lead 2026-10-05): B ep1967 with margin 0 (seeds 19672 = T40 seed, 19671, 19673) then margin 0.010 (19671, 19673).
# one server at a time on 8261; guards as launch_a.sh (working set + 8 GiB <= 64 GiB, youngest serve_safe >= 120 s, GPU used + 28 <= 130 GB).
source ${AG3S_PROJECT_ROOT}/outputs/verify/T41/p/env.sh
R=$T41/ctl/runs; mkdir -p $R
memok() { local need=$1 t0=$(date +%s); while :; do cur=$(wset); rs=$(recent_server_s); \
  if [ $(( cur + need )) -le $(( 64 * 1024 * 1024 * 1024 )) ] && { [ -z "$rs" ] || [ "$rs" -ge 120 ]; }; then echo "mem ok working_set=$cur need=$need youngest_server_s=${rs:-none} $(date -Iseconds)"; return 0; fi; \
  echo "mem wait working_set=$cur need=$need youngest_server_s=${rs:-none} $(date -Iseconds)"; [ $(( $(date +%s) - t0 )) -ge 7200 ] && return 1; sleep 60; done; }
echo "ctl start $(date -Iseconds)"
memok 8589934592 && OBSTACLE_MARGIN=0 EST_GB=28 $P/run_chain.sh E3b 8261 $R/m0/E3b "$(cat $P/lists/E3b_ctl_m0.txt)"; echo "m0 rc=$? $(date -Iseconds)"
sleep 130
memok 8589934592 && OBSTACLE_MARGIN=0.010 EST_GB=28 $P/run_chain.sh E3b 8261 $R/m10/E3b "$(cat $P/lists/E3b_ctl_m10.txt)"; echo "m10 rc=$? $(date -Iseconds)"
echo "ctl end $(date -Iseconds)"
