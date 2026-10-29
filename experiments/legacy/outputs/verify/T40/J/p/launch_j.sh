#!/bin/bash
# T40 J launcher: ONE server at a time (lead rule: T42-1 runs up to 2 servers on 8250-8259). E3b chain on 8244, then E0 chain on 8245.
# Memory guard: working set (memory.current - inactive_file) + EST_RSS <= 64 GiB and no serve_safe younger than 120 s; GPU guard inside run_chain.sh.
# Resumable: done runs (.done) are skipped. usage: nohup launch_j.sh > ${AG3S_PROJECT_ROOT}/outputs/verify/T40/J/runs/launch_j.out 2>&1 &
source ${AG3S_PROJECT_ROOT}/outputs/verify/T40/J/p/env.sh
R=$T40/J/runs; mkdir -p $R
# lead rule 2026-10-05: working set = memory.current - inactive_file (memory.stat); working set + estimate <= 64 GiB;
# and no serve_safe process (any session) younger than 120 s (stagger between verifiers).
wset() { echo $(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat) )); }
young() { for p in $(pgrep -f "AG3S.trajopt.serve_saf[e]"); do e=$(ps -o etimes= -p $p 2>/dev/null | tr -d ' '); [ -n "$e" ] && [ "$e" -lt 120 ] && return 0; done; return 1; }
memok() { local need=$1 t0=$(date +%s); while :; do ws=$(wset)
  if [ $(( ws + need )) -le $(( 64 * 1024 * 1024 * 1024 )) ] && ! young; then echo "mem ok working_set=$ws need=$need $(date -Iseconds)"; return 0; fi
  echo "mem wait working_set=$ws need=$need young_server=$(young && echo 1 || echo 0) $(date -Iseconds)"; [ $(( $(date +%s) - t0 )) -ge 3600 ] && return 1; sleep 30; done; }
echo "j start $(date -Iseconds)"
memok 10737418240 && EST_GB=22 $P/run_chain.sh E3b 8244 $R/E3b "$(cat $P/lists/E3b_J.txt)"; echo "E3b rc=$? $(date -Iseconds)"
memok 7516192768 && EST_GB=10 $P/run_chain.sh E0 8245 $R/E0 "$(cat $P/lists/E0_J.txt)"; echo "E0 rc=$? $(date -Iseconds)"
echo "j end $(date -Iseconds)"
