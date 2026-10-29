#!/bin/bash
# loadavg + pod working set every 10 s while a V11 run_chain exists (AB copy of p/load_sampler.sh; own output file)
while pgrep -f "T43/V11/p/run_chain.sh" >/dev/null; do
  ws=$(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat) ))
  echo "$(date +%s) $(cat /proc/loadavg) wset=$ws"
  sleep 10
done
