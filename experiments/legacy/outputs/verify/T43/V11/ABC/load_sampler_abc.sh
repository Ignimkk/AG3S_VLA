#!/bin/bash
# loadavg + pod working set every 10 s while a V11ABC run_chain exists
while pgrep -f "V11/ABC/run_chain_abc.sh" >/dev/null; do
  ws=$(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat) ))
  echo "$(date +%s) $(cat /proc/loadavg) wset=$ws"
  sleep 10
done
