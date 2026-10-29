#!/bin/bash
# pod cgroup cpu.stat (usage_usec nr_periods nr_throttled throttled_usec) every 30 s while a V11 run_chain exists (read-only)
while pgrep -f "T43/V11/p/run_chain.sh" >/dev/null; do
  echo "$(date +%s) $(awk '$1=="usage_usec"||$1=="nr_periods"||$1=="nr_throttled"||$1=="throttled_usec"{printf "%s=%s ", $1,$2}' /sys/fs/cgroup/cpu.stat)"
  sleep 30
done
