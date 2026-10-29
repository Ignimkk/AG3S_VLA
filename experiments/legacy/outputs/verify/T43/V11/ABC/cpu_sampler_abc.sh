#!/bin/bash
while pgrep -f "V11/ABC/run_chain_abc.sh" >/dev/null; do
  echo "$(date +%s) $(awk '$1=="usage_usec"||$1=="nr_periods"||$1=="nr_throttled"||$1=="throttled_usec"{printf "%s=%s ", $1,$2}' /sys/fs/cgroup/cpu.stat)"
  sleep 30
done
