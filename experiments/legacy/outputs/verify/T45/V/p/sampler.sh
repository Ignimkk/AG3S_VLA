#!/bin/bash
# loadavg + nvidia-smi + cgroup throttling every 15 s until $1 exists
OUT=$2
while [ ! -e "$1" ]; do
  echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ') $(grep -E '^nr_throttled|^throttled_usec' /sys/fs/cgroup/cpu.stat | tr '\n' ' ')" >> $OUT
  sleep 15
done
