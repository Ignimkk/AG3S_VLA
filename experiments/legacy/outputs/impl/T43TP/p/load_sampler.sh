#!/bin/bash
# T43 TP: sample pod load every 10 s until the file STOP appears. usage: load_sampler.sh OUT STOPFILE
OUT=$1; STOP=$2
while [ ! -e "$STOP" ]; do
  echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg) $(awk '{printf "%s=%s ", $1, $2}' /sys/fs/cgroup/cpu.stat)" >> "$OUT"
  sleep 10
done
