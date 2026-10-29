#!/bin/bash
# pod loadavg / working set / GPU every 60 s -> runs/loadavg_samples.txt (same format as LB)
f=${AG3S_PROJECT_ROOT}/outputs/verify/T43/LB/LB3/runs/loadavg_samples.txt
while :; do
  ws=$(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat) ))
  echo "$(date -u -Iseconds) $(cat /proc/loadavg) wset=$ws gpu_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" >> $f
  sleep 60
done
