#!/bin/bash
# T43W: every 30 s: epoch, 1-min loadavg, GPU util %, GPU used MiB, number of serve_safe / pi05_infer processes (anyone's). Ends with launch 'all end'.
O=${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/runs/load_samples.txt
while :; do
  echo "$(date +%s) $(cut -d' ' -f1 /proc/loadavg) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' | tr ',' ' ') $(pgrep -fc 'AG3S.trajopt.serve_safe') $(pgrep -fc 'python -u rby1_bringup/pi05_infer.py')" >> $O
  grep -q '^all end 56' ${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/runs/launch_56.out 2>/dev/null && break
  sleep 30
done
