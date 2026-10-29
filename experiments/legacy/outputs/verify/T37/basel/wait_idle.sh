#!/bin/bash
# wait until no python process >5% cpu (other than ourselves) for 3 consecutive 10 s checks, and GPU memory 0. max ~ $1 s
MAX=${1:-580}; ok=0; t=0
while [ $t -lt $MAX ]; do
  busy=$(ps -eo pid,pcpu,args | grep python | grep -v grep | grep -v wait_idle | awk '$2>5.0' | wc -l)
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$busy" = 0 ] && [ "$gpu" -lt 100 ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 3 ] && { echo "idle at $(date -Iseconds)"; exit 0; }
  sleep 10; t=$((t+10))
done
echo "still busy at $(date -Iseconds)"; ps -eo pid,stat,pcpu,etime,args --sort=-pcpu | grep python | grep -v grep | awk '$3>5.0' | cut -c1-250; nvidia-smi --query-gpu=memory.used --format=csv,noheader; exit 1
