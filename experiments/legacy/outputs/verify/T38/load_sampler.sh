#!/bin/bash
# T38: every 5 s, the load average and the top foreign CPU processes (to flag contaminated chunks). Stops when STOP file exists.
L=${AG3S_PROJECT_ROOT}/outputs/verify/T38/logs/load_samples.txt
while [ ! -f ${AG3S_PROJECT_ROOT}/outputs/verify/T38/logs/STOP_LOAD ]; do
  echo "=== $(date +%s.%N) $(cat /proc/loadavg)" >> $L
  ps -eo pid,pcpu,etime,args --sort=-pcpu | head -5 | cut -c1-160 >> $L
  sleep 5
done
