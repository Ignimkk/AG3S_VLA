#!/bin/bash
# SUBTASK-f V1 resume after the 12:21 OOM kill (pod memory cgroup shared with other sessions).
# Waits until the cgroup has >= 64 GB headroom (4 chains x 16 GB) for 3 consecutive minutes, then runs launch_v1.sh 4 (resumable).
source ${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-f/p/env.sh
echo "resume start $(date -Iseconds)"
ok=0
while [ $ok -lt 3 ]; do
  f=$(( ($(cat /sys/fs/cgroup/memory.max) - $(cat /sys/fs/cgroup/memory.current)) / 1073741824 ))
  echo "$(date -Iseconds) cgroup free_gb=$f"
  if [ $f -ge 64 ]; then ok=$((ok+1)); else ok=0; fi
  sleep 60
done
echo "headroom ok $(date -Iseconds)"
rm -f $SD_OUT/.v1_finished
GPU_WAIT_MAX_S=7200 $P/launch_v1.sh 4 > $SD_OUT/runs/launch_resume.out 2>&1; echo "v1 rc=$? $(date -Iseconds)"
touch $SD_OUT/.v1_finished
echo "resume end $(date -Iseconds)"
