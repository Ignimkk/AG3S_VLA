#!/bin/bash
# usage: run.sh <label> <code: tag|b1|b2|b3|b4> <run> <n> <rec: bg|sync|none> <heavy 0|1> [cpu 0|1] [keep seqs]
# Sequential, foreground. cwd ${AG3S_PROJECT_ROOT}-o4 (never ${AG3S_PROJECT_ROOT}). Code root = git archive copy of the commit.
B=${AG3S_PROJECT_ROOT}/outputs/verify/T38/B
CODES=/tmp/claude-0/-mnt-dev-work/769cea2f-2a69-4fbe-beed-8be40017af85/scratchpad/t38v/code
label=$1; code=$2; run=$3; n=$4; rec=$5; heavy=$6; cpu=${7:-0}; keep=${8:-}
O=$B/runs/$label
mkdir -p $O
cd ${AG3S_PROJECT_ROOT}-o4
{ date -Iseconds; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv; uptime; } > $O/pre_check.txt
env MUJOCO_GL=osmesa AG3S_ASSET_ROOT=${AG3S_PROJECT_ROOT}/src AG3S_CLIENT_SCRIPT=integration/rby1/bringup/pi05_infer.py \
  PYTHONPATH=$CODES/$code VR_CODE=$CODES/$code VR_OUT=$O VR_REC=$rec VR_HEAVY=$heavy VR_CPU=$cpu VR_KEEP=$keep \
  ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python $B/vreplay.py $run $n --warm 4 > $O/stdout.log 2>&1
rc=$?
{ date -Iseconds; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv; uptime; } > $O/post_check.txt
echo "$label code=$code run=$run n=$n rec=$rec heavy=$heavy cpu=$cpu rc=$rc $(date -Iseconds)" >> $B/runs.txt
echo "rc=$rc"; tail -2 $O/stdout.log | cut -c1-600
