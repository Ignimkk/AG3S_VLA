#!/bin/bash
# SUBTASK-g: serial — (1) regression baseline legacy + cuRobo, (2) E3bD = E3b flags with NO gate flag (default) on port 8230,
# ep1807 s18071 then ep1800 s18001, (3) E3bN = E3b + --no-subtask-gate on port 8231, same two runs, (4) compare_v0.py:
# E3bD vs SUBTASK-f runs/E3bG (gate on), E3bN vs T39 E3b (gate off) and vs SUBTASK-f v0/E3b (gate off). One server at a time.
# usage: nohup setsid launch_all.sh > launch_all.out 2>&1 &
source ${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-g/p/env.sh
echo "start $(date -Iseconds)"
B=$SD_OUT/baseline
{ echo "cgroup memory.current=$(cat /sys/fs/cgroup/memory.current) max=$(cat /sys/fs/cgroup/memory.max)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; } > $B/cgroup_gpu_at_start.txt 2>&1
ps aux --sort=-%cpu | head -15 > $B/concurrent_at_start.txt
$B/run_baseline.sh; echo "baseline rc=$? $(date -Iseconds)"
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $PY $B/judge.py > $B/judge.out 2>&1; echo "judge rc=$? $(grep -c '"matched": true' $B/judge.out) matched"
L="1807:18071 1800:18001"
MEM_EST_GB=16 EST_GB=22 $P/run_chain.sh E3bD 8230 $SD_OUT/runs/E3bD "$L"; echo "chain E3bD rc=$? $(date -Iseconds)"
MEM_EST_GB=16 EST_GB=22 $P/run_chain.sh E3bN 8231 $SD_OUT/runs/E3bN "$L"; echo "chain E3bN rc=$? $(date -Iseconds)"
CMP=$SD_OUT/compare; mkdir -p $CMP
T39=${AG3S_PROJECT_ROOT}/outputs/verify/T39; SF=${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-f
cmpone() { # OUT_NAME REF NEW
  if [ -f $(dirname $3)/$(basename $3).done ]; then
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $PY $P/compare_v0.py $CMP/$1.json $2 $3 > $CMP/$1.log 2>&1
    echo "compare $1 rc=$? $(grep -m1 all_identical $CMP/$1.log)"
  else echo "compare $1 skipped: run missing"; fi; }
cmpone D_vs_SFon_ep1807_s18071 $SF/runs/E3bG/E3bG_ep1807_s18071 $SD_OUT/runs/E3bD/E3bD_ep1807_s18071
cmpone D_vs_SFon_ep1800_s18001 $SF/runs/E3bG/E3bG_ep1800_s18001 $SD_OUT/runs/E3bD/E3bD_ep1800_s18001
cmpone N_vs_T39_ep1807_s18071 $T39/s21/E3b_a/E3b_ep1807_s18071 $SD_OUT/runs/E3bN/E3bN_ep1807_s18071
cmpone N_vs_T39_ep1800_s18001 $T39/runs/E3b/E3b_ep1800_s18001 $SD_OUT/runs/E3bN/E3bN_ep1800_s18001
cmpone N_vs_SFoff_ep1807_s18071 $SF/v0/E3b/E3b_ep1807_s18071 $SD_OUT/runs/E3bN/E3bN_ep1807_s18071
cmpone N_vs_SFoff_ep1800_s18001 $SF/v0/E3b/E3b_ep1800_s18001 $SD_OUT/runs/E3bN/E3bN_ep1800_s18001
echo "end $(date -Iseconds)"
