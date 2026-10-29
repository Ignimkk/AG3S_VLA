#!/bin/bash
# T43W regression baseline (copy of T41/c/baseline/run_baseline.sh; snapshot ${AG3S_PROJECT_ROOT}-t43b; guard uses working set = memory.current - inactive_file)
# guard: memory.current + 6 GiB <= 64 GiB and GPU used + 10 GB <= 130 GB before each run (re-check every 60 s).
B=${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/baseline_blind
guard() { while :; do cur=$(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat) )); g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ');
  if [ $(( cur + 6*1024*1024*1024 )) -le $(( 64*1024*1024*1024 )) ] && [ $(( (g+1023)/1024 + 10 )) -le 130 ]; then echo "guard ok $1 mem=$cur gpu_mib=$g $(date -Iseconds)" >> $B/run.out; return; fi
  echo "guard wait $1 mem=$cur gpu_mib=$g $(date -Iseconds)" >> $B/run.out; sleep 60; done; }
{ date -Iseconds; nvidia-smi --query-gpu=memory.used --format=csv; cat /sys/fs/cgroup/memory.current; ps aux --sort=-%cpu | head -8; } > $B/pre.txt 2>&1
cd ${AG3S_PROJECT_ROOT}-t43b
COMMON="--records ${AG3S_PROJECT_ROOT}/run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms"
for b in legacy curobo; do
  [ -f $B/$b.done ] && continue
  guard $b
  EXTRA=""; [ $b = curobo ] && EXTRA="--esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005"
  MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-t43b XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout $COMMON $EXTRA --out-json $B/$b.json > $B/$b.log 2>&1
  rc=$?; echo "$b rc=$rc $(date -Iseconds)" >> $B/run.out; [ $rc = 0 ] && touch $B/$b.done
done
${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python $B/judge.py > $B/judge.out 2>&1
touch $B/.done
