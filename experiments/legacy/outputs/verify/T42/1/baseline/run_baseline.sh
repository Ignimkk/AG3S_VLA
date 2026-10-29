#!/bin/bash
# T42-1 regression baseline (copy of T40/G/baseline/run_baseline.sh): legacy then cuRobo, serial, on snapshot ${AG3S_PROJECT_ROOT}-t42.
B=${AG3S_PROJECT_ROOT}/outputs/verify/T42/1/baseline
{ date -Iseconds; nvidia-smi --query-gpu=memory.used --format=csv; cat /sys/fs/cgroup/memory.current; ps aux --sort=-%cpu | head -8; } > $B/pre.txt 2>&1
cd ${AG3S_PROJECT_ROOT}-t42
COMMON="--records ${AG3S_PROJECT_ROOT}/run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms"
for b in legacy curobo; do
  [ -f $B/$b.done ] && continue
  EXTRA=""; [ $b = curobo ] && EXTRA="--esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005"
  MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-t42 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout $COMMON $EXTRA --out-json $B/$b.json > $B/$b.log 2>&1
  rc=$?; echo "$b rc=$rc $(date -Iseconds)" >> $B/run.out; [ $rc = 0 ] && touch $B/$b.done
done
touch $B/.done
