#!/bin/bash
# T43 P regression baseline, legacy backend only, CPU only (lane P: CPU only, no .venv-ag3s) -> .venv-openpi-live, JAX on CPU.
# cuRobo baseline not run (needs GPU; lane P is CPU only).
B=${AG3S_PROJECT_ROOT}/outputs/verify/T43/P/baseline
{ date -Iseconds; git -C ${AG3S_PROJECT_ROOT}/benchmark rev-parse --short HEAD; git -C ${AG3S_PROJECT_ROOT}/benchmark status --short -- '*.py'; cat /sys/fs/cgroup/memory.current; awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat; } > $B/pre.txt 2>&1
cd ${AG3S_PROJECT_ROOT}
COMMON="--records ${AG3S_PROJECT_ROOT}/run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms"
env MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 nice -n 19 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout $COMMON --out-json $B/legacy.json > $B/legacy.log 2>&1
echo "legacy rc=$? $(date -Iseconds)" >> $B/run.out
