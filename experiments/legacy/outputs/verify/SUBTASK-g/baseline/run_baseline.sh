#!/bin/bash
# SUBTASK-g regression baseline (same as SUBTASK-f baseline): legacy then cuRobo, serial, on snapshot ${AG3S_PROJECT_ROOT}-sg.
B=${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-g/baseline
{ date -Iseconds; nvidia-smi --query-gpu=memory.used --format=csv; ps aux --sort=-%cpu | head -8; } > $B/pre.txt 2>&1
cd ${AG3S_PROJECT_ROOT}-sg
COMMON="--records ${AG3S_PROJECT_ROOT}/run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms"
MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-sg XLA_PYTHON_CLIENT_PREALLOCATE=false ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout $COMMON --out-json $B/legacy.json > $B/legacy.log 2>&1
echo "legacy rc=$? $(date -Iseconds)" >> $B/run.out
MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-sg XLA_PYTHON_CLIENT_PREALLOCATE=false ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout $COMMON --esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005 --out-json $B/curobo.json > $B/curobo.log 2>&1
echo "curobo rc=$? $(date -Iseconds)" >> $B/run.out
touch $B/.done
