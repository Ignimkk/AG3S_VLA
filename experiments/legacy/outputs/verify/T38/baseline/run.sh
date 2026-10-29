#!/bin/bash
# T38 baseline: worktree code (${AG3S_PROJECT_ROOT}-o4), .venv-openpi-live, legacy then cuRobo, sequential.
O=${AG3S_PROJECT_ROOT}/outputs/verify/T38/baseline
cd ${AG3S_PROJECT_ROOT}-o4
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv > $O/pre.txt; nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> $O/pre.txt
MUJOCO_GL=osmesa AG3S_ASSET_ROOT=${AG3S_PROJECT_ROOT}/src AG3S_CLIENT_SCRIPT=integration/rby1/bringup/pi05_infer.py PYTHONPATH=${AG3S_PROJECT_ROOT}-o4 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout --records ${AG3S_PROJECT_ROOT}/run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms --out-json $O/legacy.json > $O/legacy.log 2>&1
echo "legacy rc=$?" >> $O/rc.txt
MUJOCO_GL=osmesa AG3S_ASSET_ROOT=${AG3S_PROJECT_ROOT}/src AG3S_CLIENT_SCRIPT=integration/rby1/bringup/pi05_infer.py PYTHONPATH=${AG3S_PROJECT_ROOT}-o4 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout --records ${AG3S_PROJECT_ROOT}/run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms --esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005 --out-json $O/curobo.json > $O/curobo.log 2>&1
echo "curobo rc=$?" >> $O/rc.txt
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv > $O/post.txt; nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> $O/post.txt
