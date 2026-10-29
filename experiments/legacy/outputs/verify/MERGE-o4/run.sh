#!/bin/bash
# MERGE-o4 verifier: regression baselines on merged working tree (${AG3S_PROJECT_ROOT}/benchmark, uncommitted merge),
# .venv-openpi-live (GPU venv). Usage: run.sh legacy|curobo   (run one at a time)
set -u
O=${AG3S_PROJECT_ROOT}/outputs/verify/MERGE-o4
K=$1
cd ${AG3S_PROJECT_ROOT}
EXTRA=""
[ "$K" = curobo ] && EXTRA="--esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005"
{ date -Is; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; uptime;
  git -C ${AG3S_PROJECT_ROOT}/benchmark rev-parse HEAD; cat ${AG3S_PROJECT_ROOT}/benchmark/.git/MERGE_HEAD 2>/dev/null || git -C ${AG3S_PROJECT_ROOT}/benchmark rev-parse MERGE_HEAD; git -C ${AG3S_PROJECT_ROOT}/benchmark status --short | grep -v '^??'; } > $O/$K.pre.txt
# GPU sampler (1 s)
( while true; do echo "$(date +%s.%N) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ') | $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ';')"; sleep 1; done ) > $O/$K.gpu_samples.txt 2>&1 &
SP=$!
T0=$(date +%s.%N)
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa AG3S_ASSET_ROOT=${AG3S_PROJECT_ROOT}/src AG3S_CLIENT_SCRIPT=integration/rby1/bringup/pi05_infer.py PYTHONPATH=${AG3S_PROJECT_ROOT} \
  ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python $O/wrap.py $O/$K.imports.json -- \
  --records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz \
  --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms $EXTRA \
  --out-json $O/$K.json > $O/$K.log 2>&1 &
PY=$!
echo $PY > $O/$K.pid
wait $PY; RC=$?
T1=$(date +%s.%N)
kill $SP
echo "$K rc=$RC wall_s=$(python3 -c "print(round($T1-$T0,1))") pid=$PY" >> $O/rc.txt
{ date -Is; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; uptime; } > $O/$K.post.txt
