#!/bin/bash
# T36-baseline: one esdf_rollout run, alone. usage: run_one.sh legacy|curobo TAG [PY]
#   legacy -> skill command exactly (default PY = scratchpad venv-ag3s-local shim, since .venv-ag3s hangs on NFS)
#   curobo -> skill command + --esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005 (PY = .venv-openpi-live)
set -u
MODE=$1; TAG=$2
SHIM=/tmp/claude-0/-mnt-dev-work/769cea2f-2a69-4fbe-beed-8be40017af85/scratchpad/venv-ag3s-local/bin/python
if [ "$MODE" = legacy ]; then PY=${3:-$SHIM}; EXTRA=""; else PY=${3:-${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python}; EXTRA="--esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005"; fi
D=${AG3S_PROJECT_ROOT}/outputs/verify/T36/$MODE; mkdir -p $D
cd ${AG3S_PROJECT_ROOT}
{
  echo "start $(date -Iseconds)"
  echo "python $PY"
  echo "loadavg $(cat /proc/loadavg)"
  echo "gpu $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
  echo "gpu_procs $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')"
  echo "python procs >5% cpu:"; ps -eo pid,stat,pcpu,etime,args --sort=-pcpu | grep -E "python" | grep -v grep | awk '$3>5.0' | cut -c1-220
} > $D/$TAG.pre.txt
CMD="MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} $PY -m AG3S.trajopt.experiments.esdf_rollout --records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms $EXTRA --out-json outputs/verify/T36/$MODE/$TAG.json"
echo "$CMD" > $D/$TAG.cmd.txt
( env MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} $PY -m AG3S.trajopt.experiments.esdf_rollout \
  --records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz \
  --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all \
  --support-surfaces field --constraint-links arms $EXTRA --out-json outputs/verify/T36/$MODE/$TAG.json ) > $D/$TAG.log 2>&1
RC=$?
{ echo "end $(date -Iseconds) rc=$RC"; echo "loadavg_end $(cat /proc/loadavg)";
  echo "python procs >5% cpu at end:"; ps -eo pid,stat,pcpu,etime,args --sort=-pcpu | grep -E "python" | grep -v grep | awk '$3>5.0' | cut -c1-220; } >> $D/$TAG.pre.txt
echo "rc=$RC"; tail -8 $D/$TAG.log
