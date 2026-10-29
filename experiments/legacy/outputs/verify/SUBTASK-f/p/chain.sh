#!/bin/bash
# SUBTASK-f top chain (detached from the agent): pytest (CPU) -> regression baseline (legacy, cuRobo; serial) -> judge
# -> stop if baseline does not match -> p/launch_all.sh (V0 gate off 2 runs, then V1 gate on 48 runs).
# usage: setsid nohup p/chain.sh > chain.out 2>&1 &
O=${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-f
echo "chain start $(date -Iseconds)"
mkdir -p $O/pytest
( cd ${AG3S_PROJECT_ROOT} && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} \
  timeout 3600 third_party/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q -p no:cacheprovider > $O/pytest/pytest.log 2>&1; echo "rc=$?" >> $O/pytest/pytest.log )
echo "pytest $(tail -2 $O/pytest/pytest.log | tr '\n' ' ') $(date -Iseconds)"
{ date -Iseconds; ps aux --sort=-%cpu | head -12; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $O/baseline/concurrent_at_start.txt 2>&1
bash $O/baseline/run_baseline.sh
echo "baseline $(cat $O/baseline/run.out | tr '\n' ' ')"
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python $O/baseline/judge.py > $O/baseline/judge.out 2>&1
m=$(python3 -c "import json; d=json.load(open('$O/baseline/judge.json')); print(int(all(d[b]['matched'] for b in d)))" 2>/dev/null)
echo "baseline matched=$m $(date -Iseconds)"
if [ "$m" != 1 ]; then echo "STOP: baseline mismatch $(date -Iseconds)"; exit 1; fi
bash $O/p/launch_all.sh > $O/launch_all.out 2>&1
echo "launch_all rc=$? $(date -Iseconds)"
echo "chain end $(date -Iseconds)"
