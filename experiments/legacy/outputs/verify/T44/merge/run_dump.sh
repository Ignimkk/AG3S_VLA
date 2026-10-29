#!/bin/bash
# T44b D-merge: dump shard k of n. nice 19, 2 threads per shard (2 shards = 4 workers).
k=$1; n=$2; att=${3:-gt}; vnpz=${4:-}
cd ${AG3S_PROJECT_ROOT}
E="env MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2"
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
S=outputs/verify/T44/merge
if [ "$att" = "gt" ]; then
  nice -n 19 $E $PY $S/dump_frames.py --out $S/raw_gt --shard $k/$n
else
  nice -n 19 $E $PY $S/dump_frames.py --out $S/raw_vla --attention vla --vla-npz $vnpz --shard $k/$n
fi
echo "shard $k rc=$? $(date -Iseconds)"
