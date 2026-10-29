#!/bin/bash
# T44c A6 implementer smoke: 2 shards (CPU worker <= 2), nice 19, 1 thread each.
cd ${AG3S_PROJECT_ROOT}
E="env PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
S=outputs/impl/T44/A6
for k in 0 1; do
  nice -n 19 $E $PY $S/a6_smoke.py --out $S/raw_demo --shard $k/2 > $S/run_shard$k.log 2>&1 &
done
wait
echo "done $(date -Iseconds)"
