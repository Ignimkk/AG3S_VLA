#!/bin/bash
# T43 FR offline driver: one process per snapshot, at most 4 at once (pod rule), CPU only. Same keys as T43 FC.
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=2 PYTHONPATH=${AG3S_PROJECT_ROOT}
OUT=outputs/impl/T43FR/offline
run() {  # snapshot name
  snap=${AG3S_PROJECT_ROOT}-$1
  keys=$(python3 -c "import json;print(' '.join(json.load(open('outputs/impl/T43FC/keys_by_snapshot.json'))['$snap']))")
  nice -n 19 .venv-openpi-live/bin/python outputs/impl/T43FR/fr_offline.py $OUT/fr_$1.json $snap $keys > $OUT/fr_$1.log 2>&1
}
run t43w & run t43w3 & run t43b & run t43z3 & wait
run t43v7
echo done > $OUT/ALL_DONE
