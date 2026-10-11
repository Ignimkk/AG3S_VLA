#!/bin/bash
k=$1
cd ${AG3S_PROJECT_ROOT}
E="env PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
S=outputs/verify/T44/E2d
F=outputs/verify/T44/E2
nice -n 19 $E $PY $S/e2d_measure.py --posed-frames $F/e2_demo_posed_frames.json --out $S/raw_demo --shard $k/2
echo "demo shard $k rc=$? $(date -Iseconds)"
nice -n 19 $E $PY $S/e2d_measure.py --lb-frames outputs/verify/T44/p/e2_frames.json --out $S/raw_lb --shard $k/2 --history 2
echo "lb shard $k rc=$? $(date -Iseconds)"
