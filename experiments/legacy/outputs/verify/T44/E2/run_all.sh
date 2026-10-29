#!/bin/bash
# T44 V-E2 verifier: shard k of 2 — demo then lb. nice 19, 4 threads.
k=$1
cd ${AG3S_PROJECT_ROOT}
E="env MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
S=outputs/verify/T44/E2
nice -n 19 $E $PY $S/e2_measure.py --posed-frames $S/e2_demo_posed_frames.json --out $S/raw_demo --shard $k/2 --sign-check
echo "demo shard $k rc=$? $(date -Iseconds)"
nice -n 19 $E $PY $S/e2_measure.py --lb-frames outputs/verify/T44/p/e2_frames.json --out $S/raw_lb --shard $k/2 --history 2 --sign-check
echo "lb shard $k rc=$? $(date -Iseconds)"
