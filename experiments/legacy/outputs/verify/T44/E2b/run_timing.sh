#!/bin/bash
# T44b dedicated timing pass: single process, run alone (no e2b_measure shard running). nice 19, 4 threads.
cd ${AG3S_PROJECT_ROOT}
E="env PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
S=outputs/verify/T44/E2b
nice -n 19 $E $PY $S/e2b_timing.py --posed-frames outputs/verify/T44/E2/e2_demo_posed_frames.json --out $S/timing/timing_demo.json
echo "demo rc=$? $(date -Iseconds)"
nice -n 19 $E $PY $S/e2b_timing.py --lb-frames outputs/verify/T44/p/e2_frames.json --history 2 --out $S/timing/timing_lb.json
echo "lb rc=$? $(date -Iseconds)"
