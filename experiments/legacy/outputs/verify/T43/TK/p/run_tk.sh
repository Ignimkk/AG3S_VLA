#!/bin/bash
# T43 TK: one process per arm (<= 3 concurrent), CPU only.
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} OMP_NUM_THREADS=2
R=outputs/verify/T43/TK/raw; PY=.venv-openpi-live/bin/python; S=outputs/verify/T43/TK/p/tk_measure.py
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v7 $PY $S V7g $R/tk_V7g.json > $R/tk_V7g.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 $PY $S V8 $R/tk_V8.json > $R/tk_V8.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43w3 $PY $S E0 $R/tk_E0.json > $R/tk_E0.log 2>&1 &
wait
