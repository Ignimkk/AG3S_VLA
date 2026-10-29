#!/bin/bash
# T43 XM part 1: one process per arm (4 concurrent <= 4), CPU only, recorded data only.
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa OMP_NUM_THREADS=2 PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8
R=outputs/verify/T43/XM/raw; PY=.venv-openpi-live/bin/python; S=outputs/verify/T43/XM/p/xm_cross.py
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43w3 $PY $S V5 $R/cross_V5.json > $R/cross_V5.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v7 $PY $S V7g $R/cross_V7g.json > $R/cross_V7g.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 $PY $S V8 $R/cross_V8.json > $R/cross_V8.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 $PY $S V9 $R/cross_V9.json > $R/cross_V9.log 2>&1 &
wait
