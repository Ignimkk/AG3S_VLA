#!/bin/bash
# T43 HX: one process per arm (3 concurrent <= 4), CPU only, recorded data only.
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa OMP_NUM_THREADS=2 PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8
R=outputs/verify/T43/HX/raw; PY=.venv-openpi-live/bin/python; S=outputs/verify/T43/HX/p/hx_measure.py
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v7 $PY $S V7g $R/hx_V7g.json > $R/hx_V7g.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 $PY $S V8 $R/hx_V8.json > $R/hx_V8.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43w3 $PY $S V5 $R/hx_V5.json > $R/hx_V5.log 2>&1 &
wait
