#!/bin/bash
# T43 SV part 2 (V3 / V4 approach-phase rows): 2 concurrent, CPU only, recorded data only.
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa OMP_NUM_THREADS=2 PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8
R=outputs/verify/T43/SV/raw; PY=.venv-openpi-live/bin/python; S=outputs/verify/T43/SV/p/sv_rows.py
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43w $PY $S V3 $R/rows_V3.json > $R/rows_V3.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43w $PY $S V4 $R/rows_V4.json > $R/rows_V4.log 2>&1 &
wait
