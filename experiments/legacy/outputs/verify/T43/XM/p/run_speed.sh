#!/bin/bash
# T43 XM part 2: V7g / V8 x stage H / B (4 concurrent <= 4), CPU only, recorded data only.
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa OMP_NUM_THREADS=2 PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8
R=outputs/verify/T43/XM/raw; PY=.venv-openpi-live/bin/python; S=outputs/verify/T43/XM/p/xm_speed.py
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v7 $PY $S V7g $R/speed_V7g_H.json /H/ > $R/speed_V7g_H.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v7 $PY $S V7g $R/speed_V7g_B.json /B/ > $R/speed_V7g_B.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 $PY $S V8 $R/speed_V8_H.json /H/ > $R/speed_V8_H.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 $PY $S V8 $R/speed_V8_B.json /B/ > $R/speed_V8_B.log 2>&1 &
wait
