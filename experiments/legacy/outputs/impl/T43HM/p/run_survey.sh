#!/bin/bash
# T43 HM offline survey: one process per arm (2 concurrent <= 4), CPU only, recorded data only.
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa OMP_NUM_THREADS=2 PYTHONPATH=${AG3S_PROJECT_ROOT}
R=outputs/impl/T43HM/raw; PY=.venv-openpi-live/bin/python; S=outputs/impl/T43HM/p/hm_survey.py
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v7 $PY $S V7g $R/survey_V7g.json > $R/survey_V7g.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 $PY $S V8 $R/survey_V8.json > $R/survey_V8.log 2>&1 &
wait
