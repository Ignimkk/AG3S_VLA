#!/bin/bash
# T43 TA offline: one run key per process (CPU only, single-threaded BLAS). usage: run_one.sh INDEX KEY OUT
OUT=$(realpath -m "$3")
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv-openpi-live/bin/python -m AG3S.trajopt.experiments.ta_offline --index "$1" --key "$2" \
  --contacts AG3S/ag3s/docs/handoff/T43HX.verify.json --out "$OUT"
