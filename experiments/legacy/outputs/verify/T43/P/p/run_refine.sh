#!/bin/bash
# T43 P local pass. usage: run_refine.sh LOGNAME TAG... (one process, nice 19, CPU only)
cd ${AG3S_PROJECT_ROOT}/outputs/verify/T43/P
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=$1; shift
nice -n 19 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python p/refine.py "$@" > raw/refine_$L.log 2>&1
echo "rc=$? $(date -Iseconds)" >> raw/refine_$L.log
