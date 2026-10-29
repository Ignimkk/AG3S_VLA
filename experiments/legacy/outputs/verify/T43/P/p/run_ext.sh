#!/bin/bash
# T43 P ep1995 extended pass (no placement passed a', b, c, d in the coarse + local passes): u 0.10..0.24, v 0.20..0.36, dyaw -75..+75.
cd ${AG3S_PROJECT_ROOT}/outputs/verify/T43/P
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PS_STEP=0.02 PS_U=0.10,0.24 PS_V=0.20,0.36 PS_DYAW=-75,-60,-45,-30,-15,0,15,30,45,60,75 PS_SUFFIX=_ext nice -n 19 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python p/psearch.py 1995_19952 > raw/psearch_ext.log 2>&1
echo "rc=$? $(date -Iseconds)" >> raw/psearch_ext.log
