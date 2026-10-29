#!/bin/bash
# T43 P coarse pass: 4 workers, nice 19, CPU only, 1 thread each.
cd ${AG3S_PROJECT_ROOT}/outputs/verify/T43/P
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
nice -n 19 $PY p/psearch.py 1807_18071 > raw/psearch_w1.log 2>&1 &
nice -n 19 $PY p/psearch.py 1968_19681 > raw/psearch_w2.log 2>&1 &
nice -n 19 $PY p/psearch.py 1982_19822 1967_19672 > raw/psearch_w3.log 2>&1 &
nice -n 19 $PY p/psearch.py 1995_19952 > raw/psearch_w4.log 2>&1 &
wait
echo "done $(date -Iseconds)" >> raw/psearch.done
