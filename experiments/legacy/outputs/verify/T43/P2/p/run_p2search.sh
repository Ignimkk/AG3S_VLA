#!/bin/bash
# T43 P2 coarse search, 4 workers, CPU, nice 19, 1 thread each.
cd ${AG3S_PROJECT_ROOT}
E="env MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 .venv-openpi-live/bin/python outputs/verify/T43/P2/p/p2search.py"
R=outputs/verify/T43/P2/raw
$E 1807_18071 1925_19251 > $R/p2search_w1.log 2>&1 &
$E 1968_19681 1834_18341 > $R/p2search_w2.log 2>&1 &
$E 1982_19822 1967_19672 > $R/p2search_w3.log 2>&1 &
$E 1995_19952 > $R/p2search_w4.log 2>&1 &
wait
echo done > $R/p2search.done
