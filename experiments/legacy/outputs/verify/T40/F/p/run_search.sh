#!/bin/bash
# T40 F: 6 episodes x N workers of search.py, detached; logs raw/search/<tag>_w<k>.log
N=${N:-10}
cd ${AG3S_PROJECT_ROOT}/outputs/verify/T40/F
for t in 1807_18071 1968_19681 1982_19822 1828_18281 1995_19952 1967_19672; do
  for k in $(seq 0 $((N-1))); do
    [ -f raw/search/${t}_w$k.json ] && continue
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      nohup ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python p/search.py $t $k $N > raw/search/${t}_w$k.log 2>&1 &
  done
done
wait
