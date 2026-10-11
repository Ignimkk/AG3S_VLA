#!/bin/bash
# T42-2: run search.py for EP:SEED x SLOT items, at most 4 at a time (nice 19, 1 thread each). usage: run_searches.sh "EP:SEED ..." "SLOT ..."
cd ${AG3S_PROJECT_ROOT}/outputs/verify/T42/2
for e in $1; do for s in $2; do
  while [ $(pgrep -fc "p/search.py") -ge 4 ]; do sleep 5; done
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=1 timeout 3600 nice -n 19 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python p/search.py ${e%:*} ${e#*:} $s > raw/logs/search_${e%:*}_$s.log 2>&1 &
  sleep 1
done; done; wait
