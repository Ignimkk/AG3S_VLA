#!/bin/bash
# T43 CX offline C toggle on the 1807 HOLD stretches (CPU, 2 workers at a time, nice 19).
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v11:${AG3S_PROJECT_ROOT}/outputs/verify/T43/CX/p OMP_NUM_THREADS=2
V=outputs/verify/T43/V11/analysis; O=outputs/verify/T43/CX/offline; PY=.venv-openpi-live/bin/python
run() { nice -n 19 $PY -u outputs/verify/T43/CX/p/cx_offline.py --index $V/analysis_$1.json --key $1/H/$1_ep1807_s$2 --out $O/$1_1807_s$2.json --variants base $3 --first-chunk $4 --max-chunks $5 --repeat-every 1000 --progress > $O/$1_1807_s$2.log 2>&1; }
run V11ABC 18071 noC 16 15 &
run V11ABC 18073 noC 15 15 &
wait
run V11AB 18071 addC 16 10 &
run V11AB 18073 addC 15 10 &
wait
