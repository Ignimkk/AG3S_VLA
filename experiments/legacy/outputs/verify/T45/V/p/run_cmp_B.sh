#!/bin/bash
cd ${AG3S_PROJECT_ROOT}
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} OMP_NUM_THREADS=1 nice -n 19   .venv-openpi-live/bin/python outputs/verify/T45/V/p/vt45.py --mode cmp --field cpu --names C1a_ep123_s1232 C1a_ep12_s121 C1a_ep12_s122 C1a_ep30_s301 C1a_ep30_s302 C1a_ep3_s31 C1a_ep3_s32 C1a_ep54_s541 C1a_ep54_s542   --out outputs/verify/T45/V/raw/cmp --progress > outputs/verify/T45/V/logs/cmp_B.log 2>&1
touch outputs/verify/T45/V/logs/cmp_B.done
