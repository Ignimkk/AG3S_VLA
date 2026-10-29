#!/bin/bash
cd ${AG3S_PROJECT_ROOT}
MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 nice -n 19   .venv-openpi-live/bin/python outputs/verify/T45/V/p/vt45.py --mode timing --field cuda --names C1a_ep0_s1 C1a_ep0_s2 C1a_ep104_s1041 C1a_ep104_s1042 C1a_ep110_s1101 C1a_ep110_s1102 C1a_ep11_s111 C1a_ep11_s112 C1a_ep123_s1231 C1a_ep123_s1232 C1a_ep12_s121 C1a_ep12_s122 C1a_ep30_s301 C1a_ep30_s302 C1a_ep3_s31 C1a_ep3_s32 C1a_ep54_s541 C1a_ep54_s542   --out outputs/verify/T45/V/raw/timing --progress > outputs/verify/T45/V/logs/timing.log 2>&1
touch outputs/verify/T45/V/logs/timing.done
