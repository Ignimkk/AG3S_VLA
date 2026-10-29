#!/bin/bash
# T43 CX: physics replays (CT/p/replay_ct.py unchanged) of the B-stage runs not replayed in V11 (CPU, 2 workers, nice 19, snapshot work-t43v11).
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v11 OMP_NUM_THREADS=1 T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v11
V=outputs/verify/T43/V11/runs; O=outputs/verify/T43/CX
nice -n 19 .venv-openpi-live/bin/python -u outputs/verify/T43/CT/p/replay_ct.py $O/replay $V/V11AB/B/V11AB_ep1967_s19672 $V/V11ABC/B/V11ABC_ep1967_s19672 > $O/raw/replay_job0.log 2>&1 &
nice -n 19 .venv-openpi-live/bin/python -u outputs/verify/T43/CT/p/replay_ct.py $O/replay $V/V11AB/B/V11AB_ep1982_s19822 $V/V11ABC/B/V11ABC_ep1982_s19822 > $O/raw/replay_job1.log 2>&1 &
wait
