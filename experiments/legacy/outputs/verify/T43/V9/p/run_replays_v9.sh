#!/bin/bash
# T43 V9 physics replays of the 14 H runs (CPU, 3 workers, nice 19; copy of CT/p/run_replays.sh; replay_ct.py from CT/p unchanged; client snapshot work-t43v8)
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} OMP_NUM_THREADS=1
V=outputs/verify/T43/V9
ls $V/lists/job_0? | xargs -P 3 -I{} bash -c 'j={}; T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 nice -n 19 .venv-openpi-live/bin/python -u outputs/verify/T43/CT/p/replay_ct.py '$V'/replay $(cat $j) > '$V'/raw/replay_$(basename $j).log 2>&1'
