#!/bin/bash
# T43 CT — physics replays (CPU, <= 4 workers; one process per snapshot-job). re-runnable (skips finished json).
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} OMP_NUM_THREADS=1
C=outputs/verify/T43/CT
ls $C/lists/job_*.list | xargs -P 4 -I{} bash -c 'j={}; b=${j%.list}; T43W_SNAP=$(cat $b.snap) .venv-openpi-live/bin/python -u '$C'/p/replay_ct.py '$C'/replay $(cat $j) > '$C'/raw/replay_$(basename $b).log 2>&1'
