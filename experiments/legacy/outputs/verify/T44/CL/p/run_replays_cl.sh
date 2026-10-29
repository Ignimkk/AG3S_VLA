#!/bin/bash
# T44d CL physics replays of every .done run (CPU, 3 workers, nice 19). run only after the chains have ended (servers' wall-clock SQP budget).
# replay = outputs/verify/T44/p/replay_contacts.py (verifier; T44_SERVER_ALIGN=1 keeps the best-effort server-block alignment)
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-t44cl OMP_NUM_THREADS=1 T43W_SNAP=${AG3S_PROJECT_ROOT}-t44cl
V=outputs/verify/T44/CL; mkdir -p $V/replay $V/raw $V/lists
ls -d $V/runs/*/R/*_ep*_s* 2>/dev/null | grep -v '\.' | sort > $V/lists/all_runs.txt
rm -f $V/lists/job_*; split -n l/3 -d $V/lists/all_runs.txt $V/lists/job_
ls $V/lists/job_* | xargs -P 3 -I{} bash -c 'j={}; nice -n 19 .venv-openpi-live/bin/python -u outputs/verify/T44/p/replay_contacts.py '$V'/replay $(cat $j) > '$V'/raw/replay_$(basename $j).log 2>&1'
