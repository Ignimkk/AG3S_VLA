#!/bin/bash
# T43 V11 physics replays of the 14 H runs of ARM (CPU, 3 workers, nice 19; copy of V8/p/run_replays_v8.sh; replay_ct.py from CT/p unchanged; client snapshot work-t43v11).
# usage: run_replays_v11.sh ARM     (run only after the round's chains have ended: replays must not load the CPU while the servers' wall-clock SQP budget runs)
ARM=${1:-V11A}
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v11 OMP_NUM_THREADS=1
V=outputs/verify/T43/V11; mkdir -p $V/lists $V/replay $V/raw
ls -d $V/runs/$ARM/H/${ARM}_ep*_s* 2>/dev/null | grep -v '\.' | sort > $V/lists/h_${ARM}.txt
split -n l/3 -d $V/lists/h_${ARM}.txt $V/lists/job_${ARM}_
ls $V/lists/job_${ARM}_* | xargs -P 3 -I{} bash -c 'j={}; T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v11 nice -n 19 .venv-openpi-live/bin/python -u outputs/verify/T43/CT/p/replay_ct.py '$V'/replay $(cat $j) > '$V'/raw/replay_$(basename $j).log 2>&1'
