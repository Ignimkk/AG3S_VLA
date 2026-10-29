#!/bin/bash
# T43 QM — CPU physics replays of the 5 B (no obstacle) runs of V8 and V11AB (replay_ct.py from CT/p unchanged; 1 worker, nice 19;
# snapshots as in V8/p/run_replays_v8.sh (work-t43v8) and V11/p/run_replays_v11.sh (work-t43v11)).
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa OMP_NUM_THREADS=1
Q=outputs/verify/T43/QM
PYTHONPATH=${AG3S_PROJECT_ROOT} T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 nice -n 19 .venv-openpi-live/bin/python -u outputs/verify/T43/CT/p/replay_ct.py $Q/replay $(cat $Q/lists/b_V8.txt) > $Q/raw/replay_b_V8.log 2>&1
PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v11 T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v11 nice -n 19 .venv-openpi-live/bin/python -u outputs/verify/T43/CT/p/replay_ct.py $Q/replay $(cat $Q/lists/b_V11AB.txt) > $Q/raw/replay_b_V11AB.log 2>&1
echo done > $Q/raw/replay_b.done
