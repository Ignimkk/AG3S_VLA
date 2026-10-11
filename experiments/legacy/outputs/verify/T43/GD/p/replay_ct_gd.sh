#!/bin/bash
# physics replay with hurdle contacts (CT/p/replay_ct.py, unchanged, only called) of every H-stage V7 and V7g run -> T43/GD/replay_ct (skips finished json)
T=${AG3S_PROJECT_ROOT}/outputs/verify/T43
export T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v7 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=${AG3S_PROJECT_ROOT} OMP_NUM_THREADS=1
for ARM in V7 V7g; do case $ARM in V7) R=$T/GD/runs;; V7g) R=$T/V7g/runs;; esac
  RUNS=""; for d in $R/$ARM/H/${ARM}_ep*_s*.done; do RUNS="$RUNS ${d%.done}"; done
  ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -u $T/CT/p/replay_ct.py $T/GD/replay_ct $RUNS >> $T/GD/raw/replay_ct_$ARM.log 2>&1
done
echo done > $T/GD/raw/replay_ct.done
