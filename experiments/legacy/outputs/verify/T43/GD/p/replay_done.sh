#!/bin/bash
# usage: replay_done.sh ARM  -> physics replay (Y/p/replay_y.py, unchanged) of every ARM run with .done into T43/GD/replay/ARM (skips existing)
ARM=$1; T=${AG3S_PROJECT_ROOT}/outputs/verify/T43; case $ARM in V7) R=$T/GD/runs;; V7g) R=$T/V7g/runs;; esac
RUNS=""; for d in $R/$ARM/H/${ARM}_ep*_s*.done $R/$ARM/B/${ARM}_ep*_s*.done; do [ -f $d ] && RUNS="$RUNS ${d%.done}"; done
export T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v7 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=${AG3S_PROJECT_ROOT}
mkdir -p $T/GD/replay/$ARM
${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python $T/Y/p/replay_y.py $T/GD/replay/$ARM $RUNS >> $T/GD/raw/replay_$ARM.log 2>&1
