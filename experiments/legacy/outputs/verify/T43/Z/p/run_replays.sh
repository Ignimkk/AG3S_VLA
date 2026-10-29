#!/bin/bash
# T43 Z4 — physics replays of the W arms not covered by Y (E0, V1, V3 H+B, V5B, V5S) with Y/p/replay_y.py unchanged. CPU only.
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} OMP_NUM_THREADS=1
P=.venv-openpi-live/bin/python; Z=outputs/verify/T43/Z
$P -u outputs/verify/T43/Y/p/replay_y.py $Z/replay $(cat $Z/lists/main_part_00) > $Z/raw/replay_main_00.log 2>&1 &
$P -u outputs/verify/T43/Y/p/replay_y.py $Z/replay $(cat $Z/lists/main_part_01) > $Z/raw/replay_main_01.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43b $P -u outputs/verify/T43/Y/p/replay_y.py $Z/replay $(cat $Z/lists/blind.txt) > $Z/raw/replay_blind.log 2>&1 &
wait
