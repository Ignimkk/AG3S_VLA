#!/bin/bash
# T43 S2: demo replays, N parallel workers (CPU, nice 19, 1 thread each incl. llvmpipe). usage: run_replays.sh N EP...
N=$1; shift
R=${AG3S_PROJECT_ROOT}/outputs/verify/T43/S2/raw/replay; mkdir -p $R
one() {
  ep=$1; R=${AG3S_PROJECT_ROOT}/outputs/verify/T43/S2/raw/replay
  [ -f $R/DEMO_ep$ep.replay.json ] && { echo "skip $ep"; return; }
  cd ${AG3S_PROJECT_ROOT}-t42/pi05_TO_hybrid && env MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-t42 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 nice -n 19 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -u \
    ${AG3S_PROJECT_ROOT}/outputs/verify/T43/S2/p/replay_demo.py $ep $R > $R/DEMO_ep$ep.client.log 2>&1
  echo "ep $ep rc=$?"
}
export -f one
printf "%s\n" "$@" | xargs -P $N -I{} bash -c 'one {}'
