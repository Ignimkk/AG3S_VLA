#!/bin/bash
# T41a post-processing (CPU only): analysis -> replay -> compare. usage: nohup post_a.sh > T41/a/analysis/post.out 2>&1 &
set -e
AN=${AG3S_PROJECT_ROOT}/outputs/verify/T41/a/analysis; P=${AG3S_PROJECT_ROOT}/outputs/verify/T41/p; PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
SRV=$(ls -d ${AG3S_PROJECT_ROOT}/outputs/verify/T41/a/runs/E3b/server_8260_* | head -1)
export MUJOCO_GL=osmesa JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=${AG3S_PROJECT_ROOT}-t41a XLA_PYTHON_CLIENT_PREALLOCATE=false
cd ${AG3S_PROJECT_ROOT}-t41a
nice -n 19 $PY $P/analyze_t41.py $AN/analysis_a.json > $AN/analysis_a.log 2>&1
T41_SERVER_DIR=$SRV T41_MANIFEST=${AG3S_PROJECT_ROOT}/outputs/verify/T41/a/runs/E3b/C/E3b_ep1807_s18071/manifest.json nice -n 19 $PY $P/replay_t41.py $AN/analysis_a.json $AN/replay_a.json > $AN/replay_a.log 2>&1
$PY $P/compare_t41.py $AN/analysis_a.json $AN/t40g_reanalysis.json $AN/replay_a.json $AN/compare.json > $AN/compare.log 2>&1
echo POST_DONE $(date -Iseconds)
