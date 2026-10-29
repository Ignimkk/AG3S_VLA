#!/bin/bash
# T43W post-processing (CPU only): analyze_t43w (cached per run) -> hold_v (physics replay, H runs, cached) -> partial_v (T43W.verify.partial.json numbers).
# usage: post_w.sh   (idempotent; flock)
AN=${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/analysis; P=${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/p; PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
export MUJOCO_GL=osmesa JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
exec 9>$P/.post.lock; flock 9
( cd ${AG3S_PROJECT_ROOT}-t43w && PYTHONPATH=${AG3S_PROJECT_ROOT}-t43w nice -n 19 $PY $P/analyze_t43w.py $AN/analysis.json > $AN/analysis.log 2>&1 ) || echo "analyze rc=$?"
RUNS=$(ls -d ${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/runs/*/H/*_ep*_s* 2>/dev/null | while read d; do [ -f $d.done ] && echo $d; done)
[ -n "$RUNS" ] && { ( cd $AN && nice -n 19 $PY $P/hold_w.py $AN/hold_w.json $RUNS >> $AN/hold_w.log 2>&1 ) || echo "hold rc=$?"; }
$PY $P/partial_w.py >> $AN/partial_w.log 2>&1 || echo "partial rc=$?"
echo POST_DONE $(date -Iseconds)
