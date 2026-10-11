#!/bin/bash
# T43V post-processing (CPU only): analyze_t43v (cached per run) -> hold_v (physics replay, H runs, cached) -> partial_v (T43V.verify.partial.json numbers).
# usage: post_v.sh   (idempotent; flock)
AN=${AG3S_PROJECT_ROOT}/outputs/verify/T43/V/analysis; P=${AG3S_PROJECT_ROOT}/outputs/verify/T43/V/p; PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
export MUJOCO_GL=osmesa JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
exec 9>$P/.post.lock; flock 9
( cd ${AG3S_PROJECT_ROOT}-t43v && PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v nice -n 19 $PY $P/analyze_t43v.py $AN/analysis.json > $AN/analysis.log 2>&1 ) || echo "analyze rc=$?"
RUNS=$(ls -d ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V/runs/*/H/*_ep*_s* 2>/dev/null | while read d; do [ -f $d.done ] && echo $d; done)
[ -n "$RUNS" ] && { ( cd $AN && nice -n 19 $PY $P/hold_v.py $AN/hold_v.json $RUNS >> $AN/hold_v.log 2>&1 ) || echo "hold rc=$?"; }
$PY $P/partial_v.py >> $AN/partial_v.log 2>&1 || echo "partial rc=$?"
echo POST_DONE $(date -Iseconds)
