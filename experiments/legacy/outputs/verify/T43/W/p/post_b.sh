#!/bin/bash
# T43W blind-round post-processing (CPU only): same scripts as post_w.sh on runs_blind/ with the blind snapshot's client / scene code.
AN=${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/analysis_blind; P=${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/p; PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
mkdir -p $AN
export MUJOCO_GL=osmesa JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
export T43W_ROOT=${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/runs_blind T43W_SNAP=${AG3S_PROJECT_ROOT}-t43b T43W_AN=$AN T43W_RUNS=runs_blind T43W_PARTIAL_KEY=numbers_blind_partial
exec 9>$P/.post_b.lock; flock 9
( cd ${AG3S_PROJECT_ROOT}-t43b && PYTHONPATH=${AG3S_PROJECT_ROOT}-t43b nice -n 19 $PY $P/analyze_t43w.py $AN/analysis.json > $AN/analysis.log 2>&1 ) || echo "analyze rc=$?"
RUNS=$(ls -d $T43W_ROOT/*/H/*_ep*_s* 2>/dev/null | while read d; do [ -f $d.done ] && echo $d; done)
[ -n "$RUNS" ] && { ( cd $AN && nice -n 19 $PY $P/hold_w.py $AN/hold_w.json $RUNS >> $AN/hold_w.log 2>&1 ) || echo "hold rc=$?"; }
$PY $P/partial_w.py >> $AN/partial_w.log 2>&1 || echo "partial rc=$?"
echo POST_DONE $(date -Iseconds)
