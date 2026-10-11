#!/bin/bash
# T41c post-processing (CPU only, verifier): analyze_t41 (T41/p, T41_ROOT=T41/c/runs) -> hold_c (physics replay, T41c runs) -> compare_c -> partial.
# usage: post_c.sh [--figs]     (idempotent; hold_c skips runs already in hold_c.json)
AN=${AG3S_PROJECT_ROOT}/outputs/verify/T41/c/analysis; P=${AG3S_PROJECT_ROOT}/outputs/verify/T41/p; CP=${AG3S_PROJECT_ROOT}/outputs/verify/T41/c/p; PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
export MUJOCO_GL=osmesa JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
exec 9>$CP/.post.lock; flock 9
RUNS=$(ls -d ${AG3S_PROJECT_ROOT}/outputs/verify/T41/c/runs/E3b/{B,C,D}/E3b_ep*_s* 2>/dev/null | while read d; do [ -f $d.done ] && echo $d; done)
[ -z "$RUNS" ] && { echo "no done runs"; exit 0; }
( cd ${AG3S_PROJECT_ROOT}-t41a && T41_ROOT=${AG3S_PROJECT_ROOT}/outputs/verify/T41/c/runs PYTHONPATH=${AG3S_PROJECT_ROOT}-t41a nice -n 19 $PY $P/analyze_t41.py $AN/analysis_c.json > $AN/analysis_c.log 2>&1 ) || echo "analyze rc=$?"
( cd $AN && nice -n 19 $PY $CP/hold_c.py $AN/hold_c.json $RUNS >> $AN/hold_c.log 2>&1 ) || echo "hold rc=$?"
$PY $CP/compare_c.py $AN/compare_c.json > $AN/compare_c.log 2>&1 || echo "compare rc=$?"
$PY $CP/partial_c.py >> $AN/partial_c.log 2>&1 || echo "partial rc=$?"
[ "$1" = --figs ] && { T41_FIGOUT=${T41_FIGOUT:-AG3S/ag3s/docs/figures/t41} $PY $CP/fig_t41c.py $AN/compare_c.json $FIGFLAGS > $AN/fig.log 2>&1 || echo "fig rc=$?"; }
echo POST_DONE $(date -Iseconds)
