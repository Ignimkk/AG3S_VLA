#!/bin/bash
# T28 P3 0f — regression baseline (skill regression-baseline) run ALONE, sequentially: working tree, then HEAD and variant copies.
B=${AG3S_PROJECT_ROOT}/outputs/verify/T28/p3/0f
ARGS="--records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms"
# (WT already measured alone at 03:37:53; this re-run covers the copies only)
for v in WTcopy infl005 min0 nocover nocover_nofinger nocover_nofinger_infl005_min0; do
  echo "start $v $(date -Iseconds)" >> $B/run_0f.log
  cd $B/tree_$v && AG3S_ASSET_ROOT=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid AG3S_CLIENT_SCRIPT=integration/rby1/bringup/pi05_infer.py MUJOCO_GL=osmesa PYTHONPATH=$B/tree_$v:${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -c "import benchmark,sys; print('benchmark from', benchmark.__file__)" > $B/base_$v.log 2>&1
  cd $B/tree_$v && AG3S_ASSET_ROOT=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid AG3S_CLIENT_SCRIPT=integration/rby1/bringup/pi05_infer.py MUJOCO_GL=osmesa PYTHONPATH=$B/tree_$v:${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -m AG3S.trajopt.experiments.esdf_rollout $ARGS --out-json $B/base_$v.json >> $B/base_$v.log 2>&1
  echo "end $v $(date -Iseconds) exit=$?" >> $B/run_0f.log
done
echo DONE >> $B/run_0f.log
