#!/bin/bash
# T40 A step 5 dry checks: every config x {stock pi05_infer, counterfactual transport xml}; no server, no GPU
cd ${AG3S_PROJECT_ROOT}/outputs/verify/T40/A/p
for f in ../configs/*.json; do
  b=$(basename $f .json); ep=${b%%_*}; rest=${b#*_}; seed=${rest%%_*}; kind=${rest#*_}
  prof="t40_ep${ep}_s${seed}_${kind}"
  echo "== $b stock"; MUJOCO_GL=osmesa timeout 300 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python dryrun_infer.py ../dry/${b}_stock.json $f $prof $ep $seed
  echo "== $b counterfactual"; MUJOCO_GL=osmesa timeout 300 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python dryrun_infer.py ../dry/${b}_cf.json $f $prof $ep $seed --cf-transport-xml
done
