#!/bin/bash
# usage: launch.sh  (4 workers, nice 19, CPU only)
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=1
IDX=outputs/verify/T43/V11/analysis/analysis_V11AB.json
O=outputs/verify/T44/I2/raw
python3 - <<'P' > outputs/verify/T44/I2/jobs.txt
import json
d=json.load(open('${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/analysis/analysis_V11AB.json'))['runs']
keys=[k for k in d if '/H/' in k]
for k in keys: print('wt',k)
for k in keys: print('head',k)
P
run1() {
  tree=$1; key=$2; tag=$(echo $key | tr '/' '_')
  if [ $tree = wt ]; then PP=${AG3S_PROJECT_ROOT}; ARMS="off offR strict sw swR swS"; EXTRA="--probes"; else PP=${AG3S_PROJECT_ROOT}/outputs/verify/T44/I2/head; ARMS="off sw"; EXTRA=""; fi
  [ -f $O/${tree}_$tag.json ] && return
  PYTHONPATH=$PP nice -n 19 .venv-openpi-live/bin/python outputs/verify/T44/I2/replay_i2.py --index $IDX --key $key --out $O/${tree}_$tag.json --arms $ARMS $EXTRA > $O/${tree}_$tag.log 2>&1
}
export -f run1; export IDX O
cat outputs/verify/T44/I2/jobs.txt | xargs -P 4 -L 1 bash -c 'run1 $0 $1'
