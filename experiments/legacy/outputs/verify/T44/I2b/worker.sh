#!/bin/bash
# usage: worker.sh   (claims jobs from jobs.txt via mkdir lock; CPU only, nice 19)
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=1 PYTHONPATH=${AG3S_PROJECT_ROOT}
O=outputs/verify/T44/I2b; IDX=outputs/verify/T43/V11/analysis/analysis_V11AB.json
mkdir -p $O/locks
while read model key; do
  tag=$(echo $key | tr '/' '_')
  mkdir $O/locks/${model}_$tag 2>/dev/null || continue
  if [ $model = main ]; then
    ARMS="off offR strict_i2 strict_b1 strict_b0 sw swR swS_i2 swS_b1"; SA=""
  else
    ARMS="offR strict_i2 strict_b1 strict_b0 swR swS_i2 swS_b1"; SA="--server-args --links arms"
  fi
  nice -n 19 .venv-openpi-live/bin/python $O/replay_i2b.py --index $IDX --key $key --out $O/raw/${model}_$tag.json --arms $ARMS --probes $SA > $O/raw/${model}_$tag.log 2>&1
done < $O/jobs.txt
