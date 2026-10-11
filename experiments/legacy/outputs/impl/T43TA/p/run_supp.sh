#!/bin/bash
# T43 TA: the supplementary C variants (+ base) on runs whose raw/<name>.json lacks them. usage: run_supp.sh INDEX [W]
D=${AG3S_PROJECT_ROOT}/outputs/impl/T43TA; INDEX=$1; W=${2:-3}
wset() { echo $(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat) )); }
keys=$(python3 -c "import json; print('\n'.join(sorted(json.load(open('$INDEX'))['runs'])))")
for k in $keys; do
  name=$(echo "$k" | tr '/' '_'); main=$D/raw/${name}.json; out=$D/raw/supp_${name}.json
  [ -s "$out" ] && continue
  while [ ! -s "$main" ]; do sleep 20; done
  python3 -c "import json,sys; d=json.load(open('$main')); sys.exit(0 if 'A16B20Cr1w' in d['variants'] else 1)" && continue
  while [ "$(pgrep -fc 'experiments.ta_offline --index')" -ge "$W" ] || [ $(( $(wset) + 3*1024*1024*1024 )) -gt $((64*1024*1024*1024)) ]; do sleep 10; done
  OUT=$out; cd ${AG3S_PROJECT_ROOT}
  JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup .venv-openpi-live/bin/python -m AG3S.trajopt.experiments.ta_offline \
    --index "$INDEX" --key "$k" --contacts AG3S/ag3s/docs/handoff/T43HX.verify.json --out "$OUT" \
    --variants base A16B20Cr1w A16B20Cr2w A16B20Cs1w A16B50Cr1w A24B20Cr1w > $D/logs/supp_${name}.log 2>&1 &
  sleep 5
done
wait
