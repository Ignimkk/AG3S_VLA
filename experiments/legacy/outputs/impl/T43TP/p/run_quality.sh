#!/bin/bash
# T43 TP quality pass (ta_offline, reset mode unless KEEP=1): every run key of an index, <= W processes,
# working set + 3.5 GiB per new process <= 64 GiB. usage: run_quality.sh INDEX PREFIX OUTDIR W VARIANTS...
D=${AG3S_PROJECT_ROOT}/outputs/impl/T43TP; INDEX=$1; PREFIX=$2; OUTDIR=$3; W=$4; shift 4; VARS="$@"
wset() { echo $(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat) )); }
EXTRA=""; [ "${KEEP:-0}" = 1 ] && EXTRA="--keep-state"
keys=$(python3 -c "import json; print('\n'.join(sorted(json.load(open('$INDEX'))['runs'])))")
for k in $keys; do
  name=$(echo "$k" | tr '/' '_'); out=$OUTDIR/${PREFIX}${name}.json
  [ -s "$out" ] && continue
  while [ "$(pgrep -fc "experiments.ta_offline --index.*$OUTDIR")" -ge "$W" ] || [ $(( $(wset) + 3584*1024*1024 )) -gt $((64*1024*1024*1024)) ]; do sleep 10; done
  ( cd ${AG3S_PROJECT_ROOT} && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup .venv-openpi-live/bin/python -m AG3S.trajopt.experiments.ta_offline \
    --index "$INDEX" --key "$k" --contacts AG3S/ag3s/docs/handoff/T43HX.verify.json --out "$out" --ref-variant A16qB20 $EXTRA \
    --variants $VARS > $D/logs/$(basename $OUTDIR)_${PREFIX}${name}.log 2>&1 ) &
  sleep 5
done
wait
