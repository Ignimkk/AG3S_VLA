#!/bin/bash
# T43 TP controlled timing: tp_timing over every run key of an index, <= W processes (CPU field unless FIELD=cuda,
# then W must be 1). usage: run_timing.sh INDEX PREFIX OUTDIR W SETTINGS...
D=${AG3S_PROJECT_ROOT}/outputs/impl/T43TP; INDEX=$1; PREFIX=$2; OUTDIR=$3; W=$4; shift 4; VARS="$@"; FIELD=${FIELD:-cpu}
wset() { echo $(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat) )); }
mkdir -p $OUTDIR
if [ "$FIELD" = cuda ]; then GPUENV=""; else GPUENV="CUDA_VISIBLE_DEVICES="; fi
keys=$(python3 -c "import json; print('\n'.join(sorted(json.load(open('$INDEX'))['runs'])))")
for k in $keys; do
  name=$(echo "$k" | tr '/' '_'); out=$OUTDIR/${PREFIX}${name}.json
  [ -s "$out" ] && continue
  while [ "$(pgrep -fc "experiments.tp_timing.*$OUTDIR")" -ge "$W" ] || [ $(( $(wset) + 4096*1024*1024 )) -gt $((64*1024*1024*1024)) ]; do sleep 5; done
  ( cd ${AG3S_PROJECT_ROOT} && env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $GPUENV MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup .venv-openpi-live/bin/python -m AG3S.trajopt.experiments.tp_timing \
    --index "$INDEX" --key "$k" --field $FIELD --out "$out" --variants $VARS > $D/logs/$(basename $OUTDIR)_${PREFIX}${name}.log 2>&1 ) &
  sleep 3
done
wait
