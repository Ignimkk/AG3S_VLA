#!/bin/bash
# attention extraction (unchanged producer AG3S/ag3s/experiments/sources/pi05_attention.py), seed 0
set -u
cd ${AG3S_PROJECT_ROOT}
CKPT=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999
OUT=${AG3S_PROJECT_ROOT}/outputs/verify/attn16d
L=outputs/live_test
run() {
  name=$1; rec=$2
  if [ -f $OUT/att_$name.npz ]; then echo "skip $name"; return; fi
  echo "=== $name $rec $(date +%T)"
  OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=${AG3S_PROJECT_ROOT} \
  .venv-openpi-live/bin/python -m AG3S.ag3s.experiments.sources.pi05_attention \
     --records $rec --checkpoint $CKPT --config pi05_rby1_randomized_pick_place_16d_lora \
     --out $OUT/att_$name.npz --noise-seeds 0
  echo "=== done $name rc=$? $(date +%T)"
}
run ep1800_headfix $L/20260925_headfix16d/records/run_0000/run_0000
run ep1800_t2 $L/20260925_t2/run_0000/run_0000
run ep1807 $L/20260925_ep1807/run_0000
run ep1850 $L/20260925_t1/ep1850/run_0000
run ep1900 $L/20260925_t1/ep1900/run_0000
echo ALLDONE
