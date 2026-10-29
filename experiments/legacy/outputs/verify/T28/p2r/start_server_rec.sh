#!/bin/bash
# T28 Phase 2R server. usage: start_server_rec.sh E0|E1|E2   (adapted from ../start_server_p2.sh, ../E0/start_server.sh)
C=$1; OUT=${AG3S_PROJECT_ROOT}/outputs/verify/T28/${C}_rec
case $C in
  E0) PORT=8201; FLAGS="--no-safe"; REC="";;
  E1) PORT=8202; FLAGS="--no-perception --no-limits --w-smooth 0 --w-continuity 0"; REC="--record-constraints $OUT/server_constraints";;
  E2) PORT=8203; FLAGS="--no-perception"; REC="--record-constraints $OUT/server_constraints";;
esac
cd ${AG3S_PROJECT_ROOT} && OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 exec .venv-openpi-live/bin/python -m AG3S.trajopt.serve_safe \
  --config pi05_rby1_randomized_pick_place_16d_lora \
  --checkpoint ${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999 \
  --model-xml robots/rby1/description/models/rby1a/mujoco/model_transport.xml --port $PORT $FLAGS $REC
