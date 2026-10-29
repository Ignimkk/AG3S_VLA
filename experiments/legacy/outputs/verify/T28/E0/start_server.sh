#!/bin/bash
# T28 P1-1 E0: bare policy server (--no-safe), port 8201
cd ${AG3S_PROJECT_ROOT} && OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 exec .venv-openpi-live/bin/python -m AG3S.trajopt.serve_safe \
  --config pi05_rby1_randomized_pick_place_16d_lora \
  --checkpoint ${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999 \
  --model-xml robots/rby1/description/models/rby1a/mujoco/model_transport.xml --port 8201 --no-safe
