#!/bin/bash
# T30 (copied from T28/p3, paths only) E3 server. usage: start_server_e3.sh COND PORT   (COND = E3a|E3b; OUT dir per port so parallel chains have separate records)
C=$1; PORT=$2; OUT=${AG3S_PROJECT_ROOT}/outputs/verify/T30/${C}_rec/server_$PORT
mkdir -p $OUT
cd ${AG3S_PROJECT_ROOT} && OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 exec .venv-openpi-live/bin/python -m AG3S.trajopt.serve_safe \
  --config pi05_rby1_randomized_pick_place_16d_lora \
  --checkpoint ${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999 \
  --model-xml robots/rby1/description/models/rby1a/mujoco/model_transport.xml --port $PORT \
  --esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized \
  --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 \
  --record-constraints $OUT/server_constraints
