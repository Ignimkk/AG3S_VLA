#!/bin/bash
# SUBTASK-e server (copied from T39 p/start_server.sh: E3b flags unchanged; snapshot ${AG3S_PROJECT_ROOT}-se).
# usage: start_server.sh COND PORT RECDIR    COND = E3b (gate off, V0) | E3bG (= E3b + --subtask-gate, V1)
source ${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-e/p/env.sh
C=$1; PORT=$2; RECDIR=$3
port_ok $PORT || { echo "port $PORT not allowed (8230-8239 only, never 8123)"; exit 2; }
E3B="--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized \
--esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32"
case $C in
  E3b)  FLAGS="$E3B";;
  E3bG) FLAGS="$E3B --subtask-gate";;
  *) echo "unknown COND $C"; exit 2;;
esac
REC="--record-constraints $RECDIR/server_constraints"
mkdir -p $RECDIR
cd $SD && OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS="$XLA" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 PYTHONPATH=$SD exec $PY -m AG3S.trajopt.serve_safe \
  --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint $CKPT --model-xml $MODEL_XML --port $PORT $FLAGS $REC
