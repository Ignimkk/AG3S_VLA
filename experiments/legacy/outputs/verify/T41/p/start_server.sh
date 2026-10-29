#!/bin/bash
# T41a server (copy of T40/p/start_server.sh; snapshot ${AG3S_PROJECT_ROOT}-t41a, autotune 0). E3b = T40 flags + --obstacle-margin 0.010
# + --no-subtask-gate (benchmark 4d090ec has the SUBTASK-g gate on by default).
# usage: start_server.sh COND PORT RECDIR    COND = E0 (--no-safe) | E3b (T34 settings, --record-constraints RECDIR/server_constraints)
source ${AG3S_PROJECT_ROOT}/outputs/verify/T41/p/env.sh
C=$1; PORT=$2; RECDIR=$3
port_ok $PORT || { echo "port $PORT not allowed (8260-8269 only, never 8123)"; exit 2; }
case $C in
  E0)  FLAGS="--no-safe"; REC="";;
  E3b) FLAGS="--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized \
--esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --no-subtask-gate"
       [ "$OBSTACLE_MARGIN" != 0 ] && FLAGS="$FLAGS --obstacle-margin $OBSTACLE_MARGIN"   # control (2026-10-05): OBSTACLE_MARGIN=0 -> flag omitted (T41a off)
       REC="--record-constraints $RECDIR/server_constraints";;
  *) echo "unknown COND $C"; exit 2;;
esac
mkdir -p $RECDIR
cd $O1 && OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS="$XLA" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 PYTHONPATH=$O1 exec $PY -m AG3S.trajopt.serve_safe \
  --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint $CKPT --model-xml $MODEL_XML --port $PORT $FLAGS $REC
