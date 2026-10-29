#!/bin/bash
# T39 S2 server (from T37/p/start_server_e3.sh and T28/p2r/start_server_rec.sh; snapshot ${AG3S_PROJECT_ROOT}-o1, autotune 0).
# usage: start_server.sh COND PORT RECDIR    COND = E0 (--no-safe) | E3b (T34 settings, --record-constraints RECDIR/server_constraints)
source ${AG3S_PROJECT_ROOT}/outputs/verify/T39/p/env.sh
C=$1; PORT=$2; RECDIR=$3
port_ok $PORT || { echo "port $PORT not allowed (8220-8229 only, never 8123)"; exit 2; }
case $C in
  E0)  FLAGS="--no-safe"; REC="";;
  E3b) FLAGS="--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized \
--esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32"
       REC="--record-constraints $RECDIR/server_constraints";;
  *) echo "unknown COND $C"; exit 2;;
esac
mkdir -p $RECDIR
cd $O1 && OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS="$XLA" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 PYTHONPATH=$O1 exec $PY -m AG3S.trajopt.serve_safe \
  --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint $CKPT --model-xml $MODEL_XML --port $PORT $FLAGS $REC
