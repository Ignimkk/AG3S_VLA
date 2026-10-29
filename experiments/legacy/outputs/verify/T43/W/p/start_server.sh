#!/bin/bash
# T43W server (copy of T43/V/p/start_server.sh; snapshot ${AG3S_PROJECT_ROOT}-t43w, autotune 0). No --gripper-cover in this round (lead).
# usage: start_server.sh ARM PORT RECDIR
#   E0 = --no-safe (VLA only)
#   V1 = T34 E3b flags + --no-subtask-gate + --obstacle-margin 0.010                 (a; client c = --safe-hold-mode measured)
#   V4 = V1 + --held-body-cover                                                       (a + c + R2)
#   V3 = V4 + --servo-model --sweep-check $SWEEP (lighter sweep, env.sh)              (a + c + R2 + Q)
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/p/env.sh
C=$1; PORT=$2; RECDIR=$3
port_ok $PORT || { echo "port $PORT not allowed (8280-8289 only, never 8123)"; exit 2; }
E3B="--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized \
--esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --no-subtask-gate"
case $C in
  E0) FLAGS="--no-safe"; REC="";;
  V1) FLAGS="$E3B --obstacle-margin 0.010"; REC="--record-constraints $RECDIR/server_constraints";;
  V4) FLAGS="$E3B --obstacle-margin 0.010 --held-body-cover"; REC="--record-constraints $RECDIR/server_constraints";;
  V3) FLAGS="$E3B --obstacle-margin 0.010 --held-body-cover --servo-model --sweep-check $SWEEP"; REC="--record-constraints $RECDIR/server_constraints";;
  *) echo "unknown ARM $C"; exit 2;;
esac
mkdir -p $RECDIR
cd $O1 && OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS="$XLA" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 PYTHONPATH=$O1 exec $PY -m AG3S.trajopt.serve_safe \
  --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint $CKPT --model-xml $MODEL_XML --port $PORT $FLAGS $REC
