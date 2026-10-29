#!/bin/bash
# T43V server (copy of T41/c/p/start_server.sh; snapshot ${AG3S_PROJECT_ROOT}-t43v, autotune 0).
# usage: start_server.sh ARM PORT RECDIR
#   E0 = --no-safe (VLA only)
#   V0 = T34 E3b flags + --no-subtask-gate (all T41/T43 flags off)
#   V1 = V0 + --obstacle-margin 0.010                      (client: --safe-hold-mode measured)
#   V2 = V1 + --gripper-cover left --gripper-cover-max-gap 3.0 --held-body-cover --servo-model --sweep-check   (no --rest-start)
#   V3 = V1 + --held-body-cover --servo-model --sweep-check   (= V2 without --gripper-cover; added by lead)
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V/p/env.sh
C=$1; PORT=$2; RECDIR=$3
port_ok $PORT || { echo "port $PORT not allowed (8280-8289 only, never 8123)"; exit 2; }
E3B="--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized \
--esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --no-subtask-gate"
case $C in
  E0) FLAGS="--no-safe"; REC="";;
  V0) FLAGS="$E3B"; REC="--record-constraints $RECDIR/server_constraints";;
  V1) FLAGS="$E3B --obstacle-margin 0.010"; REC="--record-constraints $RECDIR/server_constraints";;
  V2) FLAGS="$E3B --obstacle-margin 0.010 --gripper-cover left --gripper-cover-max-gap 3.0 --held-body-cover --servo-model --sweep-check"
      REC="--record-constraints $RECDIR/server_constraints";;
  V3) FLAGS="$E3B --obstacle-margin 0.010 --held-body-cover --servo-model --sweep-check"   # lead 2026-10-05: V2 without --gripper-cover
      REC="--record-constraints $RECDIR/server_constraints";;
  *) echo "unknown ARM $C"; exit 2;;
esac
mkdir -p $RECDIR
cd $O1 && OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS="$XLA" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 PYTHONPATH=$O1 exec $PY -m AG3S.trajopt.serve_safe \
  --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint $CKPT --model-xml $MODEL_XML --port $PORT $FLAGS $REC
