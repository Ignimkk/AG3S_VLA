#!/bin/bash
# T43V9 V9 server = V8 flags (T43/V8/p/start_server.sh) + --servo-model (no --sweep-check); V8 was V7g flags (T43/V7g/p/start_server.sh) + --finger-cover both (overflow default 2.5 mm); snapshot ${AG3S_PROJECT_ROOT}-t43v8 = benchmark 786cf1b. usage: start_server.sh V9 PORT RECDIR
#   V5 = V4 (a + c + --held-body-cover) + --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60   (V6 dropped by lead)
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V9/p/env.sh
C=$1; PORT=$2; RECDIR=$3
port_ok $PORT || { echo "port $PORT not allowed (8280-8289 only, never 8123)"; exit 2; }
E3B="--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized \
--esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32"   # V5 flags minus --no-subtask-gate (gate ON, main default)
T2="--closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60"   # T43 T2 (d6d95b0); --refused-identity-release and --held-free-observed-mm off (lead)
case $C in
  V9) FLAGS="$E3B --obstacle-margin 0.010 --held-body-cover $T2 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --servo-model"; REC="--record-constraints $RECDIR/server_constraints";;
  *) echo "unknown ARM $C (p serves V9 only)"; exit 2;;
esac
mkdir -p $RECDIR
cd $O1 && OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS="$XLA" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 PYTHONPATH=$O1 exec $PY -m AG3S.trajopt.serve_safe \
  --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint $CKPT --model-xml $MODEL_XML --port $PORT $FLAGS $REC
