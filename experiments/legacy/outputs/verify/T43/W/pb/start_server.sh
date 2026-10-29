#!/bin/bash
# T43W blind-round server (copy of T43/W/p5/start_server.sh; snapshot ${AG3S_PROJECT_ROOT}-t43b). usage: start_server.sh ARM PORT RECDIR
#   E0B = E0 (--no-safe); V4B = V4 (a + c + --held-body-cover); V5B / V5S = V5 (V4 + T2 identity flags). Server flags do not depend on blind / visible:
#   the blind mode is the client flag --obstacle-policy-hidden (run_client.sh: E0B V4B V5B on, V5S off).
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/pb/env.sh
C=$1; PORT=$2; RECDIR=$3
port_ok $PORT || { echo "port $PORT not allowed (8280-8289 only, never 8123)"; exit 2; }
E3B="--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized \
--esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --no-subtask-gate"
T2="--closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60"   # T43 T2 (d6d95b0); --refused-identity-release and --held-free-observed-mm off (lead)
case $C in
  E0B)     FLAGS="--no-safe"; REC="";;
  V4B)     FLAGS="$E3B --obstacle-margin 0.010 --held-body-cover"; REC="--record-constraints $RECDIR/server_constraints";;
  V5B|V5S) FLAGS="$E3B --obstacle-margin 0.010 --held-body-cover $T2"; REC="--record-constraints $RECDIR/server_constraints";;
  *) echo "unknown ARM $C (pb serves E0B V4B V5B V5S)"; exit 2;;
esac
mkdir -p $RECDIR
cd $O1 && OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS="$XLA" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 PYTHONPATH=$O1 exec $PY -m AG3S.trajopt.serve_safe \
  --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint $CKPT --model-xml $MODEL_XML --port $PORT $FLAGS $REC
