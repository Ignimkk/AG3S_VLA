#!/bin/bash
# T44d CL server: V11AB flags (T43/V11/p/start_server.sh) + --to-fast-eval --esdf-host-mirror, + per-arm T44 flags. snapshot ${AG3S_PROJECT_ROOT}-t44cl = benchmark f891abe.
# usage: start_server.sh C0|C1a|C1b|C2 PORT RECDIR
source ${AG3S_PROJECT_ROOT}/outputs/verify/T44/CL/p/env.sh
C=$1; PORT=$2; RECDIR=$3
port_ok $PORT || { echo "port $PORT not allowed (8280-8289 only, never 8123)"; exit 2; }
is_arm $C || { echo "unknown ARM $C"; exit 2; }
LK=$(arm_links $C)
E3B="--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links $LK --target-field-policy exclude-authorized \
--esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32"
T2="--closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60"
FLAGS="$E3B --obstacle-margin 0.010 --held-body-cover $T2 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both $(arm_flags $C)"
REC="--record-constraints $RECDIR/server_constraints"
mkdir -p $RECDIR
cd $O1 && OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS="$XLA" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 PYTHONPATH=$O1 exec $PY -m AG3S.trajopt.serve_safe \
  --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint $CKPT --model-xml $MODEL_XML --port $PORT $FLAGS $REC
