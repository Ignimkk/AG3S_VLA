#!/bin/bash
# T43 DX-1 driver: one process per client snapshot, <= 4 at once, CPU only.
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=2 PYTHONPATH=${AG3S_PROJECT_ROOT}
OUT=outputs/verify/T43/DX/raw
keys_for() {  # snapshot -> ct.json keys with an apple carry contact, arm on that snapshot
python3 - "$1" <<'PY'
import json, sys
sys.path.insert(0, "outputs/verify/T43/DX/p")
SNAP_OF = {"V1": "${AG3S_PROJECT_ROOT}-t43w", "V3": "${AG3S_PROJECT_ROOT}-t43w", "V4": "${AG3S_PROJECT_ROOT}-t43w", "E0": "${AG3S_PROJECT_ROOT}-t43w", "V5": "${AG3S_PROJECT_ROOT}-t43w3", "V5S": "${AG3S_PROJECT_ROOT}-t43b", "V5B": "${AG3S_PROJECT_ROOT}-t43b", "V5D": "${AG3S_PROJECT_ROOT}-t43z3", "V7g": "${AG3S_PROJECT_ROOT}-t43v7"}
ct = json.load(open("outputs/verify/T43/CT/raw/ct.json"))["runs"]
print(" ".join(k for k, r in ct.items() if SNAP_OF[k.split("/")[0]] == sys.argv[1] and any(c["kind"] == "apple" and c["phase"] == "carry" for c in r["contacts"])))
PY
}
run() { snap=${AG3S_PROJECT_ROOT}-$1; T43W_SNAP=$snap nice -n 19 .venv-openpi-live/bin/python outputs/verify/T43/DX/p/dx1_apple.py $OUT/dx1_$1.json $(keys_for $snap) > $OUT/dx1_$1.log 2>&1; }
run t43w & run t43w3 & run t43b & run t43z3 & wait
run t43v7
echo done > $OUT/DX1_DONE
