#!/bin/bash
# T43 DX-2 driver: the 8 FC (a) events with no bar in the recorded field; one process per snapshot, 4 at once, CPU only.
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=2 PYTHONPATH=${AG3S_PROJECT_ROOT}
OUT=outputs/verify/T43/DX/raw; P=outputs/verify/T43/DX/p/dx2_finger.py; PY=.venv-openpi-live/bin/python
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43w nice -n 19 $PY $P $OUT/dx2_t43w.json V4/H/V4_ep1982_s19822:271:ee_finger_l2 V4/H/V4_ep1995_s19952:307:ee_finger_l2 V4/H/V4_ep1995_s19953:362:ee_finger_l2 > $OUT/dx2_t43w.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43w3 nice -n 19 $PY $P $OUT/dx2_t43w3.json V5/H/V5_ep1982_s19822:279:ee_finger_l2 > $OUT/dx2_t43w3.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43b nice -n 19 $PY $P $OUT/dx2_t43b.json V5B/H/V5B_ep1968_s19681:321:ee_finger_l2 V5S/H/V5S_ep1925_s19253:288:ee_finger_l2 > $OUT/dx2_t43b.log 2>&1 &
T43W_SNAP=${AG3S_PROJECT_ROOT}-t43z3 nice -n 19 $PY $P $OUT/dx2_t43z3.json V5D/H/V5D_ep1968_s19681:313:ee_finger_l2 V5D/H/V5D_ep1995_s19952:308:ee_finger_l2 > $OUT/dx2_t43z3.log 2>&1 &
wait
echo done > $OUT/DX2_DONE
