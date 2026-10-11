#!/bin/bash
# T43 OC driver: post-release bar contacts (TK after_release list, 16 = hand 14 + arm 2); one process per snapshot slice, 3 at once, CPU only.
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=2
OUT=outputs/verify/T43/OC/raw; P=outputs/verify/T43/OC/p/oc_contact.py; PY=.venv-openpi-live/bin/python
PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v7 T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v7 nice -n 19 $PY $P $OUT/oc_V7g.json V7g/H/V7g_ep1807_s18071:261:ee_finger_l1 V7g/H/V7g_ep1807_s18073:265:ee_finger_l1 V7g/H/V7g_ep1834_s18341:237:link_right_arm_4 V7g/H/V7g_ep1925_s19251:270:ee_finger_l2 V7g/H/V7g_ep1982_s19822:275:ee_finger_l2 V7g/H/V7g_ep1982_s19823:288:ee_finger_l2 > $OUT/oc_V7g.log 2>&1 &
PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8 T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 nice -n 19 $PY $P $OUT/oc_V8a.json V8/H/V8_ep1807_s18071:314:ee_finger_l2 V8/H/V8_ep1807_s18073:270:ee_finger_l2 V8/H/V8_ep1834_s18341:249:link_right_arm_4 V8/H/V8_ep1925_s19251:279:ee_finger_l2 V8/H/V8_ep1925_s19253:279:ee_finger_l2 > $OUT/oc_V8a.log 2>&1 &
PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8 T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 nice -n 19 $PY $P $OUT/oc_V8b.json V8/H/V8_ep1967_s19672:309:ee_finger_l2 V8/H/V8_ep1968_s19681:288:ee_finger_l2 V8/H/V8_ep1982_s19823:290:ee_finger_l2 V8/H/V8_ep1995_s19952:291:ee_finger_l2 V8/H/V8_ep1995_s19953:286:ee_finger_l2 > $OUT/oc_V8b.log 2>&1 &
wait
echo done > $OUT/OC_DONE
