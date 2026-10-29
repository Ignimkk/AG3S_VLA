#!/bin/bash
# SUBTASK-g: re-run the six compares with the patched compare_v0.py (skips truncated chunk files), CPU only.
source ${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-g/p/env.sh
CMP=$SD_OUT/compare; T39=${AG3S_PROJECT_ROOT}/outputs/verify/T39; SF=${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-f
cmpone() { JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $PY $P/compare_v0.py $CMP/$1.json $2 $3 > $CMP/$1.log 2>&1; echo "compare $1 rc=$? $(grep -m1 all_identical $CMP/$1.log)"; }
cmpone D_vs_SFon_ep1807_s18071 $SF/runs/E3bG/E3bG_ep1807_s18071 $SD_OUT/runs/E3bD/E3bD_ep1807_s18071
cmpone D_vs_SFon_ep1800_s18001 $SF/runs/E3bG/E3bG_ep1800_s18001 $SD_OUT/runs/E3bD/E3bD_ep1800_s18001
cmpone N_vs_T39_ep1807_s18071 $T39/s21/E3b_a/E3b_ep1807_s18071 $SD_OUT/runs/E3bN/E3bN_ep1807_s18071
cmpone N_vs_T39_ep1800_s18001 $T39/runs/E3b/E3b_ep1800_s18001 $SD_OUT/runs/E3bN/E3bN_ep1800_s18001
cmpone N_vs_SFoff_ep1807_s18071 $SF/v0/E3b/E3b_ep1807_s18071 $SD_OUT/runs/E3bN/E3bN_ep1807_s18071
cmpone N_vs_SFoff_ep1800_s18001 $SF/v0/E3b/E3b_ep1800_s18001 $SD_OUT/runs/E3bN/E3bN_ep1800_s18001
