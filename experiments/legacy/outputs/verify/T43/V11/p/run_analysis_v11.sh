#!/bin/bash
# T43 V11 analysis chain for ARM (CPU only; run after the chains ended and the replays are done). usage: run_analysis_v11.sh ARM
ARM=${1:-V11A}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$O1"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa T43W_SNAP="$O1" V11_ARM=$ARM OMP_NUM_THREADS=2
mkdir -p "$V/raw" "$V/analysis"
$PY $P/analyze_v11.py $V/analysis/analysis_$ARM.json > $V/raw/analyze_$ARM.log 2>&1 || echo analyze FAILED
$PY $P/agg_ct_v11.py $V/ct_$ARM.json > $V/raw/agg_ct_$ARM.log 2>&1 || echo agg_ct FAILED
$PY $P/agg_v11.py $V/agg_$ARM.json > $V/raw/agg_$ARM.log 2>&1 || echo agg FAILED
$PY $P/extra_v11.py $V/extra_$ARM.json > $V/raw/extra_$ARM.log 2>&1 || echo extra FAILED
$PY $P/sharp_v11.py $V/sharp_$ARM.json > $V/raw/sharp_$ARM.log 2>&1 || echo sharp FAILED
$PY $P/xm_cross_v11.py $ARM $V/raw/cross_$ARM.json > $V/raw/cross_$ARM.log 2>&1 || echo xm_cross FAILED
$PY $P/xm_agg_v11.py $V/xm_summary_$ARM.json > $V/raw/xm_agg_$ARM.log 2>&1 || echo xm_agg FAILED
echo analysis done $ARM
