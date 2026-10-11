#!/bin/bash
# T41a control post-processing (CPU): analysis of m0 / m10 control runs, then offline QP re-solve of B1967 seq 1..2 (T41a record and T40 G record).
AN=${AG3S_PROJECT_ROOT}/outputs/verify/T41/ctl/analysis; mkdir -p $AN; P=${AG3S_PROJECT_ROOT}/outputs/verify/T41/p; PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
export MUJOCO_GL=osmesa JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=${AG3S_PROJECT_ROOT}-t41a XLA_PYTHON_CLIENT_PREALLOCATE=false
cd ${AG3S_PROJECT_ROOT}-t41a
for v in m0 m10; do T41_ROOT=${AG3S_PROJECT_ROOT}/outputs/verify/T41/ctl/runs/$v nice -n 19 $PY $P/analyze_t41.py $AN/analysis_$v.json > $AN/analysis_$v.log 2>&1; done
SRV=$(ls -d ${AG3S_PROJECT_ROOT}/outputs/verify/T41/a/runs/E3b/server_8260_* | head -1)
F41=$($PY -c "import json;print(json.load(open('${AG3S_PROJECT_ROOT}/outputs/verify/T41/a/analysis/analysis_a.json'))['runs']['E3b/B/E3b_ep1967_s19672']['server_segment']['first'])")
F40=$($PY -c "import json;print(json.load(open('${AG3S_PROJECT_ROOT}/outputs/verify/T40/G/analysis_all.json'))['runs']['E3b/B/E3b_ep1967_s19672']['server_segment']['first'])")
export T41_SERVER_DIR=$SRV T41_MANIFEST=${AG3S_PROJECT_ROOT}/outputs/verify/T41/a/runs/E3b/B/E3b_ep1967_s19672/manifest.json
nice -n 19 $PY $P/qp_resolve_ctl.py ${AG3S_PROJECT_ROOT}/outputs/verify/T41/a/runs/E3b/B/E3b_ep1967_s19672 $F41 $AN/qp_resolve_on_T41a_record.json 2 > $AN/qp_resolve_on_T41a_record.log 2>&1
nice -n 19 $PY $P/qp_resolve_ctl.py ${AG3S_PROJECT_ROOT}/outputs/verify/T40/G/runs/E3b/B/E3b_ep1967_s19672 $F40 $AN/qp_resolve_on_T40G_record.json 2 > $AN/qp_resolve_on_T40G_record.log 2>&1
echo POST_DONE $(date -Iseconds)
