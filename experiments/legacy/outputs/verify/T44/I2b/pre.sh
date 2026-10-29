#!/bin/bash
cd ${AG3S_PROJECT_ROOT}
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}
O=outputs/verify/T44/I2b
git -C benchmark rev-parse --short HEAD > $O/head.txt
(cd benchmark && md5sum trajopt/config.py trajopt/linearize.py trajopt/sweep.py trajopt/serve_safe.py ag3s/config.py ag3s/fields/curobo_builder.py ag3s/runtime/pipeline.py ag3s/constraints/to_adapter.py ag3s/experiments/sources/constraint_record.py) > $O/wt_md5_start.txt
nice -n 19 .venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout --records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms --out-json $O/baseline_legacy.json > $O/baseline_legacy.out 2>&1
nice -n 19 .venv-openpi-live/bin/python -m pytest -q -p no:cacheprovider tests/trajopt tests/ag3s > $O/pytest_full.out 2>&1
echo done > $O/pre.done
