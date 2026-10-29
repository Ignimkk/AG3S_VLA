#!/bin/bash
# runs the T40 H sweep over the 5 false-zero runs, at most 4 at a time; logs to h_sweep_<run>.log
S=${AG3S_PROJECT_ROOT}/outputs/verify/T40/H
cd ${AG3S_PROJECT_ROOT}
printf "%s\n" E3b/C/E3b_ep1995_s19952 E3b/D/E3b_ep1967_s19672 E3b/D/E3b_ep1982_s19822 E3b/D/E3b_ep1995_s19952 E0/D/E0_ep1967_s19672 | \
  xargs -P 4 -I{} bash -c 'JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}:integration/rby1/manipulation/src timeout 3000 .venv-openpi-live/bin/python '$S'/h_sweep.py {} 2>&1 | grep -v Warn > '$S'/h_sweep_$(echo {} | tr / _).log'
