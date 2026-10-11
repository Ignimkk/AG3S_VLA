#!/bin/bash
cd ${AG3S_PROJECT_ROOT}
E="env PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2"
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
S=outputs/verify/T44/E2d
ps -eo pid,pcpu,etime,args --sort=-pcpu | head -12 | cut -c1-200 > $S/timing/concurrent_procs_start.txt
nvidia-smi > $S/timing/nvidia_smi_start.txt
nice -n 19 $E $PY $S/e2d_timing.py --posed-frames outputs/verify/T44/E2/e2_demo_posed_frames.json --out $S/timing/timing_demo.json
echo "demo rc=$? $(date -Iseconds)"
nice -n 19 $E $PY $S/e2d_timing.py --lb-frames outputs/verify/T44/p/e2_frames.json --history 2 --out $S/timing/timing_lb.json
echo "lb rc=$? $(date -Iseconds)"
ps -eo pid,pcpu,etime,args --sort=-pcpu | head -12 | cut -c1-200 > $S/timing/concurrent_procs_end.txt
nvidia-smi > $S/timing/nvidia_smi_end.txt
