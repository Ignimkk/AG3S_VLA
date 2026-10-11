#!/bin/bash
# T40 A sensitivity: binding.py over the 22 pairs, 8 workers (CPU)
cd ${AG3S_PROJECT_ROOT}/outputs/verify/T40/A/p
TAGS=$(python3 -c "import json;print(' '.join(str(p['episode'])+'_'+str(p['seed']) for p in json.load(open('../raw/carry_paths_22pairs.json'))['pairs']))")
echo $TAGS | tr ' ' '\n' | xargs -P 8 -I{} sh -c "MUJOCO_GL=osmesa OMP_NUM_THREADS=1 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python binding.py {} > ../raw/binding_{}.log 2>&1"
cat ../raw/binding_*.log > ../raw/binding_all.log; echo done > ../raw/binding.done
