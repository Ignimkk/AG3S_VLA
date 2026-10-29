#!/bin/bash
# T40 A stage 1: placement search over the 22 both-success T39 pairs, 8 worker processes (CPU only, no GPU).
# usage: run_stage1.sh [--replay] [TAG ...]   (TAG = <episode>_<seed>; default = all 22 pairs)
cd ${AG3S_PROJECT_ROOT}/outputs/verify/T40/A/p
MODE=""
if [ "$1" = "--replay" ]; then MODE="--replay"; shift; fi
if [ $# -gt 0 ]; then TAGS="$*"; else
TAGS=$(python3 -c "import json;print(' '.join(str(p['episode'])+'_'+str(p['seed']) for p in json.load(open('../raw/carry_paths_22pairs.json'))['pairs']))")
fi
PFX=search_stage1; [ -n "$MODE" ] && PFX=replay_stage2
echo $TAGS | tr ' ' '\n' | xargs -P 8 -I{} sh -c "MUJOCO_GL=osmesa OMP_NUM_THREADS=1 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python search.py $MODE {} > ../raw/${PFX}_{}.log 2>&1"
cat ../raw/${PFX}_*.log > ../raw/${PFX}.log
echo done > ../raw/${PFX}.done
