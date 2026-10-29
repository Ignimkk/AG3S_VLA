#!/bin/bash
# SUBTASK-f V0 (copied from SUBTASK-e p/launch_v0.sh): gate off (E3b) on the snapshot, ep1807 s18071 and ep1800 s18001, one server
# (port 8230), then compare_v0.py against the T39 runs of the same (ep, seed) and against the SUBTASK-e V0 runs.
# usage: nohup setsid launch_v0.sh > v0/launch.out 2>&1 &
source ${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-f/p/env.sh
V=$SD_OUT/v0; mkdir -p $V
echo "v0 start $(date -Iseconds)"
EST_GB=22 $P/run_chain.sh E3b 8230 $V/E3b "$(cat $P/lists/V0.txt)"; echo "chain rc=$? $(date -Iseconds)"
T39R=${AG3S_PROJECT_ROOT}/outputs/verify/T39
for x in 1807:18071:$T39R/s21/E3b_a 1800:18001:$T39R/runs/E3b; do EP=${x%%:*}; r=${x#*:}; SEED=${r%%:*}; G=${r#*:}; N=E3b_ep${EP}_s${SEED}
  if [ -f $V/E3b/$N.done ]; then
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $PY $P/compare_v0.py $V/compare_$N.json $G/$N $V/E3b/$N > $V/compare_$N.log 2>&1
    echo "compare $N rc=$? $(grep -m1 all_identical $V/compare_$N.log)"
  else echo "compare $N skipped: run missing"; fi
  G2=${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-e/v0/E3b
  if [ -f $V/E3b/$N.done ]; then
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $PY $P/compare_v0.py $V/compare_vsSUBTASKe_$N.json $G2/$N $V/E3b/$N > $V/compare_vsSUBTASKe_$N.log 2>&1
    echo "compare vs SUBTASK-e v0 $N rc=$? $(grep -m1 all_identical $V/compare_vsSUBTASKe_$N.log)"
  fi
done
$PY $P/update_partial.py
echo "v0 end $(date -Iseconds)"
