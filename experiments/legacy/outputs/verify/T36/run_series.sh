#!/bin/bash
# run cuRobo r1..r4 strictly one after another, each only after the machine is idle (wait_idle loops until idle)
cd ${AG3S_PROJECT_ROOT}/outputs/verify/T36
for k in "$@"; do
  until bash wait_idle.sh 590; do :; done
  bash run_one.sh curobo curobo_r$k | head -1
  echo "r$k done $(date -Iseconds)"
done
