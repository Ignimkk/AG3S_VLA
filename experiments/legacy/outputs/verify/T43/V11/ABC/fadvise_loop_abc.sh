#!/bin/bash
while pgrep -f "V11/ABC/run_chain_abc.sh" >/dev/null; do
  echo "$(date -Iseconds) $(python3 ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/AB/fadvise_drop.py ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11 ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V8)" >> ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/ABC/fadvise_log.txt
  sleep 300
done
