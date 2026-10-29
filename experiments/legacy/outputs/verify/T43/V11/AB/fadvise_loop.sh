#!/bin/bash
# every 5 min while a V11 run_chain exists: fadvise DONTNEED on the verifier's own output dirs (clean pages only), log wset before/after
while pgrep -f "T43/V11/p/run_chain.sh" >/dev/null; do
  echo "$(date -Iseconds) $(python3 ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/AB/fadvise_drop.py ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11 ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V8)" >> ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/AB/fadvise_log.txt
  sleep 300
done
