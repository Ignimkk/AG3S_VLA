#!/bin/bash
# T43V watcher: after every new NAME.done under runs/, run post_v.sh. Ends when launch_v3.out says 'all end V3' (+ one last pass).
R=${AG3S_PROJECT_ROOT}/outputs/verify/T43/V/runs; last=""
while :; do
  cur=$(ls $R/*/*/*.done 2>/dev/null | sort | tr '\n' ' ')
  if [ "$cur" != "$last" ]; then echo "new done set $(date -Iseconds): $(echo $cur | wc -w)"; ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V/p/post_v.sh; last=$cur; fi
  if grep -q '^all end V3' $R/launch_v3.out 2>/dev/null; then
    cur=$(ls $R/*/*/*.done 2>/dev/null | sort | tr '\n' ' '); [ "$cur" != "$last" ] && ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V/p/post_v.sh
    echo "launch ended $(date -Iseconds)"; break; fi
  sleep 60
done
