#!/bin/bash
# T43W watcher: after every new NAME.done under runs/, run post_w.sh. Ends when launch_v5.out says 'all end V5' (+ one last pass).
R=${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/runs; last=""
while :; do
  cur=$(ls $R/*/*/*.done 2>/dev/null | sort | tr '\n' ' ')
  if [ "$cur" != "$last" ]; then echo "new done set $(date -Iseconds): $(echo $cur | wc -w)"; ${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/p/post_w.sh; last=$cur; fi
  if grep -q '^all end V5' $R/launch_v5.out 2>/dev/null; then
    cur=$(ls $R/*/*/*.done 2>/dev/null | sort | tr '\n' ' '); [ "$cur" != "$last" ] && ${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/p/post_w.sh
    echo "launch ended $(date -Iseconds)"; break; fi
  sleep 60
done
