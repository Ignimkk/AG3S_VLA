#!/bin/bash
# T43W blind-round watcher: after every new NAME.done under runs_blind/, run p/post_b.sh. Ends with 'all end blind' in runs_blind/launch.out.
R=${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/runs_blind; last=""
while :; do
  cur=$(ls $R/*/*/*.done 2>/dev/null | sort | tr '\n' ' ')
  if [ "$cur" != "$last" ]; then echo "new done set $(date -Iseconds): $(echo $cur | wc -w)"; ${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/p/post_b.sh; last=$cur; fi
  if grep -q '^all end blind' $R/launch.out 2>/dev/null; then
    cur=$(ls $R/*/*/*.done 2>/dev/null | sort | tr '\n' ' '); [ "$cur" != "$last" ] && ${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/p/post_b.sh
    echo "launch ended $(date -Iseconds)"; break; fi
  sleep 60
done
