#!/bin/bash
# T41c watcher: after every new NAME.done under T41/c/runs, run post_c.sh (partial numbers). Ends when the chain log says DONE (+ one last pass).
# usage: nohup setsid watch_c.sh > T41/c/analysis/watch.out 2>&1 &
R=${AG3S_PROJECT_ROOT}/outputs/verify/T41/c/runs/E3b; last=""
while :; do
  cur=$(ls $R/*/*.done 2>/dev/null | sort | tr '\n' ' ')
  if [ "$cur" != "$last" ]; then echo "new done set $(date -Iseconds): $(echo $cur | wc -w)"; ${AG3S_PROJECT_ROOT}/outputs/verify/T41/c/p/post_c.sh; last=$cur; fi
  grep -q '^DONE' $R/chain_8270.log 2>/dev/null && { echo "chain DONE $(date -Iseconds)"; break; }
  grep -q 'REFUSED\|server died\|chain stops' $R/chain_8270.log 2>/dev/null && { echo "chain stopped $(date -Iseconds)"; break; }
  sleep 60
done
