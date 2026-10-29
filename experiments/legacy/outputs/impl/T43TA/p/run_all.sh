#!/bin/bash
# T43 TA offline: every run key of an analysis index, at most $W processes (pod rule: workers <= 4,
# working set + estimate <= 64 GiB; one process ~2.4 GiB). usage: run_all.sh INDEX PREFIX [W]
D=${AG3S_PROJECT_ROOT}/outputs/impl/T43TA; INDEX=$1; PREFIX=$2; W=${3:-3}
wset() { echo $(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat) )); }
keys=$(python3 -c "import json,sys; print('\n'.join(sorted(json.load(open('$INDEX'))['runs'])))")
for k in $keys; do
  name=$(echo "$k" | tr '/' '_'); out=$D/raw/${PREFIX}${name}.json
  [ -s "$out" ] && continue
  while [ "$(pgrep -fc 'experiments.ta_offline --index')" -ge "$W" ] || [ $(( $(wset) + 3*1024*1024*1024 )) -gt $((64*1024*1024*1024)) ]; do sleep 10; done
  nohup $D/p/run_one.sh "$INDEX" "$k" "$out" > $D/logs/${PREFIX}${name}.log 2>&1 &
  sleep 5
done
wait
