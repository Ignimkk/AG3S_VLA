#!/bin/bash
# usage: run_condition.sh E0|E1|E2 — starts server, waits for listening, runs 6 clients sequentially, kills server.
C=$1; OUT=${AG3S_PROJECT_ROOT}/outputs/verify/T28/${C}_rec; D=${AG3S_PROJECT_ROOT}/outputs/verify/T28/p2r
case $C in E0) PORT=8201;; E1) PORT=8202;; E2) PORT=8203;; esac
date -Iseconds > $OUT/server.start
nohup $D/start_server_rec.sh $C > $OUT/serve_safe.log 2>&1 &
SPID=$!; echo $SPID > $OUT/server.pid
until grep -q "server listening on 0.0.0.0:$PORT" $OUT/serve_safe.log; do
  if ! kill -0 $SPID 2>/dev/null; then echo "server died" >> $OUT/condition.log; exit 1; fi; sleep 5; done
echo "server up $(date -Iseconds)" >> $OUT/condition.log
for EP in 1807 1800 1808; do for K in 1 2; do
  echo "start ${C}_ep${EP}_r${K} $(date -Iseconds)" >> $OUT/condition.log
  $D/run_client_rec.sh $C $EP $K
  echo "end ${C}_ep${EP}_r${K} $(date -Iseconds) $(tail -1 $OUT/${C}_ep${EP}_r${K}.client.log)" >> $OUT/condition.log
done; done
kill $SPID; sleep 5; pkill -f "serve_safe.*--port $PORT" ; sleep 3
echo "server killed $(date -Iseconds) alive=$(pgrep -f "serve_safe.*--port $PORT" | wc -l)" >> $OUT/condition.log
date -Iseconds > $OUT/server.end
echo DONE >> $OUT/condition.log
