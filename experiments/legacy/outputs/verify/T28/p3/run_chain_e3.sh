#!/bin/bash
# usage: run_chain_e3.sh COND PORT "EP:K EP:K ..." — starts server on PORT, waits, runs listed runs sequentially, kills server.
C=$1; PORT=$2; LIST=$3; OUT=${AG3S_PROJECT_ROOT}/outputs/verify/T28/${C}_rec; D=${AG3S_PROJECT_ROOT}/outputs/verify/T28/p3
SL=$OUT/serve_safe_$PORT.log; CL=$OUT/chain_$PORT.log
date -Iseconds > $OUT/server_$PORT.start
nohup $D/start_server_e3.sh $C $PORT > $SL 2>&1 &
SPID=$!; echo $SPID > $OUT/server_$PORT.pid
until grep -q "server listening on 0.0.0.0:$PORT" $SL; do
  if ! kill -0 $SPID 2>/dev/null; then echo "server died $(date -Iseconds)" >> $CL; exit 1; fi; sleep 5; done
echo "server up $(date -Iseconds)" >> $CL
if [ -n "$HOLD_AFTER_UP" ]; then touch $OUT/server_$PORT.ready; while [ ! -f $OUT/server_$PORT.go ]; do sleep 3; done; fi
for RK in $LIST; do EP=${RK%%:*}; K=${RK##*:}
  echo "start ${C}_ep${EP}_r${K} $(date -Iseconds)" >> $CL
  $D/run_client_e3.sh $C $PORT $EP $K
  echo "end ${C}_ep${EP}_r${K} $(date -Iseconds) $(tail -1 $OUT/${C}_ep${EP}_r${K}.client.log)" >> $CL
done
kill $SPID; sleep 5; pkill -f "serve_safe.*--port $PORT" ; sleep 3
echo "server killed $(date -Iseconds) alive=$(pgrep -f "serve_safe.*--port $PORT" | wc -l)" >> $CL
date -Iseconds > $OUT/server_$PORT.end
echo DONE >> $CL
