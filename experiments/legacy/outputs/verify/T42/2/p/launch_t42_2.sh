#!/bin/bash
# T42-2 launcher (verifier, 2026-10-05). Two chains in parallel: E3b on 8251 (T34 settings + --viz-avoidance-arrow), E0 on 8250;
# the E0 chain starts after the E3b server is >= 120 s old (servers_settled). Items from lists/<NAME>_<COND>.txt
# (EP:SEED:KIND:CONFIG:PROFILE). Memory guard: working set + EST <= 64 GiB before each chain (run_chain.sh re-checks before each client).
# usage: nohup launch_t42_2.sh NAME > ${AG3S_PROJECT_ROOT}/outputs/verify/T42/2/runs/launch_NAME.out 2>&1 &
source ${AG3S_PROJECT_ROOT}/outputs/verify/T42/2/p/env.sh
R=$T42/2/runs; mkdir -p $R; NAME=$1
memok() { local need=$1 t0=$(date +%s); while :; do servers_settled; cur=$(mem_ws); \
  [ $(( cur + need )) -le $MEM_LIMIT_BYTES ] && { echo "mem ok working_set=$cur need=$need $(date -Iseconds)"; return 0; }; \
  echo "mem wait working_set=$cur need=$need $(date -Iseconds)"; [ $(( $(date +%s) - t0 )) -ge 7200 ] && return 1; sleep 60; done; }
echo "t42-2 launch $NAME start $(date -Iseconds)"
( memok $(( 11 * 1024 ** 3 )) && EST_GB=22 $P/run_chain.sh E3b 8251 $R/E3b "$(cat $P/lists/${NAME}_E3b.txt)"; echo "E3b rc=$? $(date -Iseconds)" ) & p1=$!
sleep 150
( memok $(( 8 * 1024 ** 3 )) && EST_GB=10 $P/run_chain.sh E0 8250 $R/E0 "$(cat $P/lists/${NAME}_E0.txt)"; echo "E0 rc=$? $(date -Iseconds)" ) & p2=$!
wait $p1; wait $p2
$PY $P/update_partial.py >> $R/update_partial.log 2>&1
echo "t42-2 launch $NAME end $(date -Iseconds)"
