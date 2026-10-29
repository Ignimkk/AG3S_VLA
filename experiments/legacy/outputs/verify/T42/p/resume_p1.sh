#!/bin/bash
# T42-1 one-off (2026-10-05, after the lead's memory-rule change): the p1 chain1 subshell of the first launcher (waiting under the
# old memory.current rule) was killed; this starts p1 chain1 now under the new rule, then waits for the first launcher
# (pid OLD, which stops after its chain0 with "p1 INCOMPLETE") and re-runs launch_t42.sh p1 (resumes: done runs are skipped).
source ${AG3S_PROJECT_ROOT}/outputs/verify/T42/p/env.sh
R=$T42/1/runs; OLD=$1; ER=$(( 11 * 1024 * 1024 * 1024 ))
servers_settled
t0=$(date +%s); while [ $(( $(mem_ws) + ER )) -gt $MEM_LIMIT_BYTES ]; do echo "mem wait working_set=$(mem_ws) $(date -Iseconds)"; [ $(( $(date +%s) - t0 )) -ge 7200 ] && break; sleep 60; servers_settled; done
echo "p1 chain1 (resume) start working_set=$(mem_ws) $(date -Iseconds)"
EST_GB=22 $P/run_chain.sh E3b 8251 $R/E3b "$(cat $P/lists/p1_1.txt)"; echo "p1 chain1 (resume) rc=$? $(date -Iseconds)"
while kill -0 $OLD 2>/dev/null; do sleep 30; done
echo "first launcher exited $(date -Iseconds); relaunch p1"
exec $P/launch_t42.sh p1
