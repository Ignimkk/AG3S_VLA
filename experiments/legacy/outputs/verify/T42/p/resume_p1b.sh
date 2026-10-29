#!/bin/bash
# T42-1 one-off (2026-10-05 07:52 UTC): the first p1 chain0 (old run_chain.sh, old memory.current rule, waiting since 07:23) was
# stopped (run_chain + its server killed) together with resume_p1.sh; this re-runs p1 chain0 under the new rule on 8250, waits for the
# p1 chain1 run_chain (pid CH1, started by resume_p1.sh), then runs launch_t42.sh p1 (p1 all done -> p2a, p2b, p3).
source ${AG3S_PROJECT_ROOT}/outputs/verify/T42/p/env.sh
R=$T42/1/runs; CH1=$1; ER=$(( 11 * 1024 * 1024 * 1024 ))
servers_settled
t0=$(date +%s); while [ $(( $(mem_ws) + ER )) -gt $MEM_LIMIT_BYTES ]; do echo "mem wait working_set=$(mem_ws) $(date -Iseconds)"; [ $(( $(date +%s) - t0 )) -ge 7200 ] && break; sleep 60; servers_settled; done
echo "p1 chain0 (resume) start working_set=$(mem_ws) $(date -Iseconds)"
EST_GB=22 $P/run_chain.sh E3b 8250 $R/E3b "$(cat $P/lists/p1_0.txt)"; echo "p1 chain0 (resume) rc=$? $(date -Iseconds)"
while kill -0 $CH1 2>/dev/null; do sleep 30; done
echo "chain1 exited $(date -Iseconds); launch_t42.sh p1"
exec $P/launch_t42.sh p1
