#!/bin/bash
# T43W chain A relaunch (17:5x UTC): the first chain A was refused by its own V3 server-log guard (regex with '.*' across the
# multibyte path-check line did not match inside the chain; replaced by three fixed-string greps, see run_chain.sh) and stopped by hand.
#   chain A (port 8282): V3 hurdle 14 -> V3 no-obstacle 5 -> V1 hurdle 14 ; resumable (.done)
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/p/env.sh
R=$V/runs; H="$(cat $P/lists/hurdle.txt)"; B="$(cat $P/lists/clear.txt)"
echo "launch A $(date -Iseconds)"
EST_GB=28 $P/run_chain.sh V3 8282 $R/V3 "$H"; echo "A V3H rc=$? $(date -Iseconds)"; sleep 5
EST_GB=28 $P/run_chain.sh V3 8282 $R/V3 "$B"; echo "A V3B rc=$? $(date -Iseconds)"; sleep 5
EST_GB=28 $P/run_chain.sh V1 8282 $R/V1 "$H"; echo "A V1H rc=$? $(date -Iseconds)"
echo "all end A $(date -Iseconds)"
