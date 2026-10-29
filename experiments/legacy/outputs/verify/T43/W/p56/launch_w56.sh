#!/bin/bash
# T43W V5 / V6 launcher (lead: T2 arms). Waits for the snapshot-2 baseline (baseline2/judge.json matched) and for the running chains:
#   chain C (port 8282) after chain A ('all end A' in runs/launch_A.out): V6 hurdle 14 -> V6 no-obstacle 5
#   chain D (port 8283) after chain B ('all end' in runs/launch.out):     V5 hurdle 14 -> V5 no-obstacle 5
# => still <= 2 of our servers. Same guards (p56/run_chain.sh); resumable (.done).
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/W/p56/env.sh
R=$V/runs; H="$(cat $P/lists/hurdle.txt)"; B="$(cat $P/lists/clear.txt)"
echo "launch 56 $(date -Iseconds)"
until [ -f $V/baseline2/.done ]; do sleep 30; done
if ! $PY -c "import json,sys; j=json.load(open('$V/baseline2/judge.json')); sys.exit(0 if all(j[b]['matched'] for b in ('legacy','curobo')) else 1)"; then echo "REFUSED baseline2 not matched $(date -Iseconds)"; exit 3; fi
echo "baseline2 matched $(date -Iseconds)"
chainC() { until grep -q '^all end A' $R/launch_A.out 2>/dev/null; do sleep 30; done; sleep 10
           EST_GB=28 $P/run_chain.sh V6 8282 $R/V6 "$H"; echo "C V6H rc=$? $(date -Iseconds)"; sleep 5
           EST_GB=28 $P/run_chain.sh V6 8282 $R/V6 "$B"; echo "C V6B rc=$? $(date -Iseconds)"; }
chainD() { until grep -q '^all end' $R/launch.out 2>/dev/null; do sleep 30; done; sleep 40
           EST_GB=28 $P/run_chain.sh V5 8283 $R/V5 "$H"; echo "D V5H rc=$? $(date -Iseconds)"; sleep 5
           EST_GB=28 $P/run_chain.sh V5 8283 $R/V5 "$B"; echo "D V5B rc=$? $(date -Iseconds)"; }
chainC > $R/chainC.out 2>&1 &
chainD > $R/chainD.out 2>&1 &
wait
echo "all end 56 $(date -Iseconds)"
