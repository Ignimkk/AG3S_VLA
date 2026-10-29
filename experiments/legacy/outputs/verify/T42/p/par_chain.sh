#!/bin/bash
# T42-1 parallel E3b chain (2026-10-05 after the session restart): waits until any serve_safe is >= 120 s old, then run_chain.sh on PORT
# with the given items (plan-v2 rep 2 / rep 3 items computed by the caller; the main launcher's later plans skip them once .done).
# usage: par_chain.sh PORT "EP:SEED:noobs ..."
source ${AG3S_PROJECT_ROOT}/outputs/verify/T42/p/env.sh
servers_settled
EST_GB=22 $P/run_chain.sh E3b $1 $T42/1/runs/E3b "$2"; echo "par rc=$? $(date -Iseconds)"
