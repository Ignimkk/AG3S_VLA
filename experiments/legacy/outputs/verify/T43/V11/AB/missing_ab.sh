#!/bin/bash
# V11AB: run the item(s) that have no .done (H 1995:19952, moved to failed/ after a dropped connection), same run_chain.sh, port 8286, then audit with redo_bad.sh
source ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/p/env.sh
R=$V/runs/V11AB; L=($(cat $P/lists/hurdle.txt))
EST_GB=28 $P/run_chain.sh V11AB 8286 $R "${L[6]}"; echo "missing rc=$? $(date -Iseconds)"
${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/AB/redo_bad.sh
