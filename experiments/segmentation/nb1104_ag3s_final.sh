#!/usr/bin/env bash
# nb1104 (C6): AG3S final Panda configuration (AG3S-final-panda-v3-e0to) with the re-made HSR6-Panda probe.
# Runs inside nb1104_queue.sh as a SOLO line (TrajOpt closed loop: no other GPU job at the same time).
# Refuses to run unless the re-made probe passed the SL6-B VB3 gate fixed before (fold macro-F1 >= 0.70 and grasp recall >= 0.6).
# usage: nb1104_ag3s_final.sh LEVEL(I|II) [extra launch_libero_paired args]
set -u
LV=$1; shift
cd "$(dirname "$0")/../.."
source scripts/setup/env.sh >/dev/null 2>&1
B="$AG3S_STORAGE_ROOT/results/benchmark"
L=$(ls -d "$B"/*_nb1104-hsr6-panda-labels-remade 2>/dev/null | tail -1)
J="$L/sl6b_hsr6_panda.json"
for i in $(seq 1 180); do [ -f "$J" ] && break; sleep 20; done
[ -f "$J" ] || { echo "no re-made probe result ($J)"; exit 3; }
PROBE=$(.venv-ag3s/bin/python -c "
import json,sys; r=json.load(open('$J')); v=r['vb3']; ok = v.get('pass', v.get('passed')) if isinstance(v, dict) else bool(v)
print(r['probe']['path'] if ok else 'FAIL', file=sys.stdout); print(json.dumps(v), file=sys.stderr)")
[ "$PROBE" != FAIL ] && [ -f "$PROBE" ] || { echo "re-made probe failed VB3 or missing: $PROBE"; exit 4; }
echo "probe $PROBE"
.venv-ag3s/bin/python benchmark/runners/launch_libero_paired.py --task-suite-name safelibero_spatial --safety-level "$LV" \
  --task-ids 0,1,2,3 --episode-ids 0,1,2,3,4 --data-split dev --run-id "nb1104-ag3s-final-L$LV" --modes vla_v11ab \
  --ag3s-version AG3S-final-panda-v3-e0to --hsr6-probe "$PROBE" --conversion C "$@"
