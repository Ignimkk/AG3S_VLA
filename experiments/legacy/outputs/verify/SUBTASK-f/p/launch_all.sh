#!/bin/bash
# SUBTASK-f: wait for the regression baseline (baseline/.done), then V0 (gate off, 2 runs, port 8230), then V1 (gate on, 48 runs,
# 4 chains on ports 8231-8234) only if V0 executed + planning actions equal T39 for both runs. Serial: nothing overlaps the baseline.
# usage: setsid nohup launch_all.sh > ${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-f/launch_all.out 2>&1 &
source ${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-f/p/env.sh
echo "launch_all start $(date -Iseconds)"
until [ -f $SD_OUT/baseline/.done ]; do sleep 20; done
echo "baseline done $(date -Iseconds)"; cat $SD_OUT/baseline/run.out
mkdir -p $SD_OUT/v0
$P/launch_v0.sh > $SD_OUT/v0/launch.out 2>&1; echo "v0 rc=$? $(date -Iseconds)"
ok=$(JAX_PLATFORMS=cpu $PY - <<'PY'
import json, pathlib
V = pathlib.Path("${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-f/v0")
ok = True
for n in ("E3b_ep1807_s18071", "E3b_ep1800_s18001"):
    f = V / f"compare_{n}.json"
    if not f.exists(): ok = False; continue
    r = json.loads(f.read_text()); ok = ok and r["executed_actions_equal"] and r["planning_actions_equal"]
print(1 if ok else 0)
PY
)
echo "v0 actions equal T39: $ok"
if [ "$ok" != 1 ]; then echo "STOP: V0 differs, V1 not started $(date -Iseconds)"; exit 1; fi
mkdir -p $SD_OUT/runs
$P/launch_v1.sh 4 > $SD_OUT/runs/launch.out 2>&1; echo "v1 rc=$? $(date -Iseconds)"
touch $SD_OUT/.v1_finished
echo "launch_all end $(date -Iseconds)"
