#!/usr/bin/env bash
# LIBERO parity evaluation (lead G67/G68, 2026-11-07): the loosened version chosen from the observe-only calibration
# (seeds 10 · 11) runs closed loop on the EVALUATION protocol LIBERO 9 x seeds 7 · 8 · 9; adoption needs 27/27 (= VLA only).
# Then (only if 27/27) SafeLIBERO LI 11 and LIBERO-Safety L0 10 at seed 7, reported next to F8 and VLA only.
# usage: V=<version> TAG=<run tag> OUT=<outputs dir> [MIN27=25] bash gtto16_queue.sh  (2026-11-08: MIN27 = the LIBERO gate, default 27)
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
: "${V:?}" "${TAG:?}" "${OUT:?}"
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $V ${EXTRA:-}"
for s in 7 8 9; do
  run "$TAG LIBERO9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 0,1,2,3,4,6,7,8,9 --episode-ids 0 --seed $s --run-id gtto-$TAG-libero9s$s
done
ok=$($PY - "$TAG" <<'EOF'
import json, pathlib, sys
R = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
n = k = 0
for s in ("s7", "s8", "s9"):
    runs = [p for p in sorted(R.glob(f"*_gtto-{sys.argv[1]}-libero9{s}-*")) if (p / "episodes.jsonl").is_file()]
    if runs:
        for l in (runs[-1] / "episodes.jsonl").read_text().splitlines():
            if l.strip():
                n += 1; k += bool(json.loads(l)["official_success"])
print(f"{k}/{n}")
EOF
)
log "LIBERO 27 = $ok"
k27=${ok%%/*}
if [ "$k27" -ge "${MIN27:-27}" ]; then
  SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --seed 7"
  run "$TAG SafeLIBERO t1" $PY $LP $SL --task-ids 1 --episode-ids 1,4 --run-id gtto-$TAG-safelibero11-t1
  run "$TAG SafeLIBERO t2" $PY $LP $SL --task-ids 2 --episode-ids 0,1,2,3,4 --run-id gtto-$TAG-safelibero11-t2
  run "$TAG SafeLIBERO t3" $PY $LP $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-$TAG-safelibero11-t3
  LS="benchmark/runners/launch_libero_safety_paired.py --level 0 --task-ids 0 --policy-checkpoint pi05_libero --conversion C --seed 7 --modes vla_v11ab --ag3s-version $V ${EXTRA_LS:-}"
  run "$TAG LS human_safety" $PY $LS --task-suite-name human_safety --episode-ids 0,1,2,3 --run-id gtto-$TAG-ls10
  run "$TAG LS obstacle_avoidance" $PY $LS --task-suite-name obstacle_avoidance --episode-ids 0,1,2,3 --run-id gtto-$TAG-ls10
  run "$TAG LS affordance" $PY $LS --task-suite-name affordance --episode-ids 0,2 --run-id gtto-$TAG-ls10
else
  log "LIBERO not 27/27: safety benches not run (next loosening first)"
fi
log "queue end"
