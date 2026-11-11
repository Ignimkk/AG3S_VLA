#!/usr/bin/env bash
# SDM pi0.5-alone baseline queue (dev-c7, 2026-11-12 KST; user instruction 2026-11-12: run the baselines that do not wait
# for the AG3S freeze S0). Tab. 1 baseline rows other than pi0.5 use the published values (user decision 2026-11-11), so the
# only baseline we measure for the paper tables is pi0.5 alone:
#   P1 SafeLIBERO anchor: 4 suites x task 0-3 x Level I/II x init 19-28 (reporting split) = 320 ep. Paired with the weekend
#      AG3S runs: SAME launcher / client / per-episode policy_seed as the AG3S arm (launch_libero_paired.py --conversion C,
#      --modes vla_only vs vla_v11ab; G108/G109 qe2 pattern), main_aegis step budget, seed 7, official 1 mm collision rule.
#   P2 LIBERO (no obstacles): 4 suites x task 0-9 x init 19-23 = 200 ep, same launcher.
#   P3 LIBERO-Safety pi0.5 tranche C (official pi05_libero_safety), init 19, 60 ep: bl_tranche_libero_safety_queue.sh, -sdm.
#   P4 (optional) queue F remainder: LIBERO multi-link CBF libero_10 (port verification init 15-18, supplement only), -sdm.
# Every SDM run-id starts with "sdm-" or ends in "-sdm" (SDM_FRESH_SETUP_20261111_KO.md §8.1: AMRC results are merged into
# the same results dir, and done-checks look at run-ids only).
# A block is skipped when a run dir with the same run-id has client_returncode 0 and the expected episode count -> rerunning
# this script resumes. Launchers take /tmp/ag3s-vla-gpu-validation.lock themselves: NO outer flock (G53 self-deadlock).
# Do not change this checkout's HEAD while the queue runs (provenance check at the end of each run).
# usage: [PARTS="1 2 3 4"] [SL_INITS=19,...,28] [LB_INITS=19,...,23] [DRY=1] bash experiments/segmentation/sdm_pi05_anchor_queue.sh
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
PARTS=${PARTS:-"1 2 3 4"}
SL_INITS=${SL_INITS:-19,20,21,22,23,24,25,26,27,28}
LB_INITS=${LB_INITS:-19,20,21,22,23}
DRY=${DRY:-0}
OUT=outputs/2026-11-12_SDM_pi05_baseline
mkdir -p "$OUT"
LOG=$OUT/queue.log
STORE=/mnt/dev/AG3S_VLA-storage/results/benchmark
LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_only"
log() { echo "$(date +%Y-%m-%dT%H:%M:%S%z) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
label() { $PY -c "import sys; from benchmark.runners import suite_registry as r; print(r.split_label(sys.argv[1], None if sys.argv[2] == '-' else sys.argv[2], 'test'))" "$1" "$2"; }
n_of() { awk -F, '{print NF}' <<< "$1"; }
done_run() {  # $1 = run-id, $2 = expected episodes -> 0 if a finished run (rc 0, all episodes) exists
  local d
  for d in "$STORE"/*_"$1"-vla_only; do
    [ -e "$d/episodes.jsonl" ] || continue
    [[ "$(basename "$d")" == INVALID* ]] && continue
    [ "$(grep -c . "$d/episodes.jsonl")" -ge "$2" ] || continue
    $PY -c "import json,sys; m=json.load(open(sys.argv[1])); r=[e['settings'] for e in m['executions'] if e['settings'].get('role')=='launcher_result']; sys.exit(0 if r and r[-1].get('client_returncode')==0 else 1)" "$d/run_manifest.json" && return 0
  done
  return 1
}
block() {  # $1 = label, $2 = run-id, $3 = expected episodes, rest = launcher argv (without --run-id)
  local lab=$1 rid=$2 exp=$3; shift 3
  if [ "$DRY" = 1 ]; then
    "$@" --run-id "$rid" --dry-run > /dev/null 2>> "$LOG.dry" && log "dry ok $rid" || log "dry FAILED $rid (see $LOG.dry)"
    return
  fi
  if done_run "$rid" "$exp"; then log "skip $rid (done)"; return; fi
  run "$lab" "$@" --run-id "$rid"
}
log "SDM pi0.5 baseline queue: HEAD $(git rev-parse --short HEAD) parts='$PARTS' sl_inits=$SL_INITS lb_inits=$LB_INITS dry=$DRY"
if [[ " $PARTS " == *" 1 "* ]]; then
  for suite in spatial object goal long; do
    for lvl in I II; do
      split=$(label safelibero_$suite $lvl)
      block "P1 pi05 safelibero_$suite L$lvl init $SL_INITS" "sdm-pi05-sl-$suite-L$lvl" $((4 * $(n_of $SL_INITS))) \
        $PY $LP --task-suite-name safelibero_$suite --safety-level $lvl --task-ids 0,1,2,3 --episode-ids $SL_INITS \
        --seed 7 --data-split "$split" --safelibero-step-budget main_aegis
    done
  done
  log "part 1 end"
fi
if [[ " $PARTS " == *" 2 "* ]]; then
  for suite in libero_spatial libero_object libero_goal libero_10; do
    split=$(label $suite -)
    block "P2 pi05 $suite init $LB_INITS" "sdm-pi05-lb-${suite#libero_}" $((10 * $(n_of $LB_INITS))) \
      $PY $LP --task-suite-name $suite --task-ids 0,1,2,3,4,5,6,7,8,9 --episode-ids $LB_INITS --seed 7 --data-split "$split"
  done
  log "part 2 end"
fi
if [[ " $PARTS " == *" 3 "* ]]; then
  log "part 3: LIBERO-Safety tranche C (own log outputs/2026-11-09_baseline_tranche/t1/queue.log)"
  INITS=19 SUFFIX=-sdm DRY=$DRY bash experiments/segmentation/bl_tranche_libero_safety_queue.sh
  log "part 3 end"
fi
if [[ " $PARTS " == *" 4 "* ]]; then
  if [ "$DRY" = 1 ]; then
    log "part 4 dry: skipped (bl_tranche_arms_queue.sh has no dry mode)"
  else
    log "part 4: LIBERO multi-link libero_10 (own log outputs/2026-11-09_baseline_tranche/t1/queue.log)"
    export AG3S_BASELINE_VLM_WEIGHT=${AG3S_BASELINE_VLM_WEIGHT:-/root/.cache/ag3s_local/Qwen2.5-VL-7B-Instruct}
    SUFFIX=-r2-sdm SUITES="" SAFE_ARMS_SEL="" LIBERO_ARMS_SEL="multilink_cbf|" LIBERO_SUITES=libero_10 \
      bash experiments/segmentation/bl_tranche_arms_queue.sh
    log "part 4 end"
  fi
fi
log "SDM pi0.5 baseline queue end"
