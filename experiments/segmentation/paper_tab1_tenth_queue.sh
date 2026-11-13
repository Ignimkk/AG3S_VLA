#!/usr/bin/env bash
# Paper Tab. 1, this server's 1/10 (sdm-f6, 2026-11-13; user: "전체 1600 에피소드 중 1/10", "table 1~3 순서대로",
# "성공률이 제일 높게 나올 것 같은 init 을 선택", "다른 환경에서 나머지 9/10"). The episodes are in
# doc/active/experiments/PAPER_TENTH_SPLIT_20261113.json (select_inits.py): SafeLIBERO 8 cells x task 0-3 x 5 inits = 160 ep,
# LIBERO 4 suites x task 0-9 x 5 inits = 200 ep. The other 9/10 = the "complement_inits" of the same file.
# Arms (same launcher, seed 7, main_aegis step budget, same run layout for every arm -- an episode's outcome depends on its
# position in the run): ag3s = $V, gto = $V's GT-oracle twin ($VGT), pi05 = --modes vla_only.
# usage: V=<AG3S version> VGT=<GT oracle version> [ARMS="ag3s pi05 gto"] [PARTS="sl lb"] [DRY=1] bash experiments/segmentation/paper_tab1_tenth_queue.sh
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv-ag3s/bin/python
: "${V:?}" "${VGT:?}"
ARMS=${ARMS:-"ag3s pi05 gto"}
PARTS=${PARTS:-"sl lb"}
DRY=${DRY:-0}
SPLIT=doc/active/experiments/PAPER_TENTH_SPLIT_20261113.json
OUT=outputs/2026-11-13_paper_tab1_1-10
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() {
  local lab=$1; shift
  if [ "$DRY" = 1 ]; then "$@" --dry-run > /dev/null 2>> "$LOG.dry" && log "dry ok $lab" || log "dry FAILED $lab"; return; fi
  log "start $lab"; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"
}
label() { $PY -c "import sys; from benchmark.runners import suite_registry as r; print(r.split_label(sys.argv[1], None if sys.argv[2] == '-' else sys.argv[2], 'test'))" "$1" "$2"; }
inits() { $PY -c "import json,sys; d=json.load(open('$SPLIT')); print(','.join(map(str, d[sys.argv[1]][sys.argv[2]]['inits'])))" "$1" "$2"; }
LP="benchmark/runners/launch_libero_paired.py --conversion C"
mode_args() {   # $1 = arm
  case $1 in
    ag3s) echo "--modes vla_v11ab --record-constraints full --ag3s-version $V" ;;
    gto)  echo "--modes vla_v11ab --record-constraints full --ag3s-version $VGT" ;;
    pi05) echo "--modes vla_only" ;;
  esac
}
log "paper Tab. 1 1/10: HEAD $(git rev-parse --short HEAD) arms='$ARMS' parts='$PARTS' V=$V VGT=$VGT"
for arm in $ARMS; do
  if [[ " $PARTS " == *" sl "* ]]; then
    for cell in spatial:I object:I goal:I long:I spatial:II object:II goal:II long:II; do
      suite=${cell%%:*}; lvl=${cell#*:}; ids=$(inits safelibero "$suite L-$lvl")
      run "$arm safelibero_$suite L$lvl task 0-3 init $ids" $PY $LP $(mode_args $arm) --task-suite-name safelibero_$suite --safety-level $lvl \
        --task-ids 0,1,2,3 --episode-ids $ids --seed 7 --data-split "$(label safelibero_$suite $lvl)" --safelibero-step-budget main_aegis \
        --run-id "tab1sdm-$arm-sl-$suite-L$lvl"
    done
  fi
  if [[ " $PARTS " == *" lb "* ]] && [ "$arm" != gto ]; then
    for suite in spatial object goal 10; do
      ids=$(inits libero "$suite")
      run "$arm libero_$suite task 0-9 init $ids" $PY $LP $(mode_args $arm) --task-suite-name libero_$suite \
        --task-ids 0,1,2,3,4,5,6,7,8,9 --episode-ids $ids --seed 7 --data-split "$(label libero_$suite -)" --run-id "tab1sdm-$arm-lb-$suite"
    done
  fi
done
log "paper Tab. 1 1/10 queue end"
