#!/usr/bin/env bash
# SL-GTTO16 (A, user ruling 2026-11-08 00:2x): F13 = F12 + tau not applied to obstacle-object rows.
# 1. calibration: F13 observe-only (`-loose2-obst-obs`) on LIBERO 9 x seeds 10 · 11 and SafeLIBERO LI 11 seed 10
#    (--record-constraints full). FP = would-change / hold on a VLA-success LIBERO episode; TP = would-change before collide_t.
# 2. evaluation (fixed before results; user goal: success within 10 %p of VLA, i.e. LIBERO >= 25/27): F13 closed loop LIBERO 27,
#    then SafeLIBERO 11 + LIBERO-Safety 10 if >= 25/27 -- run when the calibration has <= 2 FP episodes of 17.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-08_A_장애물행_엄격_F13
mkdir -p $OUT
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
VO=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst-obs
LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $VO --record-constraints full"
for s in 10 11; do
  run "F13obs LIBERO9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 0,1,2,3,4,6,7,8,9 --episode-ids 0 --seed $s --run-id gtto-gt16obs-libero9s$s
done
SL="--task-suite-name safelibero_spatial --safety-level I --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --seed 10"
run "F13obs SafeLIBERO t1" $PY $LP $SL --task-ids 1 --episode-ids 1,4 --run-id gtto-gt16obs-safelibero11s10-t1-LI
run "F13obs SafeLIBERO t2" $PY $LP $SL --task-ids 2 --episode-ids 0,1,2,3,4 --run-id gtto-gt16obs-safelibero11s10-t2-LI
run "F13obs SafeLIBERO t3" $PY $LP $SL --task-ids 3 --episode-ids 0,2,3,4 --run-id gtto-gt16obs-safelibero11s10-t3-LI
$PY -I experiments/segmentation/gtto_obs_fp.py --tag gt16obs --out $OUT/관찰_F13_seed10_11 >> "$LOG" 2>&1
nfp=$($PY -c "import json;d=json.load(open('$OUT/관찰_F13_seed10_11/obs_fp.json'));print(sum(1 for e in d['libero'] if e['success'] and (e['would_change'] or e['would_hold'])))")
log "F13 observe: FP episodes = $nfp / VLA-success"
if [ "$nfp" -le 2 ]; then
  log "calibration ok: F13 closed loop"
  V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst TAG=gt16obst OUT=$OUT MIN27=25 bash experiments/segmentation/gtto16_queue.sh
else
  log "calibration: $nfp FP episodes -- closed loop not run (analyse first)"
  log "queue end"
fi
