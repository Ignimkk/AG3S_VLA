#!/usr/bin/env bash
# dev-38/dev-e5 SL-QE2 calibration (2026-11-09 02:xx KST, G111): does `--ee-action-qp-translation-only` (the action QP bends
# the position only; E1a task 3 lost the grasp pose to rx / ry / rz changes of ~1.0, G110) keep LIBERO parity and improve
# "avoid and still finish" on SafeLIBERO, on CALIBRATION data only (no E1a test init 5-14, no evaluation seeds 7 / 8 / 9):
#   LIBERO 9 x seed 10 · 11            : B5 (gt23t100) vs B5-tro            (VLA reference = gtto-gt12obs-libero9s10/s11)
#   SafeLIBERO Spatial LI, task 0-3 x dev init 0-4, seed 10, main_aegis : vla_only vs B5 vs B5-tro (paired policy_seed)
# One job per (version, chunk of episodes); the launcher takes the GPU lock per job (no outer flock).
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-09_QE2_회전고정_보정
mkdir -p "$OUT"
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
B5=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh-t100
LP="benchmark/runners/launch_libero_paired.py --conversion C"
SL="--task-suite-name safelibero_spatial --safety-level I --task-ids 0,1,2,3 --episode-ids 0,1,2,3,4 --seed 10 --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --safelibero-step-budget main_aegis"
log "qe3b: restart after the 01:3x session stop (vla_only run interrupted at 4/20 -> rerun)"
run "vla_only SafeLIBERO dev s10" $PY $LP --modes vla_only $SL --run-id qe3-vla-safelibero20s10
for pair in "gt23t100 $B5" "gt23tro $B5-tro"; do
  TAG=${pair%% *}; V=${pair#* }
  for s in 10 11; do
    run "$TAG LIBERO9 seed $s" $PY $LP --modes vla_v11ab --ag3s-version $V --task-suite-name libero_spatial --task-ids 0,1,2,3,4,6,7,8,9 --episode-ids 0 --seed $s --run-id qe3-$TAG-libero9s$s
  done
  run "$TAG SafeLIBERO dev s10" $PY $LP --modes vla_v11ab --ag3s-version $V --record-constraints full $SL --run-id qe3-$TAG-safelibero20s10
done
log "queue end (qe3b)"
