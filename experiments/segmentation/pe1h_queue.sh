#!/usr/bin/env bash
# SL-PE closed loop, step h (dev-4a, 2026-11-09 02:2x KST, G162): pe1g's seed 9 job was refused at start
# (HEAD moved 4ef92202 -> a811331c by the 02:04-02:06 commits while it waited for the GPU lock; run renamed INVALID-notstarted-).
# Waits for 'queue g end', then reruns pe1234567b LIBERO t3,4,7,9 seed 9.
set -u
cd /mnt/dev/AG3S_VLA
source scripts/setup/env.sh >/dev/null 2>&1
cd /mnt/dev/AG3S_VLA
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dev/AG3S_VLA/.runtime/magickwand/lib"
PY=.venv-ag3s/bin/python
OUT=outputs/2026-11-08_PE_경로끝점_identity/closed_loop
LOG=$OUT/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }
run() { log "start $1"; shift; "$@" >> "$LOG.out" 2>&1; log "done rc=$?"; }
until grep -q "queue g end" "$LOG"; do sleep 30; done
log "pe1h: rerun pe1234567b seed 9 (refused at start by commit change)"
V=AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe1234567; TAG=pe1234567b
LP="benchmark/runners/launch_libero_paired.py --conversion C --modes vla_v11ab --ag3s-version $V --record-constraints full"
for s in 9; do
  run "$TAG LIBERO t3,4,7,9 seed $s" $PY $LP --task-suite-name libero_spatial --task-ids 3,4,7,9 --episode-ids 0 --seed $s \
    --run-id gtto-$TAG-libero4s$s
done
log "queue h end"
