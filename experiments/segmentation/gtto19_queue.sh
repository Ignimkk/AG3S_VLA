#!/usr/bin/env bash
# SL-GTTO17 (B, user ruling 2026-11-08): F14 = F13 + held object as a 20 mm point shell. No tuned parameter (20 mm fixed a priori)
# -> straight to the evaluation protocol (LIBERO 27, gate >= 25/27 = within 10 %p of VLA; then SafeLIBERO 11 + LIBERO-Safety 10),
# --record-constraints full for the failure analysis and the AG3S feature scenes. Waits for the F13 queue (gtto18) to end.
set -u
cd /mnt/dev/AG3S_VLA
until grep -q "queue end" outputs/2026-11-08_A2_장애물행_엄격_F13b/queue.log 2>/dev/null; do sleep 20; done
V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst2-pts TAG=gt17pts2 OUT=outputs/2026-11-08_B_쥔물체_점모델 MIN27=25 \
  EXTRA="--record-constraints full" EXTRA_LS="--record-constraints full" bash experiments/segmentation/gtto16_queue.sh
