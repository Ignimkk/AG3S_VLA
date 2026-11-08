#!/usr/bin/env bash
# SL-GTTO19 (lead G85, 2026-11-08 02:4x): B'' (A'' GT obstacle objects + B point model) then A'' alone, evaluation protocol
# (LIBERO 27 gate >= 25/27 -> SafeLIBERO 11 + LIBERO-Safety 10), --record-constraints full. No observe calibration: A'' has
# no tuned parameter (tau_obs 10 mm = the obstacle margin, 20 mm surface tolerance a priori). Waits for the B' queue (gtto19).
set -u
cd /mnt/dev/AG3S_VLA
until grep -q "queue end" outputs/2026-11-08_B_쥔물체_점모델/queue.log 2>/dev/null; do sleep 20; done
V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts TAG=gt19pts OUT=outputs/2026-11-08_B2_GT장애물_점모델_F14c MIN27=25 \
  EXTRA="--record-constraints full" EXTRA_LS="--record-constraints full" bash experiments/segmentation/gtto16_queue.sh
V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3 TAG=gt19obst3 OUT=outputs/2026-11-08_A3_GT장애물_F13c MIN27=25 \
  EXTRA="--record-constraints full" EXTRA_LS="--record-constraints full" bash experiments/segmentation/gtto16_queue.sh
