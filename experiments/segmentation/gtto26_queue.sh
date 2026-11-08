#!/usr/bin/env bash
# lead G92 (2026-11-08 04:4x): after the A-alone ablation (gtto23 step 2), in order:
#   1. B3 = B2 + no carve + support "under"                        (TAG gt21pts)
#   2. B5 = B3 + obstacle rows over the whole chunk + general tau 100 (TAG gt23t100)
#   3. B5 without GT (scene obstacle objects, attention grounding)  (TAG gt24att)
# Same protocol (LIBERO 27 gate >= 25 -> SafeLIBERO 11 + LIBERO-Safety 10), --record-constraints full.
set -u
cd /mnt/dev/AG3S_VLA
until grep -q "queue end" outputs/2026-11-08_A3_GT장애물_F13c/queue.log 2>/dev/null; do sleep 20; done
R="EXTRA=--record-constraints\ full"
V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve TAG=gt21pts OUT=outputs/2026-11-08_B3_GT장애물_점모델_무carve_F14e MIN27=25 \
  EXTRA="--record-constraints full" EXTRA_LS="--record-constraints full" bash experiments/segmentation/gtto16_queue.sh
V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh-t100 TAG=gt23t100 OUT=outputs/2026-11-08_B5_전체창_t100_F14g MIN27=25 \
  EXTRA="--record-constraints full" EXTRA_LS="--record-constraints full" bash experiments/segmentation/gtto16_queue.sh
V=AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100 TAG=gt24att OUT=outputs/2026-11-08_GT없는_AG3S_B5_F14h MIN27=25 \
  EXTRA="--record-constraints full" EXTRA_LS="--record-constraints full" bash experiments/segmentation/gtto16_queue.sh
