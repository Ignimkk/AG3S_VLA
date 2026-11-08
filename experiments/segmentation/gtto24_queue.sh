#!/usr/bin/env bash
# lead G88 (2026-11-08 03:5x): after the A-alone ablation (gtto23 step 2, ``…-obst3`` = A'' + no carve + support under):
# 1. B''' = B'' + no carve (dev-49 review §14: HVC frees real obstacle voxels) + support "under" (B'' SafeLIBERO t2e0: the moka pot
#    next to the bowl was taken as its support -> no obstacle row before the collision);
# 2. the non-GT version (gtto20's run, scene obstacle objects with the same base-only support rule).
set -u
cd /mnt/dev/AG3S_VLA
until grep -q "queue end" outputs/2026-11-08_A3_GT장애물_F13c/queue.log 2>/dev/null; do sleep 20; done
V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve TAG=gt21pts OUT=outputs/2026-11-08_B3_GT장애물_점모델_무carve_F14e MIN27=25 \
  EXTRA="--record-constraints full" EXTRA_LS="--record-constraints full" bash experiments/segmentation/gtto16_queue.sh
V=AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts TAG=gt20att OUT=outputs/2026-11-08_GT없는_AG3S_F14d MIN27=25 \
  EXTRA="--record-constraints full" EXTRA_LS="--record-constraints full" bash experiments/segmentation/gtto16_queue.sh
