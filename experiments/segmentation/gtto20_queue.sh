#!/usr/bin/env bash
# SL-GTTO20 (user ruling 2026-11-08 "결과 좋으면 GT 사용하지 않고 AG3S 사용해서 각 벤치마크"; B'' LIBERO 27/27): B'' without GT
# perception -- attention grounding + the obstacle-object surface from AG3S's own perception (--scene-obstacle-objects).
# Same protocol and gate (LIBERO >= 25/27 -> SafeLIBERO 11 + LIBERO-Safety 10), --record-constraints full. Waits for A'' (gtto23).
set -u
cd /mnt/dev/AG3S_VLA
until grep -q "queue end" outputs/2026-11-08_A3_GT장애물_F13c/queue.log 2>/dev/null; do sleep 20; done
V=AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts TAG=gt20att OUT=outputs/2026-11-08_GT없는_AG3S_F14d MIN27=25 \
  EXTRA="--record-constraints full" EXTRA_LS="--record-constraints full" bash experiments/segmentation/gtto16_queue.sh
