#!/usr/bin/env bash
# lead G89 / G91 (2026-11-08 04:3x): B5 = B3 + obstacle rows over the whole chunk + general tau 100 mm (B4 replaced), after the
# non-GT run (gtto24 step 2). Same protocol / gate / records.
set -u
cd /mnt/dev/AG3S_VLA
until grep -q "queue end" outputs/2026-11-08_GT없는_AG3S_F14d/queue.log 2>/dev/null; do sleep 20; done
V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh-t100 TAG=gt23t100 OUT=outputs/2026-11-08_B5_전체창_t100_F14g MIN27=25 \
  EXTRA="--record-constraints full" EXTRA_LS="--record-constraints full" bash experiments/segmentation/gtto16_queue.sh
