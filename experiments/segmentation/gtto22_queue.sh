#!/usr/bin/env bash
# A' closed loop on the evaluation protocol (lead G81, 2026-11-08 01:4x): the gtto21 gate counted the calibration under the old
# beneath reach (10 mm; 3 FP episodes, 2 of them t4 cabinet-top rows 43-58 mm outside the bowl, inside the new 80 mm reach) ->
# A' is evaluated after B' (gtto19) ends.
set -u
cd /mnt/dev/AG3S_VLA
until grep -q "queue end" outputs/2026-11-08_B_쥔물체_점모델/queue.log 2>/dev/null; do sleep 20; done
V=AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst2 TAG=gt16obst2 OUT=outputs/2026-11-08_A2_장애물행_엄격_F13b MIN27=25 \
  EXTRA="--record-constraints full" EXTRA_LS="--record-constraints full" bash experiments/segmentation/gtto16_queue.sh
