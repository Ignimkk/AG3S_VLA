#!/usr/bin/env bash
# SL-ATT1 small closed-loop check (sdm-f6, 2026-11-13; user: "어텐션 자체가 틀린 건 … L12H3 으로 소량만 테스트"): the KNOWS cell
# L12H3 (``-a12h3``, head + wrist fused as served) and L12H3 agent-view only (``-a12h3h``) on the attention-error scenarios, in the
# SAME run layout as the frozen version's existing runs (so those are the paired reference; run-layout effect, plan §9.36):
#   sub-pkgFrssdm-e1a-spatial-LI-t1       spatial L-I task 1 init 5-14   (bowl on the ramekin; attention on the ramekin)
#   ms-pkgFrssdm-object-LI-i15_16         object L-I task 0-3 init 15,16 (t2 milk: attention on the wine-bottle obstacle)
#   ms-pkgFrssdm-goal-LI-i17_18           goal L-I task 0-3 init 17,18   (t3 bowl -> cabinet: attention on the plate)
# 26 episodes per version. NOT an adoption run. The launchers take the GPU lock themselves. Do not edit while it runs.
# usage: bash experiments/segmentation/sota_att1_queue.sh   [VS="a12h3h a12h3"]
set -u
cd /mnt/dev/AG3S_VLA
F="AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-rs"
VS=${VS:-"a12h3h a12h3"}
for s in $VS; do
  V="$F-$s-tro"; TAG="${s}sdm"
  V="$V" TAG="$TAG" SUB="1:5,6,7,8,9,10,11,12,13,14" bash experiments/segmentation/sota_sub_queue.sh
  V="$V" TAG="$TAG" INITS=15,16 CELLS="object:I" bash experiments/segmentation/sota_ms_queue.sh
  V="$V" TAG="$TAG" INITS=17,18 CELLS="goal:I" bash experiments/segmentation/sota_ms_queue.sh
done
echo "$(date +%Y-%m-%dT%H:%M:%S) queue sota att1 end ($VS)"
