#!/usr/bin/env bash
# Baseline tranche 1, queue D (dev-4a, 2026-11-10 05:1x; user delegation 02:0x). Replaces queue C after the multi-link
# CBF GT FLAG (64 ep CAR 46.9 vs paper 73.0) -> --gt default "exact" (paper Table III privileged condition, doc §4.12).
#   0. early checks of every changed arm on spatial LI: multi-link CBF GT exact (-r2), AEGIS / multi-link detector with
#      VLM naming (-r2)   [KNOWS frame fix already checked: -r3 spatial LI done]
#   1. settled arms' remaining blocks (pi0.5, AEGIS privileged names; LIBERO pi0.5)
#   2. multi-link CBF GT exact on every block (-r2)
#   3. AEGIS / multi-link detector VLM naming on every block (-r2) + LIBERO
#   4. KNOWS +- GT (EE-frame fix) on every block (-r3) + LIBERO KNOWS
set -u
cd /mnt/dev/AG3S_VLA
export AG3S_BASELINE_VLM_WEIGHT=${AG3S_BASELINE_VLM_WEIGHT:-/root/.cache/ag3s_local/Qwen2.5-VL-7B-Instruct}
Q=experiments/segmentation/bl_tranche_arms_queue.sh
SUITES=spatial LEVELS=I LIBERO_SUITES= SUFFIX=-r2 SAFE_ARMS_SEL="multilink_cbf|--gt aegis| multilink_cbf|" bash $Q
SAFE_ARMS_SEL="none| aegis|--gt" LIBERO_ARMS_SEL="none|" bash $Q
LIBERO_SUITES= SUFFIX=-r2 SAFE_ARMS_SEL="multilink_cbf|--gt" bash $Q
SUFFIX=-r2 SAFE_ARMS_SEL="aegis| multilink_cbf|" LIBERO_ARMS_SEL="aegis| multilink_cbf|" bash $Q
SUFFIX=-r3 SAFE_ARMS_SEL="knows|--gt knows|" LIBERO_ARMS_SEL="knows|" bash $Q
echo "$(date +%Y-%m-%dT%H:%M:%S) queue baseline t1 D end" >> outputs/2026-11-09_baseline_tranche/t1/queue.log
