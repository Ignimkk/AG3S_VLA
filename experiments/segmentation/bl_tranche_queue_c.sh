#!/usr/bin/env bash
# Baseline tranche 1, queue C (dev-4a, 2026-11-10 04:0x; user delegation 02:0x "judge, fix, rerun yourself"):
#   1. KNOWS +- GT spatial LI with the EE-frame fix (-r3) FIRST -> early closed-loop check of the fix (doc §4.11)
#   2. the settled arms' remaining blocks (pi0.5, AEGIS privileged names, multi-link CBF GT; LIBERO pi0.5)
#   3. AEGIS / multi-link CBF detector arms with VLM naming (vlm_name, -r2) on every block + LIBERO (doc §4.10)
#   4. KNOWS +- GT on every other block (-r3) + LIBERO KNOWS
# One HEAD for the whole queue (committed before launch). Each step = bl_tranche_arms_queue.sh (skips finished blocks).
set -u
cd /mnt/dev/AG3S_VLA
export AG3S_BASELINE_VLM_WEIGHT=${AG3S_BASELINE_VLM_WEIGHT:-/root/.cache/ag3s_local/Qwen2.5-VL-7B-Instruct}
Q=experiments/segmentation/bl_tranche_arms_queue.sh
SUITES=spatial LEVELS=I LIBERO_SUITES= SUFFIX=-r3 SAFE_ARMS_SEL="knows|--gt knows|" bash $Q
SAFE_ARMS_SEL="none| aegis|--gt multilink_cbf|--gt" LIBERO_ARMS_SEL="none|" bash $Q
SUFFIX=-r2 SAFE_ARMS_SEL="aegis| multilink_cbf|" LIBERO_ARMS_SEL="aegis| multilink_cbf|" bash $Q
SUFFIX=-r3 SAFE_ARMS_SEL="knows|--gt knows|" LIBERO_ARMS_SEL="knows|" bash $Q
echo "$(date +%Y-%m-%dT%H:%M:%S) queue baseline t1 C end" >> outputs/2026-11-09_baseline_tranche/t1/queue.log
