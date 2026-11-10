#!/usr/bin/env bash
# Baseline tranche 1, queue F (dev-4a, 2026-11-10 22:4x): after queue E.
#   1. multi-link CBF detector (VLM naming, -r2) remaining SafeLIBERO blocks + LIBERO multi-link (doc §4.14: no port fix)
#   2. LIBERO-Safety tranche C, pi0.5 alone, official pi05_libero_safety, init 19 (doc §4.16: reporting split 19-49)
set -u
cd /mnt/dev/AG3S_VLA
export AG3S_BASELINE_VLM_WEIGHT=${AG3S_BASELINE_VLM_WEIGHT:-/root/.cache/ag3s_local/Qwen2.5-VL-7B-Instruct}
SUFFIX=-r2 SAFE_ARMS_SEL="multilink_cbf|" LIBERO_ARMS_SEL="multilink_cbf|" bash experiments/segmentation/bl_tranche_arms_queue.sh
INITS=19 bash experiments/segmentation/bl_tranche_libero_safety_queue.sh
echo "$(date +%Y-%m-%dT%H:%M:%S) queue baseline t1 F end" >> outputs/2026-11-09_baseline_tranche/t1/queue.log
