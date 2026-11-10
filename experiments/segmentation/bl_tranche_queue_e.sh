#!/usr/bin/env bash
# Baseline tranche 1, queue E (dev-4a, 2026-11-10 15:2x; user delegation). Same HEAD as queue D (d452c804), order only:
# the multi-link CBF detector arm FLAGged at 64 ep (CAR 40.6 vs paper 73.0, doc §4.14) -> its remaining blocks are
# deferred until the diagnosis is done; the GPU keeps running the arms whose version is settled.
#   1. AEGIS detector (VLM naming, -r2) remaining blocks + LIBERO AEGIS
#   2. KNOWS +- GT (EE-frame fix, -r3) every block + LIBERO KNOWS
set -u
cd /mnt/dev/AG3S_VLA
export AG3S_BASELINE_VLM_WEIGHT=${AG3S_BASELINE_VLM_WEIGHT:-/root/.cache/ag3s_local/Qwen2.5-VL-7B-Instruct}
Q=experiments/segmentation/bl_tranche_arms_queue.sh
SUFFIX=-r2 SAFE_ARMS_SEL="aegis|" LIBERO_ARMS_SEL="aegis|" bash $Q
SUFFIX=-r3 SAFE_ARMS_SEL="knows|--gt knows|" LIBERO_ARMS_SEL="knows|" bash $Q
echo "$(date +%Y-%m-%dT%H:%M:%S) queue baseline t1 E end" >> outputs/2026-11-09_baseline_tranche/t1/queue.log
