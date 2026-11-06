#!/usr/bin/env bash
# nb1104 (C6): re-made HSR6-Panda probe on this machine with the SL6-B code unchanged (CPU part).
#   phase_export --finger-contacts of the re-made SL4b-protocol rollout -> sl6_hsr6_panda (labels + probe + VB3).
set -u
cd "$(dirname "$0")/../.."
source scripts/setup/env.sh >/dev/null 2>&1
B="$AG3S_STORAGE_ROOT/results/benchmark"
DEV=$(ls -d "$B"/*_nb1104-probe-dev-pi05only-t0-9e1-4 | tail -1)
until [ "$(grep -c . "$DEV/episodes.jsonl" 2>/dev/null)" = 40 ] && [ -f "$DEV/.nb1104_done" -o -n "$(grep -l client_returncode "$DEV"/logs/*.log 2>/dev/null)" -o 1 = 1 ]; do sleep 20; done
OUT="$B/$(TZ=KST-9 date +%Y%m%dT%H%M%S)_fe3abf215dc4_nb1104-hsr6-panda-labels-remade"
mkdir -p "$OUT"
echo "phase export -> $OUT/phase $(date +%T)"
nice -n 5 .venv-libero/bin/python benchmark/adapters/libero_panda/phase_export.py --run "$DEV" --out "$OUT/phase" --finger-contacts
echo "phase export rc=$? $(date +%T)"
until ls -d "$B"/*_nb1104-probe-attention-audit-dev-e1-4 >/dev/null 2>&1 && [ "$(ls "$(ls -d "$B"/*_nb1104-probe-attention-audit-dev-e1-4 | tail -1)"/attention/*_attn.npz 2>/dev/null | wc -l)" = 40 ] && ! pgrep -f '^[^ ]*python[^ ]* benchmark/runners/attention_audit.py' >/dev/null; do sleep 20; done
ATT=$(ls -d "$B"/*_nb1104-probe-attention-audit-dev-e1-4 | tail -1)
echo "audit $ATT complete $(date +%T)"
.venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.sl6_hsr6_panda --out-run "$OUT" --dev-run "$DEV" --attn-run "$ATT" --phase-dir "$OUT/phase"
echo "probe rc=$? $(date +%T)"
