#!/usr/bin/env bash
# P1 (제약 단계 추종, 2026-11-10): per-frame finger/target contact export of the pe123458 dev-split runs (CPU, read only).
#   LIBERO      -> benchmark/adapters/libero_panda/phase_export.py --finger-contacts --no-segmentation
#   SafeLIBERO  -> AG3S/ag3s/experiments/studies/t91b_safelibero_export.py (seed 7, Level I)
# usage: bash experiments/segmentation/stage_track/export_contacts.sh {libero|safelibero} OUT_ROOT
set -u
cd /mnt/dev/AG3S_VLA
B=/mnt/dev/AG3S_VLA-storage/results/benchmark
which=$1; OUT=$2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
latest() { ls -d $B/*_"$1"-vla_v11ab-AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123458-C | tail -1; }
if [ "$which" = libero ]; then
  for tag in gtto-pe123458-libero9s7 gtto-pe123458-libero9s8 gtto-pe123458-libero9s9 gtto-pe123458-libero9s10 \
             gtto-pe123458-libero9s11 gtto-pe123458-libero4s7 gtto-pe123458-libero4s8 gtto-pe123458-libero4s9; do
    R=$(latest $tag); echo "== $(date +%T) $tag $R"
    nice -n 10 .venv-libero/bin/python benchmark/adapters/libero_panda/phase_export.py --run "$R" --out "$OUT/$tag" \
      --no-segmentation --finger-contacts 2>&1 | grep -E '^\{|Error|error' | grep -v Warning
  done
else
  for spec in "gtto-pe123458-safelibero11-t1-LI 1" "gtto-pe123458-safelibero11-t2-LI 2" "gtto-pe123458-safelibero11-t3-LI 3" \
              "qe-pe123458-e1a-spatial-LI-t0-LI 0" "qe-pe123458-e1a-spatial-LI-t1-LI 1" \
              "qe-pe123458-e1a-spatial-LI-t2-LI 2" "qe-pe123458-e1a-spatial-LI-t3-LI 3"; do
    set -- $spec; tag=$1; tid=$2; R=$(latest $tag); echo "== $(date +%T) $tag t$tid $R"
    nice -n 10 .venv-libero/bin/python AG3S/ag3s/experiments/studies/t91b_safelibero_export.py "$OUT/$tag" $tid \
      --run "$R" --bench /mnt/dev/AG3S_VLA 2>&1 | grep -E '^\{|Error|error' | grep -v Warning
  done
fi
echo "== $(date +%T) done $which"
