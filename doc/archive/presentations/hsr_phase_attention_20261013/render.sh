#!/usr/bin/env bash
# Render the deck to render/slide-NN.jpg via LibreOffice (skill wrapper) + pdftoppm.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
SKILL="${PPTX_SKILL:-/root/.claude/skills/synced/d816952f-88f7-4d41-a0ff-71482dc719e2_4cde0165-1e5b-4a72-9f58-bb3e44a9da4c/pptx}"
mkdir -p "$HERE/render"
cd "$HERE/render"
"$PY" "$SKILL/scripts/office/soffice.py" --headless --convert-to pdf --outdir "$HERE/render" "$HERE/HSR_phase_attention_20261013.pptx"
rm -f slide-*.jpg
pdftoppm -jpeg -r 110 HSR_phase_attention_20261013.pdf slide
ls -1 "$HERE"/render/slide-*.jpg
