#!/usr/bin/env bash
# Render the deck to render_v3/slide-NN.jpg via LibreOffice (skill wrapper) + pdftoppm.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
SKILL="${PPTX_SKILL:-/root/.claude/skills/synced/d816952f-88f7-4d41-a0ff-71482dc719e2_4cde0165-1e5b-4a72-9f58-bb3e44a9da4c/pptx}"
mkdir -p "$HERE/render_v3"
cd "$HERE/render_v3"
"$PY" "$SKILL/scripts/office/soffice.py" --headless --convert-to pdf --outdir "$HERE/render_v3" "$HERE/HSR_phase_attention_20261013_v3.pptx"
rm -f slide-*.jpg
pdftoppm -jpeg -r 110 HSR_phase_attention_20261013_v3.pdf slide
ls -1 "$HERE"/render_v3/slide-*.jpg
