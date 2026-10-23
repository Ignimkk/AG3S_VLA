#!/usr/bin/env bash
# Create this checkout's .venv-baselines (Python 3.11, uv) for Baseline/paper_baselines (PAPER-BASELINES).
# Separate from .venv-ag3s (numpy 2.4.6 pipeline) and .venv-openpi-live so neither is touched.
#   - interpreter: this checkout's .python (uv-managed CPython 3.11)
#   - packages:    requirements-baselines.txt (full lock) if present, else requirements-baselines.in (top-level pins)
#   - sam2:        editable install of the pinned Grounded-SAM-2 clone (external/Grounded-SAM-2) with
#                  SAM2_BUILD_CUDA=0 (the optional connected-components CUDA op is skipped; sam2 falls back to
#                  its python path) and --no-build-isolation (re-uses the pinned torch)
#   - caches:      uv cache in .cache/uv; model weights in .cache/paper_baselines (see paper_baselines_env.sh)
# Usage: bash Baseline/paper_baselines/setup/setup_venv.sh [--freeze] [--check]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
source "$ROOT/scripts/setup/env.sh" >/dev/null
UV="${AG3S_UV:-$AG3S_STORAGE_ROOT/tools/uv}"
[ -x "$UV" ] || { echo "uv not found: $UV (set AG3S_UV)" >&2; exit 1; }
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.python"
ENV_DIR="$ROOT/.venv-baselines"
PY="$ENV_DIR/bin/python"
IN="$HERE/requirements-baselines.in"
LOCK="$HERE/requirements-baselines.txt"
GSAM2="$ROOT/Baseline/paper_baselines/external/Grounded-SAM-2"

if [ "${1:-}" != "--check" ]; then
  "$UV" python install --no-bin --install-dir "$ROOT/.python" 3.11
  # the checkout's own interpreter by path (`uv python find` may return an active venv's python)
  PYTHON_BIN="$(readlink -f "$ROOT/.python/cpython-3.11-linux-x86_64-gnu/bin/python3.11")"
  case "$PYTHON_BIN" in "$ROOT/.python/"*) ;; *) echo "Python 3.11 resolved outside this checkout: $PYTHON_BIN" >&2; exit 2;; esac
  [ -x "$PY" ] || "$UV" venv --python "$PYTHON_BIN" "$ENV_DIR"
  REQ="$IN"; [ -f "$LOCK" ] && REQ="$LOCK"
  echo "[setup] installing from $REQ"
  "$UV" pip install --link-mode copy --python "$PY" -r "$REQ"
  [ -d "$GSAM2/sam2" ] || python3 -I "$ROOT/Baseline/paper_baselines/external/fetch_sources.py" --only Grounded-SAM-2
  SAM2_BUILD_CUDA=0 "$UV" pip install --link-mode copy --python "$PY" --no-build-isolation --no-deps -e "$GSAM2"
  if [ "${1:-}" = "--freeze" ] || [ ! -f "$LOCK" ]; then
    { echo "# Full lock of .venv-baselines, written by setup_venv.sh ($(date -u +%Y-%m-%dT%H:%M:%SZ))."
      echo "# The editable sam2 (external/Grounded-SAM-2 @ pinned SHA) is excluded; setup_venv.sh installs it separately."
      "$UV" pip freeze --python "$PY" | grep -vE '^(-e |sam-2|sam2)' ; } > "$LOCK.tmp"
    mv "$LOCK.tmp" "$LOCK"
    echo "[setup] wrote $LOCK"
  fi
fi

CUDA_VISIBLE_DEVICES="" "$PY" - <<'EOF'
import importlib, json
mods = ["torch", "torchvision", "transformers", "sam2", "ultralytics", "scipy", "skimage", "cv2", "PIL", "numpy", "timm", "pytest"]
out = {}
for m in mods:
    try:
        mod = importlib.import_module(m)
        out[m] = getattr(mod, "__version__", "ok")
    except Exception as exc:  # report, do not hide
        out[m] = f"IMPORT FAILED: {type(exc).__name__}: {exc}"
import torch
out["torch.version.cuda"] = torch.version.cuda
print(json.dumps(out, indent=1))
EOF
