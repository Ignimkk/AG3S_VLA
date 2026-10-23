#!/usr/bin/env bash
# Create this checkout's .venv-baselines (Python 3.11, uv) for Baseline/paper_baselines (PAPER-BASELINES).
# Separate from .venv-ag3s (numpy 2.4.6 pipeline) and .venv-openpi-live so neither is touched.
#   - interpreter: this checkout's .python (uv-managed CPython 3.11)
#   - packages:    requirements-baselines.txt (full lock) if present, else requirements-baselines.in (top-level pins)
#   - sam2:        editable install of the pinned Grounded-SAM-2 clone (external/Grounded-SAM-2) with
#                  --no-build-isolation (re-uses the pinned torch). SAM2_BUILD_CUDA defaults to 1 (builds the
#                  connected-components CUDA op sam2._C, as lead did 2026-10-23); it needs nvcc:
#                  CUDA_HOME=/usr/local/cuda-12.6 (nvcc = $CUDA_HOME/bin/nvcc, CUDA 12.6 = torch's cu126),
#                  TORCH_CUDA_ARCH_LIST=9.0 (H100/H200, sm_90), MAX_JOBS=2 (pod CPU budget). Defaults below.
#                  SAM2_BUILD_CUDA=0 skips the op (sam2 then falls back to its python path, with a warning).
#                  Same as lead's build command:
#                    CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=2 \
#                    uv pip install --no-build-isolation --no-deps --reinstall-package sam2 -e external/Grounded-SAM-2
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
  SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-1}"
  if [ "$SAM2_BUILD_CUDA" = "1" ]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
    export MAX_JOBS="${MAX_JOBS:-2}"
    # Grounded-SAM-2 setup.py:101-104 hard-codes -gencode sm_70/75/80/86 and ignores TORCH_CUDA_ARCH_LIST, so the
    # H200 kernels (sm_90 cubin + PTX) are appended through nvcc's own env hook (lead, 2026-10-23; verify with
    # `cuobjdump --list-elf external/Grounded-SAM-2/sam2/_C.so`).
    export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:--gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90}"
    [ -x "$CUDA_HOME/bin/nvcc" ] || { echo "SAM2_BUILD_CUDA=1 needs nvcc at $CUDA_HOME/bin/nvcc (set CUDA_HOME, or SAM2_BUILD_CUDA=0)" >&2; exit 3; }
    echo "[setup] sam2 CUDA op: CUDA_HOME=$CUDA_HOME NVCC_APPEND_FLAGS=$NVCC_APPEND_FLAGS MAX_JOBS=$MAX_JOBS"
  fi
  SAM2_BUILD_CUDA="$SAM2_BUILD_CUDA" "$UV" pip install --link-mode copy --python "$PY" --no-build-isolation --no-deps \
    --reinstall-package sam2 -e "$GSAM2"
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
mods = ["torch", "torchvision", "transformers", "sam2", "sam2._C", "ultralytics", "scipy", "skimage", "cv2", "PIL", "numpy", "timm", "pytest"]
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
