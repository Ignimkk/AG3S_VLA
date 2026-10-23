#!/usr/bin/env bash
# Create this checkout's Python 3.10 LIBERO-Safety client environment (.venv-libero-safety). BM-SETUP-B.
# LIBERO-Safety (benchmark/LIBERO-Safety) installs a package named `libero` like LIBERO (.venv-libero) and SafeLIBERO,
# so it gets its own venv. Steps follow the LIBERO-Safety README "Installation":
#   pip install -e . ; pip install -r requirements.txt ; pip install -r extra_requirements.txt ;
#   pip install -e third_party/robosuite-1.4
# plus the openpi client (editable). Python 3.10: the code uses `X | None` annotations evaluated at def time (>=3.10),
# and transformers==4.21.1 / matplotlib==3.5.3 have no wheels for 3.11+.
# System dependency `libmagickwand` (README: apt-get install libmagickwand-dev) is NOT installed system-wide: the .deb
# files are extracted into .runtime/magickwand (apt-get download + dpkg -x; no dpkg database change) and exposed
# through MAGICK_HOME / LD_LIBRARY_PATH by scripts/setup/libero_safety_env.sh.
# Usage: bash scripts/setup/create_libero_safety_env.sh [--check]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/setup/env.sh"
UV="${AG3S_UV:-$AG3S_STORAGE_ROOT/tools/uv}"
[ -x "$UV" ] || { echo "uv not found: $UV (set AG3S_UV)" >&2; exit 1; }
# Override with AG3S_UV_CACHE_DIR when the shared NFS cache fails to unpack large wheels (torch, 2026-10-23: EBADF).
export UV_CACHE_DIR="${AG3S_UV_CACHE_DIR:-$ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="$ROOT/.python"
PY_VERSION=3.10
ENV_DIR="$ROOT/.venv-libero-safety"
LS="$ROOT/benchmark/LIBERO-Safety"
REQ="$ROOT/environments/libero-safety/requirements.in"
RECORD_DIR="$ROOT/environments/libero-safety"
TORCH_INDEX=https://download.pytorch.org/whl/cpu
MAGICK="$ROOT/.runtime/magickwand"
# robomimic==0.2.0 -> egl-probe 1.0.2 (cmake_minimum_required < 3.5); CMake >= 4 refuses without this policy floor.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

if [ "${1:-}" != "--check" ]; then
  # 1. libmagickwand, user-space (needed at import time by libero/libero/envs/env_wrapper.py)
  if [ ! -e "$MAGICK/lib/libMagickWand-6.Q16.so" ]; then
    mkdir -p "$MAGICK/debs" "$MAGICK/lib"
    ( cd "$MAGICK/debs" && apt-get download libmagickwand-6.q16-6 libmagickcore-6.q16-6 imagemagick-6-common \
        liblqr-1-0 libfftw3-double3 )
    for f in "$MAGICK"/debs/*.deb; do dpkg -x "$f" "$MAGICK/root"; done
    for f in "$MAGICK"/root/usr/lib/x86_64-linux-gnu/*.so.*; do ln -sfn "$f" "$MAGICK/lib/$(basename "$f")"; done
    ln -sfn "$MAGICK/root/usr/lib/x86_64-linux-gnu/libMagickWand-6.Q16.so.6" "$MAGICK/lib/libMagickWand-6.Q16.so"
    ln -sfn "$MAGICK/root/usr/lib/x86_64-linux-gnu/libMagickCore-6.Q16.so.6" "$MAGICK/lib/libMagickCore-6.Q16.so"
  fi
  # 2. interpreter + venv
  "$UV" python install --no-bin --install-dir "$ROOT/.python" "$PY_VERSION"
  PYTHON_BIN="$("$UV" python find --managed-python "$PY_VERSION")"
  case "$PYTHON_BIN" in "$ROOT/.python/"*) ;; *) echo "Python $PY_VERSION resolved outside this checkout: $PYTHON_BIN" >&2; exit 2;; esac
  [ -x "$ENV_DIR/bin/python" ] || "$UV" venv --python "$PYTHON_BIN" "$ENV_DIR"
  # 3. official requirements + chosen pins, then the editables (robosuite fork, LIBERO-Safety, openpi-client)
  "$UV" pip install --link-mode copy --python "$ENV_DIR/bin/python" -r "$REQ" \
    --extra-index-url "$TORCH_INDEX" --index-strategy=unsafe-best-match
  "$UV" pip install --link-mode copy --python "$ENV_DIR/bin/python" \
    -e "$LS/third_party/robosuite-1.4" -e "$ROOT/third_party/openpi/packages/openpi-client" \
    "numpy==1.26.4" "mujoco==3.2.3" --extra-index-url "$TORCH_INDEX" --index-strategy=unsafe-best-match
  "$UV" pip install --link-mode copy --python "$ENV_DIR/bin/python" --reinstall-package libero \
    --config-settings editable_mode=compat -e "$LS" "numpy==1.26.4" \
    --extra-index-url "$TORCH_INDEX" --index-strategy=unsafe-best-match
fi

# Assets (HF LIBERO-Safety/libero_safety_assets assets.zip @94ccca77, 10.67 GB zip / 15.5 GB, 902,920 files) live in
# storage; the code reads them only from <pkg>/assets (bddl_base_domain.py DIR_PATH/../assets), hence a symlink,
# hidden from the submodule's status through its local info/exclude.
ASSETS="${LIBERO_SAFETY_ASSETS:-$AG3S_STORAGE_ROOT/benchmarks/libero-safety/assets_94ccca77/assets}"
if [ -d "$ASSETS" ]; then
  ln -sfn "$ASSETS" "$LS/libero/libero/assets"
  EXCL="$(git -C "$LS" rev-parse --absolute-git-dir)/info/exclude"
  grep -qx "libero/libero/assets" "$EXCL" 2>/dev/null || echo "libero/libero/assets" >> "$EXCL"
else
  echo "[create_libero_safety_env] assets missing: $ASSETS (download + extract first, see BM-SETUP-B.impl.md)" >&2
fi

# LIBERO-Safety config (its own LIBERO_CONFIG_PATH; never ~/.libero, never .runtime/libero of .venv-libero).
CFG_DIR="${LIBERO_SAFETY_CONFIG_PATH:-$ROOT/.runtime/libero-safety}"
mkdir -p "$CFG_DIR"
PKG="$LS/libero/libero"
TMP="$(mktemp "$CFG_DIR/.config.XXXXXX")"
cat > "$TMP" <<EOC
benchmark_root: $PKG
bddl_files: $PKG/bddl_files
init_states: $PKG/init_files
datasets: $AG3S_STORAGE_ROOT/benchmarks/libero-safety/datasets
assets: $PKG/assets
EOC
mv "$TMP" "$CFG_DIR/config.yaml"

mkdir -p "$RECORD_DIR"
if [ "${1:-}" != "--check" ]; then
  {
    echo "# Validated .venv-libero-safety freeze ($("$ENV_DIR/bin/python" -V 2>&1)), $(TZ=KST-9 date '+%Y-%m-%d %H:%M KST')."
    echo "# Editable sources (LIBERO-Safety, its robosuite-1.4 fork, openpi-client) are this checkout's and excluded below."
    "$UV" pip freeze --python "$ENV_DIR/bin/python" --exclude-editable
  } > "$RECORD_DIR/validated-freeze.txt"
fi
"$UV" pip check --python "$ENV_DIR/bin/python" || echo "[create_libero_safety_env] pip check reported conflicts (recorded, not hidden)" >&2
