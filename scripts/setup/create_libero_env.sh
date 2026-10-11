#!/usr/bin/env bash
# Create this checkout's Python 3.8 LIBERO client environment (.venv-libero).
# Mirrors third_party/openpi/examples/libero/README.md ("Without Docker"):
#   uv venv --python 3.8; uv pip sync <example requirements> <LIBERO requirements>
#     --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
#   uv pip install -e packages/openpi-client; uv pip install -e third_party/libero
# Own-root interpreter and cache only; never copy or activate a legacy venv.
# Re-running converges to the same set (sync first removes the editables, which are then re-added).
# Usage: bash scripts/setup/create_libero_env.sh [--check]
#   --check  only re-run the dry-run reproduction against environments/libero/validated-freeze.txt.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/setup/env.sh"
UV="${AG3S_UV:-$AG3S_STORAGE_ROOT/tools/uv}"
[ -x "$UV" ] || { echo "uv not found: $UV (set AG3S_UV)" >&2; exit 1; }
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.python"
PY_VERSION=3.8.20
ENV_DIR="$ROOT/.venv-libero"
OFFICIAL_REQ="$ROOT/third_party/openpi/examples/libero/requirements.txt"
LIBERO_REQ="$ROOT/benchmark/LIBERO/requirements.txt"
RECORD_DIR="$ROOT/environments/libero"
TORCH_INDEX=https://download.pytorch.org/whl/cu113

if [ "${1:-}" != "--check" ]; then
  "$UV" python install --no-bin --install-dir "$ROOT/.python" "$PY_VERSION"
  PYTHON_BIN="$("$UV" python find --managed-python "$PY_VERSION")"
  case "$PYTHON_BIN" in "$ROOT/.python/"*) ;; *) echo "Python $PY_VERSION resolved outside this checkout: $PYTHON_BIN" >&2; exit 2;; esac
  [ -x "$ENV_DIR/bin/python" ] || "$UV" venv --python "$PYTHON_BIN" "$ENV_DIR"
  "$UV" pip sync --link-mode copy --python "$ENV_DIR/bin/python" "$OFFICIAL_REQ" "$LIBERO_REQ" \
    --extra-index-url "$TORCH_INDEX" --index-strategy=unsafe-best-match
  "$UV" pip install --link-mode copy --python "$ENV_DIR/bin/python" \
    -e "$ROOT/third_party/openpi/packages/openpi-client" \
    --extra-index-url "$TORCH_INDEX" --index-strategy=unsafe-best-match
  # LIBERO's top-level `libero/` has no __init__.py, so setuptools' default (strict finder) editable
  # maps no package. The official README therefore also exports PYTHONPATH=<LIBERO>; compat mode
  # puts the same directory on sys.path via a .pth so the venv itself is complete.
  "$UV" pip install --link-mode copy --python "$ENV_DIR/bin/python" --reinstall-package libero \
    --config-settings editable_mode=compat -e "$ROOT/benchmark/LIBERO" \
    --extra-index-url "$TORCH_INDEX" --index-strategy=unsafe-best-match
fi

# LIBERO asks interactive questions on first import unless the config exists.
# Paths point at this checkout's submodule; the dataset path is an (unused) storage location
# that is neither created nor written here.
mkdir -p "$LIBERO_CONFIG_PATH"
LIBERO_PKG="$ROOT/benchmark/LIBERO/libero/libero"
CONFIG="$LIBERO_CONFIG_PATH/config.yaml"
CONFIG_TMP="$(mktemp "$LIBERO_CONFIG_PATH/.config.XXXXXX")"
cat > "$CONFIG_TMP" <<EOF
benchmark_root: $LIBERO_PKG
bddl_files: $LIBERO_PKG/bddl_files
init_states: $LIBERO_PKG/init_files
datasets: $AG3S_STORAGE_ROOT/datasets/libero
assets: $LIBERO_PKG/assets
EOF
mv "$CONFIG_TMP" "$CONFIG"

# Record what was resolved; a dry-run sync against the freeze must report no change.
mkdir -p "$RECORD_DIR"
if [ "${1:-}" != "--check" ]; then
  cp "$OFFICIAL_REQ" "$RECORD_DIR/requirements.txt"
  cp "$LIBERO_REQ" "$RECORD_DIR/libero-requirements.txt"
  {
    echo "# Validated .venv-libero freeze ($("$ENV_DIR/bin/python" -V 2>&1)), $(TZ=KST-9 date '+%Y-%m-%d %H:%M KST')."
    echo "# Editable sources are this checkout's submodules and are excluded below."
    "$UV" pip freeze --python "$ENV_DIR/bin/python" --exclude-editable
  } > "$RECORD_DIR/validated-freeze.txt"
fi
# Reproduction check: syncing (no dependency resolution, extraneous packages removed) to the
# recorded freeze plus the two checkout editables must be a no-op. (uv 0.12.13: the per-package
# flag below matches the recorded build settings of libero only; the global flag used for the
# install above is applied to the build, the per-package one was not.)
CHECK_REQ="$(mktemp "$ROOT/.cache/libero-freeze-check.XXXXXX")"
trap 'rm -f "$CHECK_REQ"' EXIT
{ cat "$RECORD_DIR/validated-freeze.txt"
  echo "-e $ROOT/third_party/openpi/packages/openpi-client"
  echo "-e $ROOT/benchmark/LIBERO"; } > "$CHECK_REQ"
SYNC_OUT="$("$UV" pip sync --dry-run --python "$ENV_DIR/bin/python" "$CHECK_REQ" \
  --config-settings-package libero:editable_mode=compat \
  --extra-index-url "$TORCH_INDEX" --index-strategy=unsafe-best-match 2>&1)"
echo "$SYNC_OUT"
case "$SYNC_OUT" in *"Would make no changes"*) echo "[create_libero_env] freeze reproduction: no changes";;
  *) echo "[create_libero_env] freeze reproduction FAILED" >&2; exit 3;; esac
"$UV" pip check --python "$ENV_DIR/bin/python" || echo "[create_libero_env] pip check reported conflicts (recorded, not hidden)" >&2
