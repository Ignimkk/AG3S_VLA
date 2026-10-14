#!/usr/bin/env bash
# SafeLIBERO (benchmark/SafeLIBERO, the vlsa-aegis fork of LIBERO) config for .venv-libero.
# The fork uses the package name `libero`; benchmark/runners/libero_safe_client.py puts the fork first on
# sys.path and points LIBERO_CONFIG_PATH here, so this config must name the fork's files, not LIBERO's.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG_DIR="${SAFELIBERO_CONFIG_PATH:-$ROOT/.runtime/safelibero}"
PKG="$ROOT/benchmark/SafeLIBERO/safelibero/libero/libero"
[ -d "$PKG/bddl_files/safelibero_spatial" ] || { echo "SafeLIBERO submodule missing: $PKG" >&2; exit 1; }
mkdir -p "$CFG_DIR"
TMP="$(mktemp "$CFG_DIR/.config.XXXXXX")"
cat > "$TMP" <<EOT
benchmark_root: $PKG
bddl_files: $PKG/bddl_files
init_states: $PKG/init_files
datasets: ${AG3S_STORAGE_ROOT:-$ROOT/../AG3S_VLA-storage}/datasets/safelibero
assets: $PKG/assets
EOT
mv "$TMP" "$CFG_DIR/config.yaml"
echo "$CFG_DIR/config.yaml"
