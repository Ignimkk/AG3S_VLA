#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV="${AG3S_UV:-$(command -v uv || true)}"
STORAGE="${AG3S_STORAGE_ROOT:-$(dirname "$ROOT")/AG3S_VLA-storage}"
if [ -z "$UV" ] && [ -x "$STORAGE/tools/uv" ]; then
  UV="$STORAGE/tools/uv"
fi
[ -n "$UV" ] || { echo 'Install uv or set AG3S_UV to its executable.' >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="$ROOT/.python"
"$UV" python install --no-bin --install-dir "$ROOT/.python" 3.11.16
PYTHON_BIN="$("$UV" python find --managed-python 3.11.16)"
if [ ! -x "$ROOT/.venv-ag3s/bin/python" ]; then
  "$UV" venv --python "$PYTHON_BIN" "$ROOT/.venv-ag3s"
fi
"$UV" pip install --link-mode copy --python "$ROOT/.venv-ag3s/bin/python" -r "$ROOT/environments/ag3s/requirements.txt"
"$UV" pip install --link-mode copy --python "$ROOT/.venv-ag3s/bin/python" -r "$ROOT/environments/testing/requirements.txt"
"$UV" pip install --link-mode copy --python "$ROOT/.venv-ag3s/bin/python" --no-deps -e "$ROOT"
source "$ROOT/scripts/setup/env.sh"
"$ROOT/.venv-ag3s/bin/python" "$ROOT/scripts/setup/check_workspace.py"
