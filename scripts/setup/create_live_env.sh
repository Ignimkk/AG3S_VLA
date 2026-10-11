#!/usr/bin/env bash
# Recreate own-root live packages; never copy or activate a legacy venv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/setup/env.sh"
UV="${AG3S_UV:-$AG3S_STORAGE_ROOT/tools/uv}"
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.python"
"$UV" python install --no-bin --install-dir "$ROOT/.python" 3.11.16
PYTHON_BIN="$("$UV" python find --managed-python 3.11.16)"
ENV_DIR="$ROOT/${1:-.venv-openpi-live}"
case "$ENV_DIR" in "$ROOT/.venv-openpi-live"|"$ROOT/.venv-curobo") ;; *) echo 'Use .venv-openpi-live or .venv-curobo' >&2; exit 2;; esac
[ -x "$ENV_DIR/bin/python" ] || "$UV" venv --python "$PYTHON_BIN" "$ENV_DIR"
if [ "$(basename "$ENV_DIR")" = .venv-curobo ]; then
  "$UV" pip install --link-mode copy --python "$ENV_DIR/bin/python" -r "$ROOT/environments/curobo/live-requirements.txt" -c "$ROOT/environments/openpi-live/inventory-constraints.txt" -c "$ROOT/environments/curobo/validated-constraints.txt"
  "$UV" pip install --link-mode copy --python "$ENV_DIR/bin/python" --no-deps -e "$ROOT" -e "$ROOT/third_party/curobo"
else
  "$UV" pip install --link-mode copy --python "$ENV_DIR/bin/python" -r "$ROOT/environments/openpi-live/requirements.txt" -c "$ROOT/environments/openpi-live/inventory-constraints.txt" -c "$ROOT/environments/openpi-live/validated-constraints.txt"
  "$UV" pip install --link-mode copy --python "$ENV_DIR/bin/python" --no-deps -e "$ROOT" -e "$ROOT/third_party/openpi" -e "$ROOT/third_party/openpi/packages/openpi-client" -e "$ROOT/third_party/curobo" -e "$ROOT/integration/rby1/manipulation"
  "$UV" pip install --link-mode copy --python "$ENV_DIR/bin/python" --no-deps gym-aloha==0.1.1
  # Frozen upstream LeRobot source is fetched into this checkout's private uv cache.
  "$UV" pip install --link-mode copy --python "$ENV_DIR/bin/python" --no-deps 'lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'
fi
