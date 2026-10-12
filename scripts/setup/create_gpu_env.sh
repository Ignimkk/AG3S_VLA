#!/usr/bin/env bash
# Reinstall wheels/source into independent environments; never copy an old venv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STORAGE="${AG3S_STORAGE_ROOT:-$(dirname "$ROOT")/AG3S_VLA-storage}"
UV="${AG3S_UV:-$(command -v uv || true)}"
if [ -z "$UV" ]; then UV="$STORAGE/tools/uv"; fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="$ROOT/.python"
"$UV" python install --no-bin --install-dir "$ROOT/.python" 3.11.16
PYTHON_BIN="$("$UV" python find --managed-python 3.11.16)"
if [ ! -x "$ROOT/.venv-openpi-live/bin/python" ]; then
  "$UV" venv --python "$PYTHON_BIN" "$ROOT/.venv-openpi-live"
fi
"$UV" pip install --link-mode copy --python "$ROOT/.venv-openpi-live/bin/python" --no-deps -r "$ROOT/environments/openpi-live/requirements.txt"
"$UV" pip install --link-mode copy --python "$ROOT/.venv-openpi-live/bin/python" --no-deps -e "$ROOT" -e "$ROOT/third_party/openpi" -e "$ROOT/third_party/openpi/packages/openpi-client"
# serve_safe imports cuRobo in-process (--esdf-backend curobo); the previous live env had the same editable install.
SETUPTOOLS_SCM_PRETEND_VERSION=0.8.0.post1.dev43 "$UV" pip install --link-mode copy --python "$ROOT/.venv-openpi-live/bin/python" --no-deps -e "$ROOT/third_party/curobo"
# The fork's documented Torch transformer adaptations, copied only into own venv.
"$ROOT/.venv-openpi-live/bin/python" - "$ROOT" <<'PY'
from pathlib import Path
import shutil, sys
root = Path(sys.argv[1])
dst = root / '.venv-openpi-live/lib/python3.11/site-packages/transformers'
shutil.copytree(root / 'third_party/openpi/src/openpi/models_pytorch/transformers_replace', dst, dirs_exist_ok=True)
PY
if [ "${1:-}" = "--with-curobo" ]; then
  if [ ! -x "$ROOT/.venv-curobo/bin/python" ]; then
    "$UV" venv --python "$PYTHON_BIN" "$ROOT/.venv-curobo"
  fi
  # Frozen result of requirements-live.txt constrained by the observed live versions.
  "$UV" pip install --link-mode copy --python "$ROOT/.venv-curobo/bin/python" --no-deps -r "$ROOT/environments/curobo/resolved.txt"
  SETUPTOOLS_SCM_PRETEND_VERSION=0.8.0.post1.dev43 "$UV" pip install --link-mode copy --python "$ROOT/.venv-curobo/bin/python" --no-deps -e "$ROOT/third_party/curobo"
  "$UV" pip install --link-mode copy --python "$ROOT/.venv-curobo/bin/python" --no-deps -e "$ROOT"
fi
