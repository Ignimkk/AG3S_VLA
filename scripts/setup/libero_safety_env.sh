#!/usr/bin/env bash
# source scripts/setup/libero_safety_env.sh -- env.sh + the LIBERO-Safety client environment (BM-SETUP-B).
# LIBERO_CONFIG_PATH points at the LIBERO-Safety config; LD_LIBRARY_PATH is *replaced* (the pod default puts
# /usr/local/lib/python3.10/dist-packages/torch/lib first, which breaks a venv torch) by the local MagickWand libs.
_LS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$_LS_ROOT/scripts/setup/env.sh"
export LIBERO_SAFETY_CONFIG_PATH="${LIBERO_SAFETY_CONFIG_PATH:-$AG3S_PROJECT_ROOT/.runtime/libero-safety}"
export LIBERO_CONFIG_PATH="$LIBERO_SAFETY_CONFIG_PATH"
export MAGICK_HOME="$AG3S_PROJECT_ROOT/.runtime/magickwand"
export LD_LIBRARY_PATH="$MAGICK_HOME/lib"
export AG3S_LIBERO_SAFETY_PYTHON="$AG3S_PROJECT_ROOT/.venv-libero-safety/bin/python"
unset _LS_ROOT
