#!/usr/bin/env bash
# source Baseline/paper_baselines/setup/paper_baselines_env.sh — paths for .venv-baselines runs.
# Caches and weights stay inside this checkout (.cache/paper_baselines), never in shared storage or ~.
_PB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$_PB_ROOT/scripts/setup/env.sh" >/dev/null
export PAPER_BASELINES_CACHE="$_PB_ROOT/.cache/paper_baselines"
export HF_HOME="$PAPER_BASELINES_CACHE/huggingface"
export TORCH_HOME="$PAPER_BASELINES_CACHE/torch"
export YOLO_CONFIG_DIR="$PAPER_BASELINES_CACHE/ultralytics_config"
export PAPER_BASELINES_WEIGHTS="$PAPER_BASELINES_CACHE/weights"
export AG3S_BASELINES_PYTHON="$_PB_ROOT/.venv-baselines/bin/python"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$YOLO_CONFIG_DIR" "$PAPER_BASELINES_WEIGHTS"
unset _PB_ROOT
# ultralytics: never pip-install at runtime, keep settings/weights inside the checkout cache
export YOLO_AUTOINSTALL=False
