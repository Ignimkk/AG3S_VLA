#!/usr/bin/env bash
# source scripts/setup/env.sh from either independent checkout.
AG3S_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export AG3S_PROJECT_ROOT
export AG3S_STORAGE_ROOT="${AG3S_STORAGE_ROOT:-$(dirname "$AG3S_PROJECT_ROOT")/AG3S_VLA-storage}"
# Derived compatibility alias, never an old source checkout.
export AG3S_LEGACY_STORAGE="$AG3S_STORAGE_ROOT/legacy-inputs"
export AG3S_CHECKPOINT_ROOT="$AG3S_STORAGE_ROOT/checkpoints"
export AG3S_DATASET_ROOT="$AG3S_STORAGE_ROOT/datasets"
export AG3S_RESULTS_ROOT="$AG3S_STORAGE_ROOT/results/development"
export AG3S_CUROBO_PYTHON="$AG3S_PROJECT_ROOT/.venv-curobo/bin/python"
export CUDA_CACHE_PATH="$AG3S_PROJECT_ROOT/.cache/cuda"
export WARP_CACHE_PATH="$AG3S_PROJECT_ROOT/.cache/warp"
export TORCH_EXTENSIONS_DIR="$AG3S_PROJECT_ROOT/.cache/torch-extensions"
# Retire inherited paths from an old terminal while keeping other explicit overrides.
for AG3S_PATH_VAR in OPENPI_DATA_HOME LIBERO_CONFIG_PATH HF_LEROBOT_HOME LEROBOT_HOME HF_DATASETS_CACHE HF_HOME TRANSFORMERS_CACHE AG3S_OPENPI_PYTHON AG3S_CLIENT_SCRIPT AG3S_ASSET_ROOT; do
  case "${!AG3S_PATH_VAR:-}" in
    /mnt/dev/work|/mnt/dev/work/*|/work|/work/*|/home/mk/dev_ws/vla/pi0_TO_ws|/home/mk/dev_ws/vla/pi0_TO_ws/*) unset "$AG3S_PATH_VAR" ;;
  esac
done
unset AG3S_PATH_VAR
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$AG3S_STORAGE_ROOT/datasets/rby1}"
export HF_HOME="${HF_HOME:-$AG3S_PROJECT_ROOT/.cache/huggingface}"
export JAX_COMPILATION_CACHE_DIR="$AG3S_PROJECT_ROOT/.cache/jax"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$AG3S_PROJECT_ROOT/.cache/huggingface/datasets}"
export AG3S_OPENPI_PYTHON="${AG3S_OPENPI_PYTHON:-$AG3S_PROJECT_ROOT/.venv-openpi-live/bin/python}"
export PYTHONPATH="$AG3S_PROJECT_ROOT:$AG3S_PROJECT_ROOT/integration/rby1/manipulation/src:$AG3S_PROJECT_ROOT/third_party/openpi/src:$AG3S_PROJECT_ROOT/third_party/openpi/packages/openpi-client/src"
export RBY1_DESCRIPTION_ROOT="$AG3S_PROJECT_ROOT/robots/rby1/description"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$AG3S_PROJECT_ROOT/.cache/openpi}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$AG3S_PROJECT_ROOT/.runtime/libero}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
mkdir -p "$LIBERO_CONFIG_PATH" "$CUDA_CACHE_PATH" "$WARP_CACHE_PATH" "$TORCH_EXTENSIONS_DIR"
cd "$AG3S_PROJECT_ROOT"
