#!/usr/bin/env bash
# source scripts/setup/env.sh from either independent checkout.
AG3S_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export AG3S_PROJECT_ROOT
export AG3S_STORAGE_ROOT="${AG3S_STORAGE_ROOT:-$(dirname "$AG3S_PROJECT_ROOT")/AG3S_VLA-storage}"
# Derived compatibility alias, never an old source checkout.
export AG3S_LEGACY_STORAGE="$AG3S_STORAGE_ROOT/legacy-inputs"
export AG3S_CHECKPOINT_ROOT="$AG3S_STORAGE_ROOT/checkpoints"
export AG3S_DATASET_ROOT="$AG3S_STORAGE_ROOT/datasets"
# Retire inherited paths from an old terminal while keeping other explicit overrides.
for AG3S_PATH_VAR in OPENPI_DATA_HOME LIBERO_CONFIG_PATH HF_LEROBOT_HOME LEROBOT_HOME HF_DATASETS_CACHE HF_HOME TRANSFORMERS_CACHE AG3S_OPENPI_PYTHON AG3S_CLIENT_SCRIPT AG3S_ASSET_ROOT; do
  case "${!AG3S_PATH_VAR:-}" in
    /mnt/dev/work|/mnt/dev/work/*|/work|/work/*|/home/mk/dev_ws/vla/pi0_TO_ws|/home/mk/dev_ws/vla/pi0_TO_ws/*) unset "$AG3S_PATH_VAR" ;;
  esac
done
unset AG3S_PATH_VAR
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$AG3S_STORAGE_ROOT/datasets/rby1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$AG3S_PROJECT_ROOT/.cache/huggingface/datasets}"
export AG3S_OPENPI_PYTHON="${AG3S_OPENPI_PYTHON:-$AG3S_PROJECT_ROOT/.venv-openpi-live/bin/python}"
export PYTHONPATH="$AG3S_PROJECT_ROOT:$AG3S_PROJECT_ROOT/integration/rby1/manipulation/src:$AG3S_PROJECT_ROOT/third_party/openpi/src:$AG3S_PROJECT_ROOT/third_party/openpi/packages/openpi-client/src"
export RBY1_DESCRIPTION_ROOT="$AG3S_PROJECT_ROOT/robots/rby1/description"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$AG3S_PROJECT_ROOT/.cache/openpi}"
export HF_HOME="${HF_HOME:-$AG3S_PROJECT_ROOT/.cache/huggingface}"
export XDG_CACHE_HOME="$AG3S_PROJECT_ROOT/.cache"
export WARP_CACHE_PATH="$AG3S_PROJECT_ROOT/.cache/warp"
export TORCH_EXTENSIONS_DIR="$AG3S_PROJECT_ROOT/.cache/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$AG3S_PROJECT_ROOT/.cache/torch-inductor"
export CUDA_CACHE_PATH="$AG3S_PROJECT_ROOT/.cache/cuda"
export JAX_COMPILATION_CACHE_DIR="$AG3S_PROJECT_ROOT/.cache/jax"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$AG3S_PROJECT_ROOT/.runtime/libero}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
mkdir -p "$LIBERO_CONFIG_PATH" "$AG3S_PROJECT_ROOT/outputs"
# A user-level python3.11 shim may still resolve into the retired workspace.
# Shadow only that name with this checkout's interpreter; an active venv keeps precedence.
case "$(readlink -f "$(command -v python3.11 2>/dev/null)" 2>/dev/null)" in
  /mnt/dev/work/*)
    if [ -x "$AG3S_PROJECT_ROOT/.python/cpython-3.11-linux-x86_64-gnu/bin/python3.11" ]; then
      mkdir -p "$AG3S_PROJECT_ROOT/.runtime/bin"
      ln -sfn "$AG3S_PROJECT_ROOT/.python/cpython-3.11-linux-x86_64-gnu/bin/python3.11" "$AG3S_PROJECT_ROOT/.runtime/bin/python3.11"
      export PATH="$AG3S_PROJECT_ROOT/.runtime/bin:$PATH"
    fi ;;
esac
cd "$AG3S_PROJECT_ROOT"
