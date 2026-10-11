#!/usr/bin/env bash
# source scripts/setup/env.sh from either independent checkout.
AG3S_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export AG3S_PROJECT_ROOT
export AG3S_LEGACY_STORAGE="${AG3S_LEGACY_STORAGE:-/mnt/dev/work}"
export PYTHONPATH="$AG3S_PROJECT_ROOT:$AG3S_PROJECT_ROOT/integration/rby1/manipulation/src:$AG3S_PROJECT_ROOT/third_party/openpi/src:$AG3S_PROJECT_ROOT/third_party/openpi/packages/openpi-client/src"
export RBY1_DESCRIPTION_ROOT="$AG3S_PROJECT_ROOT/robots/rby1/description"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$AG3S_PROJECT_ROOT/.cache/openpi}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$AG3S_PROJECT_ROOT/.runtime/libero}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
mkdir -p "$LIBERO_CONFIG_PATH" "$AG3S_PROJECT_ROOT/outputs"
cd "$AG3S_PROJECT_ROOT"
