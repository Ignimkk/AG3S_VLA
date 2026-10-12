#!/usr/bin/env bash
# Fixed CPU regression for the Panda benchmark branch (SL1 D.10). No GPU, no server, no storage writes.
#
#   bash scripts/testing/run_cpu_regression.sh [extra pytest args]
#
# Suite A -- the 5 files the SL0 verifier reconstructed for the 73-test live-environment run
#            (.venv-openpi-live through run_without_legacy.py, old-checkout reads denied).
# Suite B -- tests/benchmark under .venv-openpi-live (same guard; includes the serve_policy test).
# Suite C -- tests/benchmark under .venv-ag3s (the interpreter SL0 used; openpi-dependent test skips).
# tests/benchmark spawns .venv-libero once (tests/benchmark/libero_py38_checks.py, ~20 s, ~1 GB).
# Each suite prints its own pytest summary line; the script exits non-zero if any suite fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/setup/env.sh" >/dev/null
cd "$ROOT"

SUITE_A=(tests/test_run_artifacts.py tests/test_policy_recorder_destination.py tests/ag3s/test_record_timing.py
         tests/ag3s/test_frame_record.py tests/trajopt/test_t43ta_path_shape.py)
PYTEST=(-m pytest -q -p no:cacheprovider -rs)
status=0

echo "== A: SL0 5-file regression (.venv-openpi-live, run_without_legacy) =="
"$ROOT/.venv-openpi-live/bin/python" scripts/setup/run_without_legacy.py "${PYTEST[@]}" "${SUITE_A[@]}" "$@" || status=1

echo "== B: tests/benchmark (.venv-openpi-live, run_without_legacy) =="
"$ROOT/.venv-openpi-live/bin/python" scripts/setup/run_without_legacy.py "${PYTEST[@]}" tests/benchmark "$@" || status=1

echo "== C: tests/benchmark (.venv-ag3s, run_without_legacy) =="
"$ROOT/.venv-ag3s/bin/python" scripts/setup/run_without_legacy.py "${PYTEST[@]}" tests/benchmark "$@" || status=1

echo "== cpu regression exit status: $status =="
exit "$status"
