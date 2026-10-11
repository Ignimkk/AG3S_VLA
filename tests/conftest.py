"""Test-suite-wide settings.

T26: in unit tests an exclusion-invariant violation **raises** (`ExclusionInvariantViolation`)
instead of being refused and recorded — admissible-only exclusion geometry is a property of what
reaches `AG3S._exclusion_gate`, so a test that reaches it with an inadmissible object has found a
bug. A test that checks the production behaviour (refuse + record) passes `strict_invariants=False`.
"""

import os

os.environ.setdefault("AG3S_STRICT_INVARIANTS", "1")


def pytest_configure(config):
    # SUBTASK-c: the GPU test (checkpoint + JAX) is marked `slow` and opt-in (`AG3S_GPU_TESTS=1`).
    config.addinivalue_line("markers", "slow: needs a GPU and the checkpoint; opt-in")
