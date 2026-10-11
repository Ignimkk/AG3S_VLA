"""Print which `benchmark` package these tests import — the worktree and main share this tests/ tree.

Run from the worktree: `cd ${AG3S_PROJECT_ROOT}-o4 && PYTHONPATH=${AG3S_PROJECT_ROOT}-o4 python -m pytest ...`.
With cwd `${AG3S_PROJECT_ROOT}`, `python -m` puts main first on `sys.path` and every O4 test skips.
"""


def pytest_report_header(config):
    try:
        import benchmark
        return f"benchmark imported from: {benchmark.__file__}"
    except Exception as exc:  # noqa: BLE001
        return f"benchmark import failed: {exc!r}"
