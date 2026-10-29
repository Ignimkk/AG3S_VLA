"""AG3S-PRE-CORE-TO D1 — the serving process limits its BLAS pools (`--blas-threads`, default 1).

  - every loaded OpenBLAS (numpy's and scipy's) reports the requested count afterwards, and can be set back;
  - the flag defaults to 1, accepts 0 (library default) and is parsed as an int.
"""

from __future__ import annotations

from AG3S.trajopt import serve_safe
from AG3S.trajopt.blas_threads import limit_blas_threads


def test_every_loaded_openblas_is_set_and_restorable():
    first = limit_blas_threads(1)
    assert first, "no OpenBLAS found in this process"
    assert all(r["after"] == 1 for r in first)
    original = max(r["before"] for r in first)
    try:
        again = limit_blas_threads(2)
        assert all(r["before"] == 1 and r["after"] == 2 for r in again)
    finally:
        limit_blas_threads(original)


def test_flag_default_and_zero():
    base = ["--config", "c", "--checkpoint", "k", "--model-xml", "m"]
    ap = serve_safe.build_parser()
    assert ap.parse_args(base).blas_threads == 1
    assert ap.parse_args(base + ["--blas-threads", "0"]).blas_threads == 0
    assert ap.parse_args(base + ["--blas-threads", "4"]).blas_threads == 4
