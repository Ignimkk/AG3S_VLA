"""AG3S-PRE-CORE-TO D1 — limit the BLAS thread pools of the serving process.

Measured (AG3S-GPU-STAGES P0-d, 2026-10-25): with no thread setting the server's OpenBLAS pools run 64 threads on a
loaded 256-core pod, and every small matrix product of the TrajOpt solve pays for waking them. The same server under
`OPENBLAS_NUM_THREADS=1` returned bit-identical commands on 32 / 32 chunks and its TrajOpt solve median fell from
135.8 to 36.3 ms; the 18-episode replay was bit-identical on 683 / 683 chunks (119.8 → 43.4 ms, CPU time per solve
2459 → 44 ms).

The environment variable only acts when a library is loaded, which `python -m AG3S.trajopt.serve_safe` has already
done for numpy by the time `main` runs. So the pools are set through each loaded OpenBLAS's own setter instead:
numpy and scipy each bundle one (`openblas_set_num_threads64_`, `scipy_openblas_set_num_threads`). Nothing here
changes a result — OpenBLAS's per-element arithmetic does not depend on the thread count for the products measured.
"""
from __future__ import annotations

import ctypes
from typing import Any

#: (setter, getter) pairs, one per OpenBLAS build seen in this environment's wheels.
_SYMBOLS = (
    ("openblas_set_num_threads64_", "openblas_get_num_threads64_"),
    ("openblas_set_num_threads", "openblas_get_num_threads"),
    ("scipy_openblas_set_num_threads", "scipy_openblas_get_num_threads"),
    ("scipy_openblas_set_num_threads64_", "scipy_openblas_get_num_threads64_"),
)


def _loaded_openblas() -> list[str]:
    paths = []
    with open("/proc/self/maps") as f:
        for line in f:
            p = line.split()[-1]
            if p.startswith("/") and "openblas" in p.rsplit("/", 1)[-1].lower() and p not in paths:
                paths.append(p)
    return paths


def limit_blas_threads(n: int) -> list[dict[str, Any]]:
    """Set every loaded OpenBLAS pool to `n` threads; returns one record per library (path, before, after)."""
    import numpy  # noqa: F401  — load both bundled libraries before scanning
    import scipy.linalg  # noqa: F401

    out = []
    for path in _loaded_openblas():
        lib = ctypes.CDLL(path)
        for setter, getter in _SYMBOLS:
            if hasattr(lib, setter) and hasattr(lib, getter):
                before = int(getattr(lib, getter)())
                getattr(lib, setter)(ctypes.c_int(int(n)))
                out.append({"library": path.rsplit("/", 1)[-1], "before": before,
                            "after": int(getattr(lib, getter)())})
                break
    return out
