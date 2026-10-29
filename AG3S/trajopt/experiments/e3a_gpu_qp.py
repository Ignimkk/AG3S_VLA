"""AG3S-PRE-CORE-TO E3-a — offline settings for the GPU QP backend (`tp_timing`) and the summary of a pass.

Settings (`tp_timing --settings-module AG3S.trajopt.experiments.e3a_gpu_qp`, each on top of the run's server config;
every `e2_multi_seed` setting — `E0`, `E1C`, `E2_16/32/64`, … — is included, so one pass can hold the CPU references):

| name | what |
|---|---|
| `E3a_0` | E0 with `--qp-backend osqp-cuda` (`qp.solver = "osqp-cuda"`): the standard solve only, on the CUDA algebra |
| `E3a_16` | `E2_16` with `--qp-backend osqp-cuda`: the standard solve and the 16 multi-seed solves on the CUDA algebra |
| `SMOKE_IT1_E3a_0` · `SMOKE_IT1_E3a_16` | smoke only (one SQP iteration, as `e2_multi_seed`'s `SMOKE_IT1_*`) |

Run GPU-side passes with `--field cuda`. A CUDA setting records per chunk `qp_backend` (the refiner's
`to.metrics.qp_backend`: QP solves, CUDA solved / failed / raised, CUDA setups, CPU fall-backs solved / failed, failed
CUDA statuses). The CUDA extension is read from `<checkout>/.cache/osqp-cu12` or `$AG3S_OSQP_CUDA_DIR` (`trajopt/qp.py`).

    python -m AG3S.trajopt.experiments.e3a_gpu_qp summarize [--base NAME] OUT/*.json

The summary is `e2_multi_seed summarize` (same keys: over the chunks where `base` (E0) leaves a violation > 0.1 mm, the
resolved share with goal error <= 10 mm, chosen seed kinds, solve ms, multi-seed ms per triggered chunk) plus
`qp_backend[v]`, the CUDA counts summed over every chunk of each CUDA setting and over the base-violated chunks.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from AG3S.trajopt.experiments import e2_multi_seed as E2

#: The backend switch of every E3-a setting (`--qp-backend osqp-cuda`).
CUDA_QP = {"solver": "osqp-cuda"}

SETTINGS: dict[str, dict] = dict(E2.SETTINGS)
SETTINGS["E3a_0"] = {"qp": dict(CUDA_QP)}
SETTINGS["E3a_16"] = {**{k: dict(v) for k, v in E2.SETTINGS["E2_16"].items()}, "qp": dict(CUDA_QP)}
#: **Smoke only, not for the comparison** (one SQP iteration, so that the standard solve leaves violations).
SMOKE_SETTINGS: dict[str, dict] = {
    "SMOKE_IT1_E3a_0": {**{k: dict(v) for k, v in E2.SETTINGS["SMOKE_IT1_E0"].items()}, "qp": dict(CUDA_QP)},
    "SMOKE_IT1_E3a_16": {**{k: dict(v) for k, v in E2.SETTINGS["SMOKE_IT1_E2_16"].items()}, "qp": dict(CUDA_QP)},
}
SETTINGS.update(SMOKE_SETTINGS)

_COUNT_KEYS = ("qp_solves", "cuda_solved", "cuda_failed", "cuda_errors", "cuda_setups", "cpu_fallback_solved",
               "cpu_fallback_failed")


def _add(acc: dict[str, Any], rec: Any) -> None:
    if not isinstance(rec, dict):
        acc["chunks_without_record"] += 1
        return
    acc["chunks"] += 1
    for k in _COUNT_KEYS:
        acc[k] += int(rec.get(k) or 0)
    for s, n in (rec.get("failed_statuses") or {}).items():
        acc["failed_statuses"][s] = acc["failed_statuses"].get(s, 0) + int(n)


def _new() -> dict[str, Any]:
    return {"chunks": 0, "chunks_without_record": 0, **{k: 0 for k in _COUNT_KEYS}, "failed_statuses": {}}


def summarize(paths: list[str], *, base: str = "E0") -> dict[str, Any]:
    """`e2_multi_seed.summarize(paths, base=base)` + `qp_backend: {v: {"all": counts, "base_violated": counts}}` for
    every setting that carries a `qp_backend` record (the CUDA settings)."""
    out = E2.summarize(paths, base=base)
    backend: dict[str, dict[str, Any]] = {}
    for path in paths:
        d = json.load(open(path))
        for ch in d.get("chunks", []):
            vs = ch.get("variants") or {}
            b = vs.get(base)
            violated = b is not None and float(b.get("max_violation_mm") or 0.0) > E2.RESOLVED_MM
            for v, e in vs.items():
                if "qp_backend" not in e:
                    continue
                acc = backend.setdefault(v, {"all": _new(), "base_violated": _new()})
                _add(acc["all"], e.get("qp_backend"))
                if violated:
                    _add(acc["base_violated"], e.get("qp_backend"))
    out["qp_backend"] = backend
    return out


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "summarize":
        raise SystemExit("usage: python -m AG3S.trajopt.experiments.e3a_gpu_qp summarize [--base NAME] OUT.json ...")
    rest, base = argv[1:], "E0"
    if rest[:1] == ["--base"]:
        base, rest = rest[1], rest[2:]
    print(json.dumps(summarize(rest, base=base), indent=1))


if __name__ == "__main__":
    sys.exit(main())
