"""AG3S-PRE-CORE-TO E3-b — offline settings for the GPU-batched seeds (`tp_timing`) and the summary of a pass.

Settings (`tp_timing --settings-module AG3S.trajopt.experiments.e3b_gpu_seeds --field cuda`, each on top of the run's
server config; every E1 / E2 setting of `e2_multi_seed` is included, so `E0` · `E2_16` sit in the same pass):

| name | what |
|---|---|
| `E3b_16` | `E2_16` (E0 + `--to-multi-seed 16`) + `--to-gpu-seeds curobo` (`sqp.gpu_seeds`) |
| `E3b_32` · `E3b_64` | the same with 32 / 64 seeds |
| `E3b_16_fp` | `E3b_16` + `sqp.gpu_seeds_check = "first-pass"` (checks in the choosing order, stops at the first that resolves) |
| `E1C_E3b_16` | E1-C (`--grasp-anchor 2000 --grasp-anchor-path-weight 0`) + the same (the goal term is on the GPU) |
| `SMOKE_IT1_E3b_16` | smoke only (one SQP iteration, `e2_multi_seed.SMOKE_SETTINGS`); not for the comparison |

A GPU setting records per chunk `multi_seed` as E2 does (`candidates` = the standard solve + the N re-checked GPU
results) plus `multi_seed.backend` (`curobo`, or `sqp (cuRobo failed)` with `gpu.error` when the GPU path fell back to
E2's CPU seeds) and `multi_seed.gpu` (load / optimize / check ms, cold, per-seed GPU costs, world, settings). Its
`solve_ms` includes everything. The pass records `gpu_seeds_warm_up` (the build and CUDA-graph capture done before the
first chunk, as the server does at start-up).

    python -m AG3S.trajopt.experiments.e3b_gpu_seeds summarize [--base NAME] OUT/*.json

Summary = `e2_multi_seed summarize` (same keys, same criterion: chunks where `E0` leaves > 0.1 mm; resolved ∧ goal
error ≤ 10 mm) + per setting `gpu`: the backend counts on the triggered chunks and the median / p95 / max of
`gpu.gpu_ms` (world load + L-BFGS), `gpu.optimize_ms`, `gpu.check_ms` (our full check of the N results) and
`gpu.total_ms` (+ `n_checked` for a first-pass setting).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any

import numpy as np

from AG3S.trajopt.experiments import e2_multi_seed as E2

#: Seed counts. `E3b_16` is the comparison of task §6 E3 (E2's user decision: 16 seeds, E1-C off).
SEED_COUNTS = (16, 32, 64)

SETTINGS: dict[str, dict] = dict(E2.SETTINGS)
for _n in SEED_COUNTS:
    SETTINGS[f"E3b_{_n}"] = {"sqp": {"multi_seed": _n, "gpu_seeds": "curobo"}}
SETTINGS["E3b_16_fp"] = {"sqp": {"multi_seed": 16, "gpu_seeds": "curobo", "gpu_seeds_check": "first-pass"}}
SETTINGS["E1C_E3b_16"] = {**{k: dict(v) for k, v in E2.SETTINGS["E1C"].items()},
                          "sqp": {"multi_seed": 16, "gpu_seeds": "curobo"}}
#: **Smoke only** — the GPU path on a recorded run whose standard solves mostly resolve.
SETTINGS["SMOKE_IT1_E3b_16"] = {"sqp": {"max_iterations": 1, "min_iterations": 1, "multi_seed": 16,
                                        "gpu_seeds": "curobo"}}

GPU_TIMES = ("gpu_ms", "optimize_ms", "load_ms", "check_ms", "total_ms")


def _stats(a) -> dict[str, Any]:
    if not a:
        return {"n": 0}
    return {"n": len(a), "median": round(float(np.median(a)), 1), "p95": round(float(np.percentile(a, 95)), 1),
            "max": round(float(np.max(a)), 1)}


def summarize(paths: list[str], *, base: str = "E0") -> dict[str, Any]:
    """`e2_multi_seed.summarize` + `per_setting[v]["gpu"]` (see the module docstring)."""
    out = E2.summarize(paths, base=base)
    gpu: dict[str, dict[str, Any]] = {}
    for path in paths:
        d = json.load(open(path))
        for ch in d.get("chunks", []):
            vs = ch.get("variants") or {}
            b = vs.get(base)
            if b is None or float(b.get("max_violation_mm") or 0.0) <= E2.RESOLVED_MM:
                continue
            for v, e in vs.items():
                ms = e.get("multi_seed")
                if not (isinstance(ms, dict) and ms.get("triggered") and "backend" in ms):
                    continue
                g = gpu.setdefault(v, {"backend": Counter(), "cold": 0, "n_checked": [],
                                       **{k: [] for k in GPU_TIMES}})
                g["backend"][str(ms["backend"])] += 1
                rec = ms.get("gpu") or {}
                g["cold"] += int(bool(rec.get("cold")))
                for k in GPU_TIMES:
                    if rec.get(k) is not None:
                        g[k].append(float(rec[k]))
                if isinstance(rec.get("first_pass"), dict):          # `E3b_16_fp`: how many results were checked
                    g["n_checked"].append(float(rec["first_pass"]["n_checked"]))
    for v, g in gpu.items():
        out["per_setting"].setdefault(v, {})["gpu"] = {
            "backend": dict(g["backend"]), "n_cold": g["cold"], **{k: _stats(g[k]) for k in GPU_TIMES},
            **({"n_checked": _stats(g["n_checked"])} if g["n_checked"] else {})}
    return out


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "summarize":
        raise SystemExit("usage: python -m AG3S.trajopt.experiments.e3b_gpu_seeds summarize [--base NAME] OUT.json ...")
    rest, base = argv[1:], "E0"
    if rest[:1] == ["--base"]:
        base, rest = rest[1], rest[2:]
    print(json.dumps(summarize(rest, base=base), indent=1))


if __name__ == "__main__":
    sys.exit(main())
