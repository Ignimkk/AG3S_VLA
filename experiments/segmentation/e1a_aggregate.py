#!/usr/bin/env python3
"""E1a SafeLIBERO baseline R aggregation (protocol paper/experiments/configs/
E1_SAFELIBERO_LIBEROSAFETY_BASELINE_PROTOCOL_20261104_KO.md §2.5, fixed before results).

Runs: storage ``results/benchmark/*_e1a-<arm>-<suite>-L<level>-*`` (launch_method.py, budget main_aegis, test init 5-14).
Per (suite, level, arm) cell: n, TSR (official success), CAR (no collision), SSR (success and no collision, per episode),
ETS (mean env steps excluding wait over all episodes), SBU (success with collision), CNF (no collision, not finished),
Wilson 95% CI. Arm vs VLA-only (`none`): episodes paired on (suite, level, task, init); discordant counts, exact two-sided
McNemar, and a scenario-cluster bootstrap (scenario = suite x level x task; 2000 resamples, seed 20261104) 95% CI of the
paired SSR / TSR / CAR difference. 4-suite averages weight suites equally (mean of suite rates). Missing episodes are
reported, never counted as failures or successes.

usage: python3 experiments/segmentation/e1a_aggregate.py --out DIR
"""
import argparse
import json
import math
import pathlib
import random
import re

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
ARMS = ["none", "aegis", "aegis-gt", "multilink_cbf", "multilink_cbf-gt", "knows", "knows-gt"]
LABEL = {"none": "π0.5 only", "aegis": "AEGIS port (detector)", "aegis-gt": "AEGIS port (privileged names)",
         "multilink_cbf": "multi-link CBF (detector)", "multilink_cbf-gt": "multi-link CBF (privileged GT)",
         "knows": "KNOWS port (detector)", "knows-gt": "KNOWS port (privileged GT)"}
SUITES = ["spatial", "object", "goal", "long"]
LEVELS = ["I", "II"]
RUN_RE = re.compile(r"_e1a-(?P<arm>[a-z_]+(?:-gt)?)-(?P<suite>spatial|object|goal|long)-L(?P<level>I{1,2})-")
EXPECTED_N = 40   # 4 tasks x init 5-14


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load():
    eps = {}   # (arm, suite, level) -> {(task, init): row}
    runs = {}
    for run in sorted(RESULTS.glob("*_e1a-*")):
        m = RUN_RE.search(run.name)
        f = run / "episodes.jsonl"
        if not m or not f.exists():
            continue
        key = (m["arm"], m["suite"], m["level"])
        if key in runs:
            raise SystemExit(f"two runs for {key}: {runs[key]} and {run.name} -- resolve by hand")
        runs[key] = run.name
        d = eps.setdefault(key, {})
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            sl = r.get("safelibero") or {}
            d[(int(r["task_id"]), int(r["episode_idx"]))] = {
                "success": bool(r["official_success"]), "collided": bool(sl.get("collided")),
                "steps": int(r["env_steps_excluding_wait"]), "status": r.get("status"), "error": r.get("error")}
    return eps, runs


def cell_stats(rows):
    n = len(rows)
    s = sum(r["success"] for r in rows)
    nc = sum(not r["collided"] for r in rows)
    ss = sum(r["success"] and not r["collided"] for r in rows)
    sbu = sum(r["success"] and r["collided"] for r in rows)
    cnf = sum((not r["success"]) and not r["collided"] for r in rows)
    ets = sum(r["steps"] for r in rows) / n if n else float("nan")
    return {"n": n, "TSR": s, "CAR": nc, "SSR": ss, "SBU": sbu, "CNF": cnf, "ETS": ets,
            "errors": sum(1 for r in rows if r["error"]),
            "ci": {k: wilson(v, n) for k, v in (("TSR", s), ("CAR", nc), ("SSR", ss))}}


def paired(eps, arm, suite, level):
    a, b = eps.get((arm, suite, level), {}), eps.get(("none", suite, level), {})
    keys = sorted(set(a) & set(b))
    out = {"n_pairs": len(keys)}
    for met, f in (("SSR", lambda r: r["success"] and not r["collided"]), ("TSR", lambda r: r["success"]),
                   ("CAR", lambda r: not r["collided"])):
        only_arm = sum(f(a[k]) and not f(b[k]) for k in keys)
        only_vla = sum(f(b[k]) and not f(a[k]) for k in keys)
        out[met] = {"only_arm": only_arm, "only_vla": only_vla, "mcnemar_p": mcnemar_exact(only_arm, only_vla)}
    return out, keys


def bootstrap(eps, arm, cells, met_f, n_boot=2000, seed=20261104):
    """Scenario-cluster bootstrap of the pooled paired difference (arm - none) over the given (suite, level) cells."""
    clusters = []
    for suite, level in cells:
        a, b = eps.get((arm, suite, level), {}), eps.get(("none", suite, level), {})
        for t in sorted({k[0] for k in set(a) & set(b)}):
            ks = [k for k in sorted(set(a) & set(b)) if k[0] == t]
            clusters.append([met_f(a[k]) - met_f(b[k]) for k in ks])
    if not clusters:
        return None
    rng = random.Random(seed)
    tot = lambda cs: sum(sum(c) for c in cs) / max(1, sum(len(c) for c in cs))
    est = tot(clusters)
    bs = sorted(tot([clusters[rng.randrange(len(clusters))] for _ in clusters]) for _ in range(n_boot))
    return {"diff": est, "ci95": (bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]), "n_clusters": len(clusters)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    eps, runs = load()
    res = {"runs": {"/".join(k): v for k, v in runs.items()}, "cells": {}, "paired_vs_none": {}, "suite_avg": {},
           "bootstrap_vs_none": {}}
    md = ["# E1a SafeLIBERO baseline R (test init 5-14, budget main_aegis)", "",
          "| suite | level | arm | n | TSR | CAR | **SSR** | ETS | SBU | CNF | SSR 95% CI | vs π0.5 SSR (arm only / π0.5 only, p) |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for suite in SUITES:
        for level in LEVELS:
            for arm in ARMS:
                d = eps.get((arm, suite, level))
                if not d:
                    continue
                st = cell_stats(list(d.values()))
                res["cells"][f"{suite}/{level}/{arm}"] = st
                pr = ""
                if arm != "none" and ("none", suite, level) in eps:
                    p, _ = paired(eps, arm, suite, level)
                    res["paired_vs_none"][f"{suite}/{level}/{arm}"] = p
                    pr = f"{p['SSR']['only_arm']} / {p['SSR']['only_vla']}, p={p['SSR']['mcnemar_p']:.3g}"
                lo, hi = st["ci"]["SSR"]
                flag = "" if st["n"] == EXPECTED_N else f" (incomplete {st['n']}/{EXPECTED_N})"
                md.append(f"| {suite} | {level} | {LABEL[arm]}{flag} | {st['n']} | {st['TSR']} | {st['CAR']} | **{st['SSR']}** | "
                          f"{st['ETS']:.1f} | {st['SBU']} | {st['CNF']} | [{lo:.2f}, {hi:.2f}] | {pr} |")
    # 4-suite averages (equal suite weight) where all 4 suites are complete for that level x arm
    md += ["", "## 4-suite average (equal suite weight; only arms with all 4 suites complete)", "",
           "| level | arm | TSR % | SSR % | CAR % | ETS | paired SSR diff vs π0.5 [cluster-bootstrap 95% CI] |", "|---|---|---:|---:|---:|---:|---|"]
    for level in LEVELS:
        for arm in ARMS:
            cs = [res["cells"].get(f"{s}/{level}/{arm}") for s in SUITES]
            if not all(c and c["n"] == EXPECTED_N for c in cs):
                continue
            avg = {m: 100 * sum(c[m] / c["n"] for c in cs) / 4 for m in ("TSR", "SSR", "CAR")}
            ets = sum(c["ETS"] for c in cs) / 4
            res["suite_avg"][f"{level}/{arm}"] = {**avg, "ETS": ets}
            bt = ""
            if arm != "none":
                b = bootstrap(eps, arm, [(s, level) for s in SUITES], lambda r: int(r["success"] and not r["collided"]))
                res["bootstrap_vs_none"][f"{level}/{arm}"] = b
                if b:
                    bt = f"{100 * b['diff']:+.1f} pp [{100 * b['ci95'][0]:+.1f}, {100 * b['ci95'][1]:+.1f}] ({b['n_clusters']} scenarios)"
            md.append(f"| {level} | {LABEL[arm]} | {avg['TSR']:.1f} | {avg['SSR']:.1f} | {avg['CAR']:.1f} | {ets:.1f} | {bt} |")
    md += ["", "TSR/CAR/SSR/SBU/CNF are episode counts out of n (40 per cell). ETS = mean env steps excluding the 20 wait "
           "steps over all episodes (failures run to the 300 / 550 policy-step budget). SSR is computed per episode. "
           "Exploratory: no multiplicity correction; a non-significant difference is not equivalence."]
    (a.out / "e1a_summary.json").write_text(json.dumps(res, indent=1, default=list))
    (a.out / "e1a_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
