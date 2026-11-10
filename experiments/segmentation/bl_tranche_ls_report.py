"""Tranche C (LIBERO-Safety) table: per (suite, level) official SR / CR + auxiliary contact columns (B1), vs paper.

B1 rule (plan §1.3): the official ``info['cost']`` decides termination and the official SR / CR as is; the client's
``geom_robot_contact`` (robot geom ids vs constraint-object geoms, policy steps) and ``full_violation`` (each
``:constraints`` predicate, B2 name-collision undone) are reported as separate columns. Because the trajectory up to
the first auxiliary contact is the same whether or not the episode is cut there, ``SR_aux`` = success and no official /
auxiliary violation equals the SR that terminating on the auxiliary detector would give (post hoc, no re-run).

    .venv-ag3s/bin/python experiments/segmentation/bl_tranche_ls_report.py --glob '*_blt1-none-ls-*-vla_only' \
        [--out outputs/2026-11-09_baseline_tranche/t1/ls_table.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

STORE = Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage")) / "results/benchmark"
ABBR = {"affordance": "AAG", "human_safety": "HRI", "obstacle_avoidance": "TSA", "obstacle_avoidance_human": "FSHOA"}
#: paper Table 3, pi0.5 SR (%) L0 / L1 / L2 (mean of 3 training seeds, 10 trials x 5 tasks each)
PAPER_SR = {"AAG": (78.7, 59.3, 35.3), "HRI": (84.7, 88.7, 83.3), "TSA": (58.0, 62.7, 56.7), "FSHOA": (55.3, 58.7, 51.3)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (100 * (c - h), 100 * (c + h))


def rows(pattern: str) -> list[dict]:
    out = []
    for d in sorted(glob.glob(str(STORE / pattern))):
        p = Path(d) / "episodes.jsonl"
        if p.exists():
            out += [dict(json.loads(l), _run=Path(d).name) for l in p.read_text().splitlines() if l.strip()]
    return out


def block(rs: list[dict]) -> dict:
    n = len(rs)
    ls = [r["libero_safety"] for r in rs]
    succ = [bool(r["official_success"]) for r in rs]
    off = [bool(x["official_violation"]) for x in ls]
    geom = [bool(x["geom_robot_contact"]) for x in ls]
    full = [bool(x["full_violation"]) for x in ls]
    sr = sum(s and not o for s, o in zip(succ, off))
    aux_any = [o or g or f for o, g, f in zip(off, geom, full)]
    sr_aux = sum(s and not a for s, a in zip(succ, aux_any))
    return {"n": n, "SR": sr, "CR": sum(off), "geom": sum(geom), "full": sum(full), "SR_aux": sr_aux,
            "CR_aux": sum(aux_any), "steps": sum(x["execution_steps"] for x in ls) / max(n, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--glob", required=True, help="run-dir glob under results/benchmark")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rs = rows(a.glob)
    keys = sorted({(r["suite"], r["level"]) for r in rs}, key=lambda k: (list(ABBR).index(k[0]), k[1]))
    lines = ["| suite | L | n | SR official % [Wilson 95%] | paper SR % | CR official % | aux geom contact % | "
             "aux per-constraint % | SR_aux % | CR_aux % | mean exec steps |", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s, lv in keys:
        b = block([r for r in rs if r["suite"] == s and r["level"] == lv])
        n = b["n"]
        lo, hi = wilson(b["SR"], n)
        pct = lambda k: f"{100 * b[k] / n:.0f}"  # noqa: E731
        lines.append(f"| {ABBR[s]} | {lv} | {n} | {pct('SR')} [{lo:.0f}, {hi:.0f}] | {PAPER_SR[ABBR[s]][lv]} | {pct('CR')} | "
                     f"{pct('geom')} | {pct('full')} | {pct('SR_aux')} | {pct('CR_aux')} | {b['steps']:.0f} |")
    ckpts = sorted({r.get("policy_checkpoint") for r in rs})
    text = (f"runs: `{a.glob}` ({len({r['_run'] for r in rs})} runs, {len(rs)} episodes), checkpoint {ckpts}\n\n"
            + "\n".join(lines) + "\n")
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
