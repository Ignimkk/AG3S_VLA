"""Interim check of the baseline tranche against the original papers (dev-4a, 2026-11-10; user ruling 01:3x KST:
"keep the current ports; check the running results midway and apply the improvement only if they are far below the paper").

Reads closed_loop_metrics.json of every finished tranche run (``bl<T>-<arm>-<suite>-L<lvl>[-rN]``; the highest -rN rerun
supersedes), pools episodes per arm x level over the suites finished so far, and compares with the paper values.

Flag rule (fixed before looking at any tranche result, doc §4.8):
  * harness sanity: pi0.5 alone within 10 pt of the KNOWS paper's pi0.5 row (else the harness, not the port, is suspect);
  * an arm is FLAGGED when its pooled CAR or SSR is more than 20 pt below its paper reference AND it has >= 64 episodes on
    that level (>= 2 suites; at 64 ep one binomial SE is <= 6.3 pt, so 20 pt is > 3 SE).
Paper references (%; KNOWS paper Tab. 1 four-suite mean; VLSA Tab. I suite mean, levels pooled, no SSR):
  none        LI 55.0 / 11.5 / 12.4  LII 57.8 / 20.0 / 23.0   (TSR / SSR / CAR)
  knows       LI 67.5 / 59.9 / 76.4  LII 71.1 / 53.8 / 68.8
  knows-gt    = KNOWS paper "Naive (GT segmentation)": LI 73.0 / 63.1 / 82.2  LII 67.8 / 55.6 / 76.1
  aegis, aegis-gt  TSR 67.5, CAR 68.9 (both levels, pooled)
  multilink_cbf(-gt)  multi-link paper Table Ib static SafeLIBERO, levels pooled: 60.3 / 51.7 / 73.0 (their pi0.5 60.0 / 15.1 / 16.6)
usage: .venv-ag3s/bin/python experiments/segmentation/bl_tranche_check.py [--tag t1] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

ROOT = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
PAPER = {
    "none": {"I": (55.0, 11.5, 12.4), "II": (57.8, 20.0, 23.0)},
    "knows": {"I": (67.5, 59.9, 76.4), "II": (71.1, 53.8, 68.8)},
    "knows-gt": {"I": (73.0, 63.1, 82.2), "II": (67.8, 55.6, 76.1)},
    "aegis": {"pooled": (67.5, None, 68.9)},
    "aegis-gt": {"pooled": (67.5, None, 68.9)},
    # multi-link CBF paper Table Ib, static SafeLIBERO, levels pooled ("Ours"; audit 2026-11-10). The GT-seg arm is compared
    # with the same row (the paper's privileged "exact geometry" row is not used: our GT arm is GT segmentation)
    "multilink_cbf": {"pooled": (60.3, 51.7, 73.0)},
    "multilink_cbf-gt": {"pooled": (60.3, 51.7, 73.0)},
}
GAP_PT, MIN_EP, SANITY_PT = 20.0, 64, 10.0


def runs(tag: str) -> dict:
    pat = re.compile(rf"^(\d{{8}}T\d{{6}})_[0-9a-f]+_bl{tag}-(.+)-(spatial|object|goal|long)-L(I|II)(-r[0-9])?-")
    best = {}
    for d in ROOT.iterdir():
        m = pat.match(d.name)
        if not m or d.name.startswith("INVALID") or not (d / "closed_loop_metrics.json").exists():
            continue
        stamp, arm, suite, lvl, r2 = m.groups()
        key = (arm, suite, lvl)
        rank = (int(r2[2:]) if r2 else 1, stamp)  # -r3 > -r2 > first run
        if key not in best or rank > best[key][0]:
            best[key] = (rank, d)
    return {k: v[1] for k, v in best.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t1")
    ap.add_argument("--json")
    a = ap.parse_args()
    pooled: dict = {}
    for (arm, suite, lvl), d in sorted(runs(a.tag).items()):
        m = json.loads((d / "closed_loop_metrics.json").read_text())
        o, s = m["safelibero_official"], m["sl5c"]
        p = pooled.setdefault((arm, lvl), {"n": 0, "succ": 0, "coll": 0, "safe": 0, "suites": []})
        p["n"] += o["n"]; p["succ"] += o["n_success"]; p["coll"] += o["n_collision"]; p["safe"] += s["safe_success"]
        p["suites"].append(suite)
    for arm in ("aegis", "aegis-gt", "multilink_cbf", "multilink_cbf-gt"):  # VLSA / multi-link report with the levels pooled -> compare on the pooled row
        lv = [pooled[(arm, l)] for l in ("I", "II") if (arm, l) in pooled]
        if lv:
            pooled[(arm, "pooled")] = {k: sum(x[k] for x in lv) for k in ("n", "succ", "coll", "safe")} | {
                "suites": sorted({s for x in lv for s in x["suites"]})}
    out, flags = [], []
    for (arm, lvl), p in sorted(pooled.items()):
        n = p["n"]
        tsr, car, ssr = 100 * p["succ"] / n, 100 * (n - p["coll"]) / n, 100 * p["safe"] / n
        ref = PAPER.get(arm, {})
        r = ref.get(lvl) if lvl != "pooled" or "pooled" in ref else None
        if arm.startswith(("aegis", "multilink")) and lvl != "pooled":
            r = None  # per-level AEGIS rows are shown, the flag uses the pooled row
        row = {"arm": arm, "level": lvl, "n": n, "suites": p["suites"], "TSR": round(tsr, 1), "SSR": round(ssr, 1),
               "CAR": round(car, 1), "paper": r}
        if r:
            gaps = {"CAR": car - r[2]} | ({"SSR": ssr - r[1]} if r[1] is not None else {})
            row["gap_pt"] = {k: round(v, 1) for k, v in gaps.items()}
            if arm == "none":
                row["sanity_ok"] = n < MIN_EP or abs(tsr - r[0]) <= SANITY_PT
            elif n >= MIN_EP and any(v < -GAP_PT for v in gaps.values()):
                row["FLAG"] = True
                flags.append(f"{arm} L{lvl}")
        out.append(row)
    print(f"{'arm':<18}{'lvl':<4}{'n':>4}  {'TSR':>5} {'SSR':>5} {'CAR':>5}   paper(TSR/SSR/CAR)   gap   suites")
    for r in out:
        print(f"{r['arm']:<18}{r['level']:<4}{r['n']:>4}  {r['TSR']:>5} {r['SSR']:>5} {r['CAR']:>5}   {str(r['paper']):<20} "
              f"{r.get('gap_pt', '')} {'FLAG' if r.get('FLAG') else ''}{'' if r.get('sanity_ok', True) else 'SANITY-FAIL'}"
              f"  {','.join(r['suites'])}")
    print("FLAGGED:", ", ".join(flags) if flags else "none")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"rows": out, "flags": flags}, indent=1))


if __name__ == "__main__":
    main()
