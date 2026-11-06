#!/usr/bin/env python3
"""nb1104 — aggregate every closed-loop arm of the 2026-11-04 overnight comparison on one machine.

Runs are found by run-id pattern ``*_nb1104-<arm>-L<level>-*`` in results/benchmark (launch_method.py and
launch_libero_paired.py runs both have episodes.jsonl with the same client rows). Metrics are the official SafeLIBERO
definitions of ``Baseline.paper_baselines.closed_loop.metrics`` (collision = active obstacle sum|dpos| > 1 mm after
the wait; SS = official success and no collision), fixed before any result was seen:

* TSR / CAR / SS / ETS (official), SBU = success but collided, CNF = collision-free but not successful (stopped / failed)
* paired vs the same-machine VLA-only run (``none``) on common (task, init): discordant counts and exact two-sided
  McNemar p (binomial on the discordant pairs) for SS, TSR and collision; Wilson 95 % CI per cell.
* HOLD calls / all calls (AG3S servers report ``safe_client.hold_calls``), median client policy latency (not compared
  across arms when two jobs shared the GPU).

usage: .venv-ag3s/bin/python experiments/segmentation/nb1104_aggregate.py --out outputs/<dir> [--suite safelibero_spatial]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Baseline.paper_baselines.closed_loop import metrics as M  # noqa: E402

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
LABEL = {
    "none": "VLA only (π0.5)",
    "aegis": "AEGIS-style CBF (detector)", "aegis-gt": "AEGIS-style CBF (GT obstacle names)",
    "knows": "KNOWS (detector)", "knows-gt": "KNOWS (GT obstacle seg)",
    "multilink_cbf": "Multi-link CBF (detector)", "multilink_cbf-gt": "Multi-link CBF (GT obstacle seg)",
    "omniguide": "OmniGuide (guidance)", "constrained_flow_matching": "CFM (detector depth)",
    "constrained_flow_matching-gt": "CFM (GT spheres)", "nudge": "NUDGE (guidance)",
    "lan_o3dp_guidance_adapted": "Lan-o3dp guidance",
    "ag3s-pb7": "AG3S V11AB-pb7 (pre-final Panda)", "ag3s-final": "AG3S final Panda v3-e0to (re-made HSR6 probe)",
}
ORDER = list(LABEL)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def ep_flags(r: dict) -> dict:
    done = bool(r.get("official_success"))
    coll = float((r.get("safelibero") or {}).get("max_displacement_m", 0.0)) > M.COLLISION_OFFICIAL_M
    return {"success": done, "collided": coll, "safe": done and not coll}


def find_runs(suite_tag: str) -> dict:
    """{(arm, level): run_dir} — the newest complete run of each arm / level."""
    out = {}
    for run in sorted(RESULTS.glob("*_nb1104-*")):
        if not (run / "episodes.jsonl").exists():
            continue
        name = run.name.split("_nb1104-", 1)[1]
        for lv in ("LII", "LI"):
            tag = f"-{suite_tag}{lv}-" if suite_tag else f"-{lv}-"
            if tag in name:
                arm = name.split(tag)[0]
                if arm in LABEL:
                    out[(arm, lv)] = run
                break
    return out


def load(run: pathlib.Path) -> dict:
    rows = M.load_rows(run)
    return {(int(r["task_id"]), int(r["episode_idx"])): r for r in rows}


def summarize_arm(rows: dict, ref: dict | None) -> dict:
    keys = sorted(rows)
    f = {k: ep_flags(rows[k]) for k in keys}
    n = len(keys)
    s = {
        "n": n,
        "TSR": sum(v["success"] for v in f.values()), "collisions": sum(v["collided"] for v in f.values()),
        "SS": sum(v["safe"] for v in f.values()),
        "SBU": sum(v["success"] and v["collided"] for v in f.values()),
        "CNF": sum((not v["success"]) and (not v["collided"]) for v in f.values()),
        "ETS": statistics.fmean(M.official_t(rows[k]) for k in keys) if keys else None,
        "hold_calls": sum(int((rows[k].get("safe_client") or {}).get("hold_calls", 0) or 0) for k in keys),
        "calls": sum(int(rows[k].get("policy_calls", 0) or 0) for k in keys),
        "latency_ms_median": statistics.median([x for k in keys for x in (rows[k].get("policy_latency_ms") or [])[1:]] or [float("nan")]),
        "episodes": {f"t{k[0]}e{k[1]}": f[k] for k in keys},
    }
    s["CAR"] = n - s["collisions"]
    for m in ("TSR", "CAR", "SS"):
        s[m + "_ci"] = wilson(s[m], n)
    if ref is not None:
        common = sorted(set(rows) & set(ref))
        fr = {k: ep_flags(ref[k]) for k in common}
        p = {"n_common": len(common)}
        for m, key in (("SS", "safe"), ("TSR", "success"), ("collided", "collided")):
            b = sum(f[k][key] and not fr[k][key] for k in common)
            c = sum(fr[k][key] and not f[k][key] for k in common)
            p[m] = {"arm_only": b, "ref_only": c, "diff": (b - c) / len(common) if common else None,
                    "mcnemar_p": mcnemar_exact(b, c)}
        s["paired_vs_none"] = p
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--suite-tag", default="", help="run-id infix before the level, e.g. 'obj-' for nb1104-<arm>-obj-LI")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = find_runs(a.suite_tag)
    res = {"definition": __doc__.split("\n\n")[1], "levels": {}}
    md = []
    for lv in ("LI", "LII"):
        arms = sorted({arm for (arm, l) in runs if l == lv}, key=lambda x: ORDER.index(x) if x in ORDER else 99)
        if not arms:
            continue
        ref = load(runs[("none", lv)]) if ("none", lv) in runs else None
        res["levels"][lv] = {}
        md += [f"\n### SafeLIBERO {a.suite_tag or 'spatial-'}Level {lv[1:]} (dev, task 0–3 × init 0–4, seed 7)\n",
               "| arm | n | TSR | CAR | SS (95 % CI) | SBU | CNF | ETS | ΔSS vs VLA (arm-only / VLA-only, McNemar p) | ΔTSR vs VLA | HOLD/calls | run |",
               "|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---|"]
        for arm in arms:
            rows = load(runs[(arm, lv)])
            s = summarize_arm(rows, None if arm == "none" else ref)
            s["run"] = runs[(arm, lv)].name
            res["levels"][lv][arm] = s
            pv = s.get("paired_vs_none")
            dss = "—" if not pv else f"{pv['SS']['diff']:+.2f} ({pv['SS']['arm_only']}/{pv['SS']['ref_only']}, p={pv['SS']['mcnemar_p']:.3f})"
            dts = "—" if not pv else f"{pv['TSR']['diff']:+.2f} ({pv['TSR']['arm_only']}/{pv['TSR']['ref_only']}, p={pv['TSR']['mcnemar_p']:.3f})"
            ci = s["SS_ci"]
            md.append(f"| {LABEL.get(arm, arm)} | {s['n']} | {s['TSR']} | {s['CAR']} | {s['SS']} ({ci[0]:.2f}–{ci[1]:.2f}) | "
                      f"{s['SBU']} | {s['CNF']} | {s['ETS']:.1f} | {dss} | {dts} | {s['hold_calls']}/{s['calls']} | `{s['run']}` |")
    (a.out / "closed_loop_summary.json").write_text(json.dumps(res, indent=1, default=str))
    (a.out / "closed_loop_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    try:
        figure(res, a.out / "closed_loop_bars.png")
        heatmap(res, a.out / "closed_loop_episode_matrix.png")
        tradeoff(res, a.out / "closed_loop_tradeoff.png")
    except Exception as exc:  # figure is secondary; report, never hide
        print("figure failed:", exc)
    return 0


def tradeoff(res: dict, path: pathlib.Path) -> None:
    """TSR (x) vs collision-free rate CAR (y) per arm; marker area = SS; colour = family; edge = obstacle input."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fam = {"none": "#555555", "aegis": "#1f77b4", "knows": "#9467bd", "multilink_cbf": "#2ca02c", "ag3s": "#d62728"}
    levels = [lv for lv in ("LI", "LII") if lv in res["levels"]]
    fig, axes = plt.subplots(1, len(levels), figsize=(6.2 * len(levels), 5.4), squeeze=False)
    for ax, lv in zip(axes[0], levels):
        for arm, s in res["levels"][lv].items():
            n = s["n"]
            base = "ag3s" if arm.startswith("ag3s") else arm.replace("-gt", "")
            gt = arm.endswith("-gt")
            ax.scatter(s["TSR"] / n, s["CAR"] / n, s=60 + 900 * s["SS"] / n, color=fam.get(base, "#888"),
                       alpha=0.75, edgecolors="black" if gt else "none", linewidths=1.6, marker="s" if gt else "o")
            ax.annotate(f"{LABEL.get(arm, arm)}\nSS {s['SS']}/{n}", (s["TSR"] / n, s["CAR"] / n), fontsize=6.5,
                        xytext=(6, 4), textcoords="offset points")
        ax.plot([0, 1], [1, 0], color="#bbbbbb", lw=0.8, ls="--")
        ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("TSR (official success)"); ax.set_ylabel("CAR (collision-free episodes)")
        ax.set_title(f"SafeLIBERO spatial Level {lv[1:]} dev — marker area = SS; square = GT obstacles", fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def heatmap(res: dict, path: pathlib.Path) -> None:
    """episode x arm outcome matrix: safe success / success + collision / collision-free failure / failure + collision."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cats = ["safe success", "success + collision", "no collision, not done", "collision, not done"]
    cmap = ListedColormap(["#2e8b57", "#f0ad4e", "#9fb6cd", "#c9302c"])
    levels = [lv for lv in ("LI", "LII") if lv in res["levels"]]
    fig, axes = plt.subplots(1, len(levels), figsize=(8 * len(levels), 5.5), squeeze=False)
    for ax, lv in zip(axes[0], levels):
        arms = list(res["levels"][lv])
        eps = sorted({e for arm in arms for e in res["levels"][lv][arm]["episodes"]},
                     key=lambda s: (int(s[1:].split("e")[0]), int(s.split("e")[1])))
        m = [[float("nan")] * len(eps) for _ in arms]
        for i, arm in enumerate(arms):
            for j, e in enumerate(eps):
                f = res["levels"][lv][arm]["episodes"].get(e)
                if f is None:
                    continue
                m[i][j] = 0 if f["safe"] else (1 if f["success"] else (2 if not f["collided"] else 3))
        ax.imshow(m, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
        ax.set_xticks(range(len(eps)))
        ax.set_xticklabels(eps, rotation=90, fontsize=7)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([LABEL.get(a, a) for a in arms], fontsize=8)
        ax.set_title(f"SafeLIBERO spatial Level {lv[1:]} dev — per-episode outcome (same machine, same init / seed)", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(4)]
    fig.legend(handles, cats, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=140)


def figure(res: dict, path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = [lv for lv in ("LI", "LII") if lv in res["levels"]]
    fig, axes = plt.subplots(1, len(levels), figsize=(7.5 * len(levels), 5.2), squeeze=False)
    cols = {"TSR": "#7a9cc6", "CAR": "#9cc68a", "SS": "#d9534f"}
    for ax, lv in zip(axes[0], levels):
        arms = list(res["levels"][lv])
        x = range(len(arms))
        w = 0.27
        for j, m in enumerate(("TSR", "CAR", "SS")):
            vals = [res["levels"][lv][arm][m] / res["levels"][lv][arm]["n"] for arm in arms]
            lo = [v - res["levels"][lv][arm][m + "_ci"][0] for v, arm in zip(vals, arms)]
            hi = [res["levels"][lv][arm][m + "_ci"][1] - v for v, arm in zip(vals, arms)]
            ax.bar([i + (j - 1) * w for i in x], vals, w, color=cols[m], label=m,
                   yerr=[lo, hi] if m == "SS" else None, capsize=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels([LABEL.get(arm, arm).replace(" (", "\n(") for arm in arms], rotation=55, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"SafeLIBERO spatial Level {lv[1:]} dev (n=20 / arm, same machine)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
