#!/usr/bin/env python3
"""(2026-11-13 sdm-f6, rule A) Paper Tab. 1, this server's 1/10: per-cell SR / SSR / CAR of every arm measured so far
(paper_tab1_tenth_queue.sh runs, run-id tab1sdm-<arm>-{sl,lb}-<suite>[-L<lvl>]). SafeLIBERO: SR = official success,
SSR = success without moving the obstacle (safelibero.safe_success), CAR = 1 - collided. LIBERO: SR only.
The latest run per run-id is used; a run whose episodes.jsonl is short of the cell size is reported as partial.
usage: .venv-ag3s/bin/python fig_paper_tab1_tenth.py OUT.png [RESULTS_DIR]  (writes OUT.json sidecar)"""
import glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SL = [f"{s}-L{l}" for l in ("I", "II") for s in ("spatial", "object", "goal", "long")]
LB = ["spatial", "object", "goal", "10"]


def load(res):
    runs = {}
    for d in sorted(glob.glob(os.path.join(res, "*_tab1sdm-*"))):
        m = re.search(r"_tab1sdm-(\w+?)-(sl|lb)-(.+?)(?:-L(?:I|II))?-(?:vla_only|vla_v11ab)$", d)
        if not m or not os.path.exists(d + "/episodes.jsonl"):
            continue
        runs[(m.group(1), m.group(2), m.group(3))] = d   # sorted by timestamp: the latest wins
    rows = []
    for (arm, part, cell), d in runs.items():
        rs = [json.loads(l) for l in open(d + "/episodes.jsonl")]
        n = len(rs)
        row = {"arm": arm, "part": part, "cell": cell, "run": os.path.basename(d), "n": n,
               "sr": sum(r["official_success"] for r in rs) / n, "errors": sum(r["error"] is not None for r in rs),
               "partial": n < (20 if part == "sl" else 50)}
        if part == "sl":
            row["ssr"] = sum(r["safelibero"]["safe_success"] for r in rs) / n
            row["car"] = 1 - sum(r["safelibero"]["collided"] for r in rs) / n
        rows.append(row)
    return rows


def main():
    out = sys.argv[1]
    res = sys.argv[2] if len(sys.argv) > 2 else "/mnt/dev/AG3S_VLA-storage/results/benchmark"
    rows = load(res)
    arms = [a for a in ("pi05", "ag3s", "gto") if any(r["arm"] == a for r in rows)]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2})
    fig, axes = plt.subplots(len(arms), 2, figsize=(13, 3.6 * len(arms)), squeeze=False,
                             gridspec_kw={"width_ratios": [2.4, 1]})
    summary = {}
    for i, arm in enumerate(arms):
        a, b = axes[i]
        sl = {r["cell"]: r for r in rows if r["arm"] == arm and r["part"] == "sl"}
        lb = {r["cell"]: r for r in rows if r["arm"] == arm and r["part"] == "lb"}
        cells = [c for c in SL if c in sl]
        x = np.arange(len(cells) + 1); w = 0.26
        n_all = sum(sl[c]["n"] for c in cells)
        agg = {k: (sum(sl[c][k] * sl[c]["n"] for c in cells) / n_all if n_all else float("nan")) for k in ("sr", "ssr", "car")}
        for j, (k, lab, col) in enumerate((("sr", "SR", BLUE), ("ssr", "SSR", ORANGE), ("car", "CAR", AQUA))):
            v = [sl[c][k] for c in cells] + [agg[k]]
            a.bar(x + (j - 1) * w, v, w - 0.02, color=col, label=lab, zorder=2)
        a.text(0.0, 1.0, f"all {n_all} ep: SR {agg['sr']:.3f} · SSR {agg['ssr']:.3f} · CAR {agg['car']:.3f}",
               transform=a.transAxes, fontsize=8.5, color=INK, va="top")
        a.set_xticks(x); a.set_xticklabels([f"{c}\n(n={sl[c]['n']}{'*' if sl[c]['partial'] else ''})" for c in cells]
                                           + [f"all\n(n={n_all})"], fontsize=8)
        a.set_ylim(0, 1.08); a.grid(axis="y", color=GRID, zorder=0); a.legend(frameon=False, ncol=3, loc="upper right")
        a.set_title(f"({chr(97 + 2 * i)}) {arm}: SafeLIBERO, task 0-3 x 5 chosen inits per cell", loc="left",
                    color=INK, fontsize=9)
        lcells = [c for c in LB if c in lb]
        n_lb = sum(lb[c]["n"] for c in lcells)
        sr_lb = sum(lb[c]["sr"] * lb[c]["n"] for c in lcells) / n_lb if n_lb else float("nan")
        v = [lb[c]["sr"] for c in lcells] + ([sr_lb] if lcells else [])
        xl = np.arange(len(v))
        b.bar(xl, v, 0.6, color=BLUE, zorder=2)
        for xx, vv in zip(xl, v):
            b.text(xx, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=7.5, color=INK)
        b.set_xticks(xl); b.set_xticklabels([f"{c}\n(n={lb[c]['n']})" for c in lcells] + ([f"all\n(n={n_lb})"] if lcells else []),
                                            fontsize=8)
        b.set_ylim(0, 1.08); b.grid(axis="y", color=GRID, zorder=0)
        b.set_title(f"({chr(98 + 2 * i)}) {arm}: LIBERO SR" + ("" if lcells else " (not run for this arm)"), loc="left",
                    color=INK, fontsize=9)
        summary[arm] = {"safelibero": {"n": n_all, **agg}, "libero": {"n": n_lb, "sr": sr_lb}}
    fig.tight_layout(); fig.savefig(out, dpi=150)
    side = {"figure": out, "made_by": "experiments/segmentation/fig_paper_tab1_tenth.py", "results_dir": res,
            "kind": "measured closed loop (paper Tab. 1, SDM 1/10 split, seed 7)",
            "split": "doc/active/experiments/PAPER_TENTH_SPLIT_20261113.json",
            "metrics": "SR = official success; SSR = safelibero.safe_success; CAR = 1 - safelibero.collided",
            "summary": summary, "rows": rows}
    json.dump(side, open(out[:-4] + ".json", "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
