"""T61 held-out — add the held-out numbers to handoff/T61.verify.json and draw the paired-difference figure.

usage: t61_heldout_verify.py B_RUN_DIR FIG_DIR      (.venv-ag3s)
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

OUT = ROOT / "AG3S/ag3s/docs/handoff/T61.verify.json"
RUN61 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261014T183529+0900_53f3911df197_dd2fad8756a6")


def main(run: pathlib.Path, fig_dir: pathlib.Path) -> None:
    d = json.loads((run / "verify/T61_heldout/heldout.json").read_text())
    cl = json.loads((run / "verify/T61_heldout/closed_loop_serving_record.json").read_text())["sets"]["T61"]["arms"]["C1a"]
    agg = json.loads((RUN61 / "verify/T61/agg.json").read_text())["arms"]["C1a"]
    status = (run / "verify/T61_heldout/status.txt").read_text().splitlines()
    windows = {}
    for w, sv in d["windows"].items():
        windows[w] = {s: {k: v for k, v in row.items() if k != "per_episode"} | {
            "per_episode_gt_identity": {k: e["gt_identity"] for k, e in row["per_episode"].items()}}
            for s, row in sv.items()}
    v = json.loads(OUT.read_text())
    v["status"] = "held-out evaluated (selection frozen b49f5f6c before opening)"
    v["heldout"] = {
        "run_dir": str(run), "code": "git archive 28539681 (serving md5 = T56 run2 = calibration code)",
        "replay_status": [ln for ln in status if not ln.startswith("wait mem")],
        "numbers": {
            "selection": d["selection"], "selection_sha256": d["selection_sha256"], "sources": d["sources"],
            "bootstrap": d["bootstrap"], "endpoints": d["endpoints"],
            "endpoints_reached": {"P": sum(e["t_P"] is not None for e in d["endpoints"].values()),
                                  "H": sum(e["t_H"] is not None for e in d["endpoints"].values())},
            "windows": windows,
            "serving_record_C1a": {"agg_t1_success": agg["t1_success"],
                                   "hold_chunks_total": agg["t4_hold"]["hold_chunks_total"],
                                   "windows": {w: {k: cl[w].get(k) for k in ("success", "success_ct", "hold_chunks",
                                                                               "hold_runs", "completion_s_median")}
                                               for w in ("full", "P", "H")}},
        },
    }
    OUT.write_text(json.dumps(v, indent=1, ensure_ascii=False, default=float) + "\n")
    print("->", OUT)

    # figure: per-episode paired difference (selection − L8H2) by window + summary table
    sel, base = d["sources"][0], d["baseline"]
    keys = d["keys"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    nums = {"selection": sel, "baseline": base, "per_episode_diff_pp": {}, "summary": {}}
    for i, w in enumerate(("full", "P", "H")):
        pe_s = d["windows"][w][sel]["per_episode"]
        pe_b = d["windows"][w][base]["per_episode"]
        diff = [100 * (pe_s[k]["gt_identity"] - pe_b[k]["gt_identity"]) for k in keys]
        x = np.arange(len(keys)) + (i - 1) * 0.27
        ax.bar(x, diff, width=0.27, color=figstyle.SEQ_BLUE[[2, 5, 9][i] if len(figstyle.SEQ_BLUE) > 9 else [1, 3, 5][i]],
               label=f"{w} window")
        nums["per_episode_diff_pp"][w] = dict(zip(keys, diff))
    ax.axhline(0, color=figstyle.INK, lw=0.8)
    ax.set_xticks(range(len(keys)), keys, rotation=60, fontsize=7)
    ax.set_ylabel("GT identity, selection − L8H2 (%p)")
    ax.set_title(f"T61 held-out (16 new episodes): {sel} vs {base}, per episode")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.axis("off")
    rows, cells = [], []
    for w in ("full", "P", "H"):
        p = d["windows"][w][sel]["paired_vs_L8H2"]
        cells.append([f"{100 * d['windows'][w][sel]['episode_mean_gt_identity']:.1f}",
                      f"{100 * d['windows'][w][base]['episode_mean_gt_identity']:.1f}",
                      f"{100 * p['mean_diff']:+.2f}",
                      f"[{100 * p['ci95_episode_bootstrap'][0]:+.1f}, {100 * p['ci95_episode_bootstrap'][1]:+.1f}]",
                      f"{p['n_up']}/{p['n_down']}"])
        rows.append(f"{w}" + (" (primary)" if w == "P" else ""))
        nums["summary"][w] = cells[-1]
    t = ax.table(cellText=cells, rowLabels=rows, colLabels=["sel %", "L8H2 %", "diff %p", "episode 95% CI", "up/down"],
                 loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.6)
    ax.set_title("episode-mean GT identity_correct (B 10000, seed 6102)", fontsize=9)
    figstyle.style_axes(fig, [axes[0]])
    fig.tight_layout()
    p = fig_dir / "t61/t61-heldout-paired.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise SystemExit(f"{p} exists")
    fig.savefig(p, dpi=150, facecolor=figstyle.SURFACE)
    p.with_suffix(".json").write_text(json.dumps(nums, indent=1, default=float))
    print("->", p)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
