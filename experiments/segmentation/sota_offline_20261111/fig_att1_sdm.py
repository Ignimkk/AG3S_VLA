#!/usr/bin/env python3
"""(2026-11-13 sdm-f6, SL-ATT1, rule A) Figure: (a) per SafeLIBERO cell, the share of pre-grasp calls whose top-attention body is
the object to pick -- served (L8H2, head + wrist fused) vs head-only L8H2 vs head-only L12H3 (attn_cells_score.py output);
(b) wrist / head peak ratio per call for L8H2 and L12H3 (replayed pooled attention), log scale.
usage: .venv-ag3s/bin/python fig_att1_sdm.py SCORE.json REPLAY_DIR OUT.png  (writes OUT.json sidecar)"""
import glob, json, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = 8; C = {"L8H2": 8 * H + 2, "L12H3": 12 * H + 3}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    score, rep, out = sys.argv[1:4]
    d = json.load(open(score))
    cells = ["spatial-LI", "spatial-LII", "object-LI", "object-LII", "goal-LI", "goal-LII", "long-LI", "long-LII"]
    cnt = {k: collections.Counter() for k in cells}
    for e in d:
        k = f"{e['suite']}-{e['level']}"
        for c in e["calls"]:
            g0 = e["goal"][0]; cnt[k]["n"] += 1
            cnt[k]["served"] += c["a_recorded"] == g0
            for name, i in C.items():
                cnt[k][name] += c["tops"][i] == g0
    tot = collections.Counter()
    for k in cells:
        tot.update(cnt[k])
    rows = {"served (L8H2, head+wrist max)": "served", "L8H2 head only": "L8H2", "L12H3 head only": "L12H3"}
    ratio = {}
    for name, (l, h) in {"L8H2": (8, 2), "L12H3": (12, 3)}.items():
        r = []
        for f in sorted(glob.glob(rep + "/*/*.npz")):
            z = np.load(f)
            if "pooled_f16" not in z.files:
                continue
            p = z["pooled_f16"][:, l, h].astype(np.float64)
            r += list(p[:, 256:512].max(1) / np.maximum(p[:, :256].max(1), 1e-12))
        ratio[name] = np.array(r)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    labels = cells + ["all"]; x = np.arange(len(labels)); w = 0.26
    for j, (lab, key) in enumerate(rows.items()):
        v = [cnt[k][key] / max(cnt[k]["n"], 1) for k in cells] + [tot[key] / tot["n"]]
        a.bar(x + (j - 1) * w, v, w - 0.02, color=[BLUE, ORANGE, AQUA][j], label=lab, zorder=2)
        a.text(x[-1] + (j - 1) * w, v[-1] + 0.015, f"{v[-1]:.2f}", ha="center", fontsize=8, color=INK)
    a.set_xticks(x); a.set_xticklabels([f"{l}\n(n={cnt[l]['n'] if l in cnt else tot['n']})" for l in labels], fontsize=8)
    a.set_ylim(0, 1.05); a.set_ylabel("share of pre-grasp calls: top-attention body = object to pick")
    a.grid(axis="y", color=GRID, zorder=0); a.legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
    a.set_title("(a) does the attention point at the object to pick? (offline, frozen-version calls)", fontsize=9,
                color=INK, loc="left")
    bins = np.logspace(-3, 6, 46)
    for name, col in (("L8H2", BLUE), ("L12H3", AQUA)):
        rr = np.clip(ratio[name], 1e-3, 1e6)
        b.hist(rr, bins=bins, histtype="step", lw=2, color=col,
               label=f"{name}: median {np.median(ratio[name]):.3g}, wrist>head {np.mean(ratio[name] > 1):.0%}")
    b.axvline(1, color=INK2, lw=1, ls="--"); b.set_xscale("log")
    b.set_xlabel("wrist peak / head peak attention (per call)"); b.set_ylabel("calls")
    b.legend(frameon=False, fontsize=8, loc="upper left"); b.grid(color=GRID)
    b.set_title("(b) which camera dominates the fused (max) cloud", fontsize=9, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    side = {"figure": out, "made_by": "experiments/segmentation/sota_offline_20261111/fig_att1_sdm.py",
            "inputs": {"score": score, "replay": rep}, "kind": "measured offline (replayed attention on recorded calls)",
            "panel_a": {k: dict(cnt[k]) for k in cells} | {"all": dict(tot)},
            "panel_b": {n: {"calls": int(len(r)), "median": float(np.median(r)), "wrist_gt_head": float(np.mean(r > 1))}
                        for n, r in ratio.items()},
            "caveat": "top-attention body is not the server's final target; the closed-loop effect is measured by "
                      "sota_att1_queue.sh"}
    json.dump(side, open(out[:-4] + ".json", "w"), indent=1)
    print(json.dumps(side["panel_b"]), {k: round(tot[v] / tot["n"], 3) for k, v in rows.items()})


if __name__ == "__main__":
    main()
