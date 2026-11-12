#!/usr/bin/env python3
"""(2026-11-12 sdm-f6) Rule-A figure for plan §9.28: SL-SOTA16 -dc (--esdf-dead-camera-seen-ok) on SDM, rejected.
(a) real scene, last video frame, pkgFrs vs -dc: Spatial L-II task 1 init 18 (HOLD to the end -> released, obstacle pushed further)
    and Goal L-I task 2 init 17 (HOLD to the end -> released, still not placed);
(b) HOLD calls per episode that had a HOLD under pkgFrs, with the episode outcome of both versions;
(c) stage-1 totals on the 56 episodes (6 cell-runs) against the pass line fixed before the run (plan §9.26).
Input = the JSON written by dc_cmp.py (per-episode comparison of the two tags).
usage: OMP_NUM_THREADS=1 .venv-ag3s/bin/python experiments/segmentation/sota_offline_20261111/fig_dc_sdm.py DC_CMP.json OUT.png"""
import glob, json, sys
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.experiments.common import figstyle

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
src, out = sys.argv[1], sys.argv[2]
D = json.load(open(src))
TAGS = [("pkgFrs", "pkgFrssdm"), ("-dc", "pkgFdcsdm")]
CELL = {"goal-LI-i17_18": "Goal L-I", "spatial-LII-i17_18": "Spatial L-II", "long-LII-i17_18": "Long L-II", "spatial-LI-i15_16_17_18": "Spatial L-I",
        "spatial-LII-i15_16": "Spatial L-II", "long-LI-i15_16": "Long L-I", "e1a-spatial-LI-t3": "E1a"}


def frame(tag, cell, t, init):
    suite, lvl, _ = cell.split("-")
    d = sorted(glob.glob(B + f"2*_ms-{tag}-{suite}-{lvl}-i*"))
    f = [x for dd in d for x in glob.glob(dd + f"/videos/*t{t}_e{init}_*.mp4")][-1]
    rd = imageio.get_reader(f)
    n = rd.count_frames()
    return np.asarray(rd.get_data(n - 1)), n


ko = figstyle.use_korean()
fig = plt.figure(figsize=(17, 9.8))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.32, wspace=0.14)
axes, meta = [], []
shots = [("spatial-LII-i17_18", 1, 18), ("goal-LI-i17_18", 2, 17)]
k = 0
for cell, t, i in shots:
    for name, tag in TAGS:
        ax = fig.add_subplot(gs[0, k]); axes.append(ax)
        e = D[cell]["pkgFrs" if name == "pkgFrs" else "dc"][f"t{t}e{i}"]
        try:
            img, n = frame(tag, cell, t, i)
            ax.imshow(img)
        except Exception as exc:  # the video is not there: say so rather than draw nothing
            n = 0
            ax.text(0.5, 0.5, f"no video\n{type(exc).__name__}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"(a{k + 1}) {CELL[cell]} t{t} e{i}, {name}\n{e['label']}, obstacle {e['d']:.0f} mm, HOLD {e['hold']} of {e['n']} calls", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        meta.append({"tag": tag, "cell": cell, "task": t, "init": i, "frames": n, "result": e["label"], "displacement_mm": e["d"], "hold_calls": e["hold"]})
        k += 1

rows = []
for cell in CELL:
    for ep, a in D[cell]["pkgFrs"].items():
        b = D[cell]["dc"][ep]
        if a["hold"] or b["hold"]:
            mm = "" if (b["d"] < 1 and a["d"] < 1) or abs(b["d"] - a["d"]) < 0.05 else f"\n{a['d']:.0f}→{b['d']:.0f}"
            rows.append((f"{CELL[cell]}\n{ep.replace('e', ' e')}\n{a['label']} → {b['label']}{mm}", a, b))
axh = fig.add_subplot(gs[1, 0:3]); axes.append(axh)
x = np.arange(len(rows)); w = 0.38
axh.bar(x - w / 2, [a["hold"] for _, a, _ in rows], w, color=figstyle.CATEGORICAL[0], label="pkgFrs (frozen)")
axh.bar(x + w / 2, [b["hold"] for _, _, b in rows], w, color=figstyle.CATEGORICAL[1], label="-dc (--esdf-dead-camera-seen-ok)")
top = max(a["hold"] for _, a, _ in rows)
for j, (_, a, b) in enumerate(rows):
    axh.text(j - w / 2, a["hold"] + 0.8, str(a["hold"]), ha="center", va="bottom", fontsize=8)
    axh.text(j + w / 2, b["hold"] + 0.8, str(b["hold"]), ha="center", va="bottom", fontsize=8)
axh.set_xticks(x); axh.set_xticklabels([r[0] for r in rows], fontsize=7.5)
axh.set_ylim(0, top * 1.28); axh.set_ylabel("HOLD calls in the episode")
axh.legend(fontsize=8.5, loc="upper left")
axh.set_title("(b) episodes with a HOLD under pkgFrs: HOLD calls; tick = outcome pkgFrs → -dc (F fail, F+C fail + collision), "
              "obstacle mm where changed", fontsize=9.5, loc="left")

axc = fig.add_subplot(gs[1, 3]); axes.append(axc)
ta, tb = D["stage1_total"]["pkgFrs"], D["stage1_total"]["dc"]
labs = ["TSR", "CAR", "SSR", "> 50 mm"]
va, vb = [ta[1], ta[2], ta[3], ta[4]], [tb[1], tb[2], tb[3], tb[4]]
xs = np.arange(4)
axc.bar(xs - w / 2, va, w, color=figstyle.CATEGORICAL[0], label="pkgFrs")
axc.bar(xs + w / 2, vb, w, color=figstyle.CATEGORICAL[1], label="-dc")
for j in range(4):
    axc.text(j - w / 2, va[j] + 0.6, str(va[j]), ha="center", fontsize=8.5)
    axc.text(j + w / 2, vb[j] + 0.6, str(vb[j]), ha="center", fontsize=8.5)
axc.plot([2 - 0.48, 2 + 0.48], [22, 22], color=figstyle.INK, lw=1.3, ls="--", label="pass line: SSR >= 22, > 50 mm <= 6")
axc.plot([3 - 0.48, 3 + 0.48], [ta[4] + 1] * 2, color=figstyle.INK, lw=1.3, ls="--")
axc.set_xticks(xs); axc.set_xticklabels(labs)
axc.set_ylim(0, 64); axc.set_ylabel("episodes / 56")
axc.legend(fontsize=8, loc="upper right")
axc.set_title("(c) stage 1: 6 cell-runs, 56 episodes", fontsize=10)
figstyle.style_axes(fig, axes)
e1a_a, e1a_b = D["e1a-spatial-LI-t3"]["cnt_pkgFrs"], D["e1a-spatial-LI-t3"]["cnt_dc"]
fig.suptitle(f"SL-SOTA16 -dc on SDM: dead-camera HOLD calls {ta[6]} → {tb[6]} (all HOLD {ta[5]} → {tb[5]}), but SSR {ta[3]} → {tb[3]} and "
             f"> 50 mm collisions {ta[4]} → {tb[4]} on the 56 episodes — rejected, pkgFrs stays frozen", fontsize=12, y=0.975)
fig.savefig(out, dpi=140, bbox_inches="tight")
json.dump({"figure": out, "source": src,
           "set": "stage 1 = 6 cell-runs with a HOLD episode under pkgFrs (Goal L-I init 17,18; Spatial L-I init 15-18; Spatial L-II init 15,16 and 17,18; "
                  "Long L-I init 15,16; Long L-II init 17,18), task 0-3, seed 7; stage 2 = E1a task 3 init 5-14. In-sample: the candidate was found on these episodes.",
           "columns": D["stage1_total"]["cols"], "stage1_pkgFrs": ta, "stage1_dc": tb,
           "pass_line_fixed_before_run": {"dead_camera_hold": 0, "ssr_min": 22, "over_50mm_max": ta[4] + 1},
           "verdict": "rejected (SSR 20 < 22, > 50 mm +2)",
           "stage2_e1a_task3": {"pkgFrs": e1a_a, "dc": e1a_b},
           "per_cell": {c: {"pkgFrs": D[c]["cnt_pkgFrs"], "dc": D[c]["cnt_dc"]} for c in CELL},
           "hold_episodes": [{"episode": " ".join(r[0].split("\n")[:2]), "pkgFrs": {k2: r[1][k2] for k2 in ("label", "d", "n", "hold", "tail")},
                              "dc": {k2: r[2][k2] for k2 in ("label", "d", "n", "hold", "tail", "note")}} for r in rows],
           "note": "'note' counts calls whose record carries esdf_camera_outside_workspace; it is a lower bound (the AG3S block is not written on calls with status ok).",
           "scene_frames": meta, "korean_font": bool(ko)}, open(out[:-4] + ".json", "w"), indent=1, ensure_ascii=False)
print(ta, tb, len(rows))
