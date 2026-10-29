"""Figure 3 v2 — HSR panel: sub-task probe p_t (pick / place / home) and the physical grasp state over one recorded episode.

  cd /mnt/dev/AG3S_VLA-development; .venv-ag3s/bin/python paper/tools/plotting/fig3_v2_hsr_timeline.py

Same recorded run as the other Figure 3 assets (T53 C1a ep1842, "Put the pear in the basket."), read through
`paperbanana_render.Run`. The grasp band is the recorded grasp-latch state reduced to held / not held; the three curves are the
recorded `summary.subtask.p`. Directly labelled, no legend; one short label per curve.
"""
import json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paperbanana_render as pr  # noqa: E402

OUT = pr.REPO / "paper/figures/paperbanana/figure3/assets_v2"
COL = {"pick": "#2E75B6", "place": "#E8711A", "home": "#548235"}
HELD = "#7030A0"


def main():
    run = pr.Run()
    states, p = [], []
    for c in run.chunks:
        s = json.loads(str(np.load(c)["summary_json"]))
        states.append(s.get("grasp", {}).get("state"))
        sp = s.get("subtask", {}).get("p") or {}
        p.append([sp.get(k, np.nan) for k in COL])
    p = np.asarray(p, float); n = len(states); x = np.arange(n)
    held = np.array([st == "held" for st in states])
    fig = plt.figure(figsize=(3.6, 1.35), dpi=400)
    ax = fig.add_axes([0.02, 0.10, 0.96, 0.62]); band = fig.add_axes([0.02, 0.80, 0.96, 0.13], sharex=ax)
    band.axhspan(0, 1, color="#EDEDED", lw=0)
    for j in np.flatnonzero(held):
        band.axvspan(j - 0.5, j + 0.5, color=HELD, alpha=0.85, lw=0)
    hx = np.flatnonzero(held)
    band.text(hx.mean(), 0.5, "held", color="white", ha="center", va="center", fontsize=6.5, fontweight="bold")
    band.set_ylim(0, 1); band.set_axis_off()
    for k, (lab, c) in enumerate(COL.items()):
        ax.plot(x, p[:, k], color=c, lw=1.6, solid_capstyle="round")
        j = int(np.nanargmax(p[:, k] > 0.5)); j = {0: 4, 1: (j + hx.max()) // 2, 2: n - 9}[k]
        ax.text(j, 1.07, lab, color=c, fontsize=6.5, ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(-0.05, 1.25); ax.set_xlim(-0.5, n - 0.5)
    ax.set_yticks([]); ax.set_xticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#7F7F7F")
    p_png = OUT / "f3v2_hsr_timeline.png"
    fig.savefig(p_png, dpi=400, transparent=False, facecolor="white"); plt.close(fig)
    (OUT / "f3v2_hsr_timeline.json").write_text(json.dumps({
        "asset": str(p_png.relative_to(pr.REPO)), "run": "T53 C1a", "episode": pr.KEY, "source_run_dir": str(pr.RUN),
        "script": "paper/tools/plotting/fig3_v2_hsr_timeline.py", "n_chunks": n, "held_chunks": hx.tolist(),
        "grasp_states": states, "encoding": {"curves": "recorded summary.subtask.p (pick blue, place orange, home green)",
                                             "band": "grasp-latch state 'held' (purple) vs any other state (grey)", "x": "action chunk index"},
        "is_measured_result": False, "note": "One recorded episode, illustration; no axis values printed."}, indent=1) + "\n")
    print("wrote", p_png, "held", hx.min(), hx.max(), "of", n)


if __name__ == "__main__":
    main()
