#!/usr/bin/env python3
"""nb1104 (C11) — VB2-style check figure of the re-made HSR6-Panda labels: for the 4 VB2 episodes of SL6-B, the physical
phase label per policy request (band) and the agent-view frame at the first request of each phase.

usage: .venv-ag3s/bin/python experiments/segmentation/nb1104_probe_bands_figure.py LABELS_RUN DEV_RUN OUT.png
"""
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"approach": "#9e9e9e", "grasp": "#e6a23c", "lift": "#5b8ff9", "transport": "#61c2a2", "placement": "#d9534f"}


def main() -> int:
    lab, dev, out = map(pathlib.Path, sys.argv[1:4])
    bands = json.loads((lab / "labels/vb2_bands.json").read_text())
    stems = list(bands)
    fig = plt.figure(figsize=(14, 3.1 * len(stems)))
    gs = fig.add_gridspec(len(stems) * 2, 5, height_ratios=[0.25, 1] * len(stems))
    for i, stem in enumerate(stems):
        reqs = bands[stem]["requests"]
        ax = fig.add_subplot(gs[2 * i, :])
        for r in reqs:
            ax.barh(0, 1, left=r["k"], color=COL[r["label"]], edgecolor="none")
        ax.set_xlim(0, len(reqs)); ax.set_yticks([]); ax.set_xlabel("policy request", fontsize=7)
        ax.set_title(f"{stem}  (re-made labels, same sl6_phase_labels rule)", fontsize=8, loc="left")
        imgs = np.load(dev / "episodes" / f"{stem}.npz")["obs_agentview_image"]
        firsts = {}
        for r in reqs:
            firsts.setdefault(r["label"], r)
        for j, ph in enumerate(COL):
            axi = fig.add_subplot(gs[2 * i + 1, j]); axi.axis("off")
            if ph in firsts:
                f = min(int(firsts[ph]["frame"]), len(imgs) - 1)
                axi.imshow(imgs[f][::-1, ::-1])
                axi.set_title(f"first {ph} (request {firsts[ph]['k']}, frame {f})", fontsize=7, color=COL[ph])
            else:
                axi.set_title(f"no {ph}", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()]
    fig.legend(handles, list(COL), loc="upper right", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
