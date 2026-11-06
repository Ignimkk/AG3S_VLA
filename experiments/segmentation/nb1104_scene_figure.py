#!/usr/bin/env python3
"""nb1104 — real-scene grid: rows = arms, columns = chosen episodes; each cell = agent-view frame at the first official
collision (``safelibero.collide_t``) or, without collision, the last frame. Title = success / collision of that arm.

usage: .venv-ag3s/bin/python experiments/segmentation/nb1104_scene_figure.py --out FIG.png [--level LI] [--episodes t0e0,t1e0,...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/segmentation"))
import nb1104_aggregate as A  # noqa: E402


def frame(run: pathlib.Path, row: dict) -> tuple[np.ndarray, str]:
    npz = np.load(run / row["npz"], allow_pickle=False)
    imgs = npz["obs_agentview_image"]
    sl = row.get("safelibero") or {}
    coll = float(sl.get("max_displacement_m", 0.0)) > A.M.COLLISION_OFFICIAL_M
    t = int(sl.get("collide_t", len(imgs) - 1)) if coll else len(imgs) - 1
    t = max(0, min(t, len(imgs) - 1))
    tag = ("success" if row.get("official_success") else "fail") + (" + COLLISION" if coll else ", no collision")
    return imgs[t][::-1, ::-1], f"{tag} (t={t})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--level", default="LI")
    ap.add_argument("--episodes", default="t0e0,t1e0,t2e0,t3e0")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = {arm: r for (arm, lv), r in A.find_runs("").items() if lv == a.level}
    arms = [x for x in A.ORDER if x in runs]
    eps = [tuple(int(v) for v in e[1:].split("e")) for e in a.episodes.split(",")]
    fig, axes = plt.subplots(len(arms), len(eps), figsize=(2.6 * len(eps), 2.75 * len(arms)), squeeze=False)
    for i, arm in enumerate(arms):
        rows = A.load(runs[arm])
        for j, k in enumerate(eps):
            ax = axes[i][j]
            ax.axis("off")
            if k not in rows:
                ax.set_title("(not run)", fontsize=7)
                continue
            img, tag = frame(runs[arm], rows[k])
            ax.imshow(img)
            col = "#2a7d2a" if "success, no" in tag else ("#b22222" if "COLLISION" in tag else "#555555")
            ax.set_title(tag, fontsize=7, color=col)
            if j == 0:
                ax.text(-0.06, 0.5, A.LABEL.get(arm, arm).replace(" (", "\n("), transform=ax.transAxes, fontsize=7,
                        ha="right", va="center")
    for j, k in enumerate(eps):
        axes[0][j].text(0.5, 1.22, f"task {k[0]} · init {k[1]}", transform=axes[0][j].transAxes, ha="center", fontsize=8,
                        weight="bold")
    fig.suptitle(f"SafeLIBERO spatial Level {a.level[1:]} dev — frame at first official collision (else last frame)",
                 fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(left=0.2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
