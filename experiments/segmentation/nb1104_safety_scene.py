#!/usr/bin/env python3
"""nb1104 — LIBERO-Safety L0 VLA-only scene grid: rows = suite, columns = (task, init 0..1); frame at the first official
violation, else the last frame. usage: .venv-ag3s/bin/python experiments/segmentation/nb1104_safety_scene.py OUT.png"""
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
runs = sorted(R.glob("*_nb1104-safety-vla-*"))
cols = [(0, 0), (0, 1), (1, 0), (1, 1)]
fig, axes = plt.subplots(len(runs), len(cols), figsize=(2.8 * len(cols), 3.1 * len(runs)), squeeze=False)
for i, run in enumerate(runs):
    rows = {(int(r["task_id"]), int(r["episode_idx"])): r for r in map(json.loads, (run / "episodes.jsonl").read_text().splitlines())}
    for j, k in enumerate(cols):
        ax = axes[i][j]; ax.axis("off")
        r = rows.get(k)
        if r is None:
            continue
        imgs = np.load(run / r["npz"])["obs_agentview_image"]
        s = r["libero_safety"]; v = (s.get("official_first_violation") or [None])[0]
        t = min(int(v), len(imgs) - 1) if v is not None else len(imgs) - 1
        ax.imshow(imgs[t][::-1, ::-1])
        tag = ("success" if r["official_success"] else "fail") + (" + VIOLATION" if s["official_violation"] else "")
        ax.set_title(f"{r['suite']} t{k[0]} i{k[1]}: {tag} (t={t})\n{r['task_description'][:48]}", fontsize=6.5,
                     color="#2a7d2a" if r["official_success"] and not s["official_violation"] else "#b22222")
fig.suptitle("LIBERO-Safety Level 0, π0.5-LIBERO VLA only (same machine) — frame at first violation, else last frame", fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=120); print(sys.argv[1])
