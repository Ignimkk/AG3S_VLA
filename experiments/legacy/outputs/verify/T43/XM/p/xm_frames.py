"""T43 XM — V9 video contact sheets for the user's check (frame t of third_person_front.mp4 = post state of control row t, 15 fps).
usage: xm_frames.py   (reads raw/cross_summary.json v9_classes; writes frames/<run>.png and frames/index.json)
Frames per run: closing, attach, attach+8, max apple lift, drop-4, drop, drop+4, palm crossing-8, palm crossing (those that exist).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, subprocess, tempfile
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

X = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/XM'))
S = json.load(open(X / "raw/cross_summary.json"))
OUT = X / "frames"; OUT.mkdir(exist_ok=True)
index = {}
for key, v in S["v9_classes"].items():
    f = v["video"]["frames"]; mp4 = v["video"]["mp4"]
    want = []
    def add(lab, t):
        if t is not None and t >= 0 and t not in [w[1] for w in want]:
            want.append((lab, int(t)))
    add("closing", f.get("closing")); add("attach", f.get("attach"))
    if f.get("attach") is not None:
        add("attach+8", f["attach"] + 8)
    add("max lift", f.get("max_lift"))
    if f.get("drop") is not None:
        add("drop-4", f["drop"] - 4); add("drop", f["drop"]); add("drop+4", f["drop"] + 4)
    if f.get("palm_cross") is not None:
        add("palm cross-8", f["palm_cross"] - 8); add("palm cross", f["palm_cross"])
    want.sort(key=lambda w: w[1])
    with tempfile.TemporaryDirectory() as td:
        sel = "+".join(f"eq(n\\,{t})" for _, t in want)
        subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, "-vf", f"select='{sel}'", "-vsync", "0", f"{td}/f%02d.png"], check=True)
        imgs = sorted(pathlib.Path(td).glob("f*.png"))
        n = len(imgs); cols = min(5, n); rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows + 0.4))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, (lab, t), im in zip(axes, want, imgs):
            ax.imshow(mpimg.imread(im)); ax.set_title(f"{lab}: t={t} ({t / 15:.1f} s)", fontsize=8)
        fig.suptitle(f"{key.split('/')[-1]}  class={v['class']}", fontsize=9)
        fig.tight_layout()
        p = OUT / f"{key.split('/')[-1]}.png"; fig.savefig(p, dpi=90); plt.close(fig)
    index[key] = {"png": str(p), "frames": want, "n_extracted": n}
    print(key, n, flush=True)
(OUT / "index.json").write_text(json.dumps(index, indent=1))
