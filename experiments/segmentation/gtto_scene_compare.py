#!/usr/bin/env python3
"""Real-scene comparison (rule A): agentview frames of the same LIBERO episode under several arms, side by side at the same
env steps (0 · 25 · 50 · 75 % of the longest episode · last), plus each arm's video copied next to the figure.
usage: gtto_scene_compare.py --out DIR --stem libero_spatial_t4_e0 --arm VLA=<run glob> --arm F8=<run glob> ..."""
import argparse
import json
import pathlib
import shutil

import numpy as np

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")


def newest(pattern):
    runs = [p for p in sorted(RESULTS.glob(pattern)) if (p / "episodes.jsonl").is_file()]
    return runs[-1] if runs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME=<run dir glob under results/benchmark>")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    arms = []
    for spec in a.arm:
        name, pat = spec.split("=", 1)
        run = newest(pat)
        if run is None:
            raise SystemExit(f"no run for {name}: {pat}")
        z = np.load(run / "episodes" / f"{a.stem}.npz", allow_pickle=True)
        img = z["obs_agentview_image"]
        t, e = a.stem.split("_t")[1].split("_e")
        ep = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()
              and json.loads(l)["task_id"] == int(t) and json.loads(l)["episode_idx"] == int(e)][0]
        arms.append({"name": name, "run": run.name, "img": img, "success": bool(ep["official_success"])})
        vids = sorted((run / "videos").glob(f"*{a.stem}*")) if (run / "videos").is_dir() else []
        for v in vids[:1]:
            shutil.copy2(v, a.out / f"{name}_{a.stem}_{'success' if ep['official_success'] else 'failure'}{v.suffix}")
    n = max(len(x["img"]) for x in arms)
    ts = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(arms), len(ts), figsize=(2.6 * len(ts), 2.8 * len(arms)))
    ax = np.atleast_2d(ax)
    for i, x in enumerate(arms):
        for j, t in enumerate(ts):
            k = min(t, len(x["img"]) - 1)
            ax[i, j].imshow(x["img"][k][::-1, ::-1])   # LIBERO agentview is stored upside down
            ax[i, j].set_xticks([]), ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(f"step {t}" + (" (ended)" if t >= len(x["img"]) else ""), fontsize=9)
            if j == 0:
                ax[i, j].set_ylabel(f"{x['name']}\n{'success' if x['success'] else 'FAIL'} ({len(x['img'])} steps)", fontsize=9)
    fig.suptitle(a.title or a.stem, fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out / f"장면비교_{a.stem}.png", dpi=110)
    (a.out / f"장면비교_{a.stem}.json").write_text(json.dumps([{k: v for k, v in x.items() if k != "img"} for x in arms], indent=1))
    print(a.out / f"장면비교_{a.stem}.png")


if __name__ == "__main__":
    main()
