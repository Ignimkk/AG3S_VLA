"""Figures for the BM-SETUP-B CPU probe (``libero_safety_probe.py`` output). ``.venv-ag3s`` (matplotlib).

    .venv-ag3s/bin/python -m benchmark.evaluators.libero_safety_probe_figures --probe outputs/verify/BM-SETUP-B/probe \
        --out outputs/verify/BM-SETUP-B/figures

0. ``probe_all_tasks.png``: every probed task after the wait steps (suite x level rows, task columns).
1. ``probe_scenes.png``: per suite x level (task 0), agentview at the end of the wait steps (OpenCV rows) and at the first
   detected violation of the scripted drive.
2. ``probe_detection.png``: per probe, first drive step at which each detector fired (official ``info['cost']``,
   per-constraint ``_eval_predicate``, geom-id robot contact) -- the table of the CPU check, as a dot plot.
3. ``image_orientation.png``: released dataset frame 0 vs this env's raw render under rot180 / vflip / none.
Each PNG has a same-name ``.json`` sidecar with the numbers drawn.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _style():
    import matplotlib

    matplotlib.use("Agg")
    from AG3S.ag3s.experiments.common import figstyle

    figstyle.use_korean()
    return figstyle


def load(probe):
    rows = []
    for p in sorted(pathlib.Path(probe).glob("*_L*_t*.json")):
        r = json.loads(p.read_text())
        f = p.with_name(p.stem + "_frames.npz")
        r["_frames"] = str(f) if f.exists() else None
        rows.append(r)
    return rows


def scenes(rows, out, fs):
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r.get("ok") and r["_frames"] and r["task_id"] == 0]
    fig, axes = plt.subplots(2, len(ok), figsize=(1.6 * len(ok), 3.8))
    axes = np.atleast_2d(axes).reshape(2, -1)
    side = []
    for j, r in enumerate(ok):
        z = np.load(r["_frames"])
        axes[0, j].imshow(z["agentview_rgb_cv"])
        v = z["agentview_rgb_cv_violation"]
        if v.size:
            axes[1, j].imshow(v)
        axes[0, j].set_title("{}\nL{} t{}".format(r["suite"].replace("obstacle_avoidance", "OA"), r["level"],
                                                   r["task_id"]), fontsize=6, color=fs.INK)
        for i in range(2):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        side.append({"suite": r["suite"], "level": r["level"], "task": r["task"],
                     "first_detection_step": r["drive_first_detection_step"], "drive_target": r["drive_target"]})
    axes[0, 0].set_ylabel("after wait", fontsize=7)
    axes[1, 0].set_ylabel("first violation", fontsize=7)
    fs.style_axes(fig, axes)
    fig.suptitle("LIBERO-Safety CPU probe: agentview (OpenCV rows), init state 0", fontsize=9, color=fs.INK)
    fig.tight_layout()
    fig.savefig(out / "probe_scenes.png", dpi=170)
    (out / "probe_scenes.json").write_text(json.dumps(side, indent=1))
    plt.close(fig)


def grid(rows, out, fs):
    """All probed tasks after the wait steps: rows = suite x level, columns = task id."""
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r.get("ok") and r["_frames"]]
    keys = sorted({(r["suite"], r["level"]) for r in ok})
    tids = sorted({r["task_id"] for r in ok})
    fig, axes = plt.subplots(len(keys), len(tids), figsize=(1.5 * len(tids) + 1.2, 1.5 * len(keys)))
    axes = np.atleast_2d(axes)
    side = {}
    for r in ok:
        i, j = keys.index((r["suite"], r["level"])), tids.index(r["task_id"])
        axes[i, j].imshow(np.load(r["_frames"])["agentview_rgb_cv"])
        axes[i, j].set_title(r["task"][:28], fontsize=4.5, color=fs.INK)
        side["{}_L{}_t{}".format(r["suite"], r["level"], r["task_id"])] = r["task"]
    for i, (s_, l_) in enumerate(keys):
        axes[i, 0].set_ylabel("{}\nL{}".format(s_.replace("obstacle_avoidance", "OA"), l_), fontsize=6)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fs.style_axes(fig, axes)
    fig.suptitle("LIBERO-Safety: all 60 rollout tasks after the wait steps (init state 0, agentview)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "probe_all_tasks.png", dpi=150)
    (out / "probe_all_tasks.json").write_text(json.dumps(side, indent=1))
    plt.close(fig)


def detection(rows, out, fs):
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r.get("ok")]
    labels = ["{} L{} t{}".format(r["suite"], r["level"], r["task_id"]) for r in ok]
    keys = [("official", "official info['cost']"), ("full", "per-constraint _eval_predicate"),
            ("geom", "geom-id robot contact"), ("done", "task success")]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(ok) + 1.2))
    side = {}
    for k, (key, name) in enumerate(keys):
        xs, ys = [], []
        for i, r in enumerate(ok):
            v = r["drive_first_detection_step"][key]
            side.setdefault(labels[i], {})[key] = v
            if v is not None:
                xs.append(v)
                ys.append(i + (k - 1.5) * 0.15)
        ax.scatter(xs, ys, s=18, color=fs.CATEGORICAL[k], label=name, zorder=3)
    for i, r in enumerate(ok):
        if r["drive_target"][0] is None:
            msg = ("no constraint (affordance: safety is in the goal)" if not r["constraints"] else
                   "not driven: only {} (see official_cost_check.json)".format(r["constraints"][0][0]))
            ax.text(0, i, msg, fontsize=6, va="center", color=fs.INK_2)
        elif r["drive_first_detection_step"]["geom"] is None:
            ax.text(0, i, "drive did not reach {} (min {:.2f} m)".format(r["drive_target"][0], r["drive_min_distance_m"]),
                    fontsize=6, va="center", color=fs.INK_2)
    ax.set_yticks(range(len(ok)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("scripted drive step at first detection (none = never fired)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fs.style_axes(fig, ax)
    ax.set_title("Scripted gripper-obstacle contact: official cost never fires, geom-id contact does", fontsize=9, color=fs.INK)
    fig.tight_layout()
    fig.savefig(out / "probe_detection.png", dpi=170)
    (out / "probe_detection.json").write_text(json.dumps(side, indent=1))
    plt.close(fig)


def orientation(rows, out, fs, dataset_png):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    target = [r for r in rows if r.get("ok") and r["_frames"] and r["task"].startswith("pour_the_water")]
    if not target or not pathlib.Path(dataset_png).exists():
        return
    z = np.load(target[0]["_frames"])
    raw = z["agentview_raw_init"][::-1]  # stored as OpenCV rows; raw render = OpenGL rows
    ds = (mpimg.imread(dataset_png)[..., :3] * 255).astype(np.float64)
    cands = {"rot180 [::-1, ::-1]": raw[::-1, ::-1], "vflip [::-1]": raw[::-1], "none": raw}
    err = {k: float(np.mean(np.abs(v.astype(np.float64) - ds))) for k, v in cands.items()}
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.4))
    axes[0].imshow(ds.astype(np.uint8))
    axes[0].set_title("dataset ep0 frame0", fontsize=7)
    for ax, (k, v) in zip(axes[1:], cands.items()):
        ax.imshow(v)
        ax.set_title("{}\nmean |diff| {:.1f}".format(k, err[k]), fontsize=7)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fs.style_axes(fig, axes)
    fig.tight_layout()
    fig.savefig(out / "image_orientation.png", dpi=170)
    (out / "image_orientation.json").write_text(json.dumps({"task": target[0]["task"], "level": target[0]["level"],
                                                            "mean_abs_diff": err}, indent=1))
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-png", default=str(ROOT / "outputs/verify/BM-SETUP-B/dataset_frames/"
                                                        "ep0_observation.image_f0.png"))
    a = ap.parse_args(argv)
    fs = _style()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(a.probe)
    scenes(rows, out, fs)
    grid(rows, out, fs)
    detection(rows, out, fs)
    orientation(rows, out, fs, a.dataset_png)


if __name__ == "__main__":
    main()
