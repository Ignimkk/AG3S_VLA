"""HSR task-stage figure with the six stages (approach, grasp, lift, transport, place, retreat).

Top: one MuJoCo replay render per stage of one recorded closed-loop episode (T53 C1a ep1842), from an
external camera that follows the grasping hand. Bottom: per-chunk stage bands of the same episode,
physics ground truth vs the six-class HSR probe (T64 Part B, offline on the recorded features), and the
recorded gripper-latch grasp state.

Frame rule (fixed before rendering): the middle chunk of each ground-truth stage segment.

Run (CPU renderer):
    MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/hsr6_task_phase_figure.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "paper/tools/plotting"))
import paperbanana_render as PR  # noqa: E402

PARTB = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/"
                     "20261029T013324+0900_b3b69311f948_7da63daefa49/verify/T64/partB/partB.json")
GRASP_JSON = REPO / "paper/figures/paperbanana/figure3/assets_v2/f3v2_hsr_timeline.json"
OUT = REPO / "AG3S/ag3s/docs/figures/core-eval/paper"
NAME = "paper-hsr-task-phase-6stage"

STAGES = ["Approach", "Grasp", "Lift", "Transport", "Place", "Retreat"]
STAGE_COL = ["#E07A7A", "#E8B04A", "#8CC97E", "#3FAF8F", "#6F9BD8", "#A98BD3"]
GRASP_STATES = ["searching", "latched", "closing", "held", "placed"]
GRASP_COL = {"searching": "#EEEEEE", "latched": "#CFCFCF", "closing": "#F2A7C3", "held": "#7030A0", "placed": "#B9C7D6"}


def sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def segments(labels):
    segs, s = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            segs.append((labels[s], s, i - 1))
            s = i
    return segs


def band(ax, y, labels, colours, h=0.8):
    for lab, a, b in segments(list(labels)):
        ax.add_patch(Rectangle((a - 0.5, y - h / 2), b - a + 1, h, color=colours(lab), lw=0))


def main():
    ep = json.loads(PARTB.read_text())["episodes"]["ep1842"]
    truth, hsr = np.asarray(ep["truth"]), np.asarray(ep["hsr"])
    grasp = json.loads(GRASP_JSON.read_text())["grasp_states"]
    run, sc = PR.Run(), PR.Scene(1200, 900)
    n = len(truth)
    assert n == len(hsr) == len(grasp) == len(run.steps), (n, len(hsr), len(grasp), len(run.steps))

    # one frame per stage: middle chunk of each ground-truth segment
    picks = {}
    for lab, a, b in segments(list(truth)):
        picks.setdefault(int(lab), (a + b) // 2)
    frames = [picks[k] for k in range(6)]

    site = mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_SITE, "right_ee")
    imgs, cams = [], []
    for j in frames:
        sc.reset_colours()
        sc.set(run.step(j)["qpos"])
        ee = sc.d.site_xpos[site].copy()
        cam = {"lookat": [float(ee[0]), float(ee[1]), float(ee[2] - 0.08)], "dist": 0.50, "az": 155, "el": -22}
        imgs.append(sc.render(cam))
        cams.append(cam)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    W, H = 7.16, 2.55
    fig = plt.figure(figsize=(W, H), dpi=300)
    left, right, gap = 0.012, 0.012, 0.006
    iw = (1 - left - right - 5 * gap) / 6
    ih = iw * W * 0.75 / H
    itop = 0.905
    for k, (img, j) in enumerate(zip(imgs, frames)):
        ax = fig.add_axes([left + k * (iw + gap), itop - ih, iw, ih])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_edgecolor(STAGE_COL[k]); s_.set_linewidth(2.0)
        ax.set_title(STAGES[k], fontsize=8.5, pad=2.5,
                     bbox=dict(boxstyle="round,pad=0.22", fc=STAGE_COL[k], ec="none", alpha=0.9))

    bl = 0.14
    ax = fig.add_axes([bl, 0.155, 1 - bl - right, itop - ih - 0.205])
    band(ax, 2, truth, lambda l: STAGE_COL[int(l)])
    band(ax, 1, hsr, lambda l: STAGE_COL[int(l)])
    band(ax, 0, grasp, lambda l: GRASP_COL[l])
    for k, j in enumerate(frames):
        ax.plot([j, j], [-0.45, 2.45], color="black", lw=0.7, ls=(0, (2, 2)))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["Stage (physics)", "Stage (HSR)", "Grasp state"], fontsize=7.5)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlabel("Action chunk (8 control steps each); dashed lines = frames above", fontsize=6.5, labelpad=0)
    ax.tick_params(axis="x", labelsize=6.5, length=2, pad=1)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    gh = [Rectangle((0, 0), 1, 1, color=GRASP_COL[g]) for g in GRASP_STATES]
    fig.legend(gh, [f"grasp: {g}" for g in GRASP_STATES], ncol=5, fontsize=6.5, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5 + bl / 2, -0.005), handlelength=1.0, columnspacing=1.2)

    OUT.mkdir(parents=True, exist_ok=True)
    png, pdf = OUT / f"{NAME}.png", OUT / f"{NAME}.pdf"
    for p in (png, pdf):
        if p.exists():
            raise SystemExit(f"{p} exists; pick a new name instead of overwriting")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)

    agree = int((truth == hsr).sum())
    side = {
        "asset": str(png.relative_to(REPO)),
        "script": "paper/tools/plotting/hsr6_task_phase_figure.py",
        "renderer": f"MuJoCo {mujoco.__version__} replay of recorded qpos (paperbanana_render.Run/Scene)",
        "is_measured_result": False,
        "episode": "T53 C1a ep1842 (pear touching an orange); T64 Part B re-simulation is bit-identical",
        "frame_rule": "middle chunk of each ground-truth stage segment (fixed before rendering)",
        "frames": dict(zip(STAGES, frames)),
        "cameras": dict(zip(STAGES, cams)),
        "truth_segments": [[STAGES[int(l)], a, b] for l, a, b in segments(list(truth))],
        "hsr_segments": [[STAGES[int(l)], a, b] for l, a, b in segments(list(hsr))],
        "agreement_frames": [agree, n],
        "printed_numbers": [],
        "encoding": {"Stage (physics)": "T64 Part B physics ground truth (MuJoCo contact, height, basket region)",
                     "Stage (HSR)": "six-class linear probe on prefix KV layer 4, raw argmax, offline on recorded features",
                     "Grasp state": "recorded gripper-latch state of the run"},
        "sources": {str(PARTB): sha(PARTB), str(GRASP_JSON.relative_to(REPO)): sha(GRASP_JSON),
                    **{str(run.steps[j]): sha(run.steps[j]) for j in frames}},
        "limits": ["One episode, illustration.",
                   "HSR here is the offline six-stage probe on recorded features; serving at recording time used three phases."],
    }
    (OUT / f"{NAME}.json").write_text(json.dumps(side, indent=1, ensure_ascii=False) + "\n")
    print(png, side["frames"], "agree", agree, "/", n)


if __name__ == "__main__":
    main()
