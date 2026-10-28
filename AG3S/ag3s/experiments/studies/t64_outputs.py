"""T64 — build the human-facing outputs folder (videos, scene strip, timelines, graphs, tables, README).

usage: t64_outputs.py T64_VERIFY_DIR OUTPUTS_DIR      (.venv-ag3s; matplotlib)
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

NAMES = ("approach", "grasp", "lift", "transport", "placement", "retreat", "failure")
T55 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261013T171325+0900_cf38c9dbca7b_2ebfd8a5a43c/verify/T55")


def main(v: pathlib.Path, o: pathlib.Path) -> None:
    o.mkdir(parents=True, exist_ok=True)
    A = json.loads((v / "partA/partA.json").read_text())
    Bj = json.loads((v / "partB_v/partB.json").read_text())
    (o / "videos").mkdir(exist_ok=True)
    for f in sorted((v / "partB_v/videos").glob("*.mp4")):
        shutil.copy2(f, o / "videos" / f.name)
    cols = figstyle.CATEGORICAL

    # 1. timelines (truth / HSR / rule) for every episode
    eps = sorted(Bj["episodes"])
    fig, axes = plt.subplots(len(eps), 1, figsize=(12, 0.62 * len(eps) + 1), sharex=True)
    for ax, k in zip(axes, eps):
        e = Bj["episodes"][k]
        for row, key in enumerate(("truth", "hsr", "rule")):
            for t, c in enumerate(e[key]):
                ax.add_patch(plt.Rectangle((t, 2 - row), 1, 0.9, color=cols[c % len(cols)], lw=0))
        ax.set_xlim(0, 75); ax.set_ylim(0, 3); ax.set_yticks([0.45, 1.45, 2.45], ["rule", "HSR", "truth"], fontsize=6)
        ax.set_ylabel(k, rotation=0, ha="right", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cols[i]) for i in range(7)]
    fig.legend(handles, NAMES, ncol=7, loc="upper center", fontsize=8)
    axes[-1].set_xlabel("policy frame (every 8 control steps)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(o / "timeline_all_episodes.png", dpi=140); plt.close(fig)

    # 2. graph: per-phase F1 (Part A expert val, Part B HSR vs rule)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(6)
    a = [A["six"]["val"]["per_class"][n]["f1"] for n in NAMES[:6]]
    h = [Bj["per_class_f1"][n]["hsr"] for n in NAMES[:6]]
    r = [Bj["per_class_f1"][n]["rule"] for n in NAMES[:6]]
    for i, (lab, vals) in enumerate((("HSR on expert val (Part A)", a), ("HSR on closed-loop (Part B)", h),
                                      ("rule on closed-loop (Part B)", r))):
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=lab, color=figstyle.SEQ_BLUE[[2, 5, 9][i] if len(figstyle.SEQ_BLUE) > 9 else [1, 3, 5][i]] if i < 2 else figstyle.CATEGORICAL[1])
    ax.set_xticks(x, NAMES[:6]); ax.set_ylabel("F1"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_title("6-phase F1 per phase")
    figstyle.style_axes(fig, [ax]); fig.tight_layout(); fig.savefig(o / "graph_phase_f1.png", dpi=150); plt.close(fig)

    # 3. graph: per-episode macro-F1 HSR vs rule
    fig, ax = plt.subplots(figsize=(9, 3.4))
    fh = [Bj["episodes"][k]["f1_hsr"] for k in eps]; fr = [Bj["episodes"][k]["f1_rule"] for k in eps]
    ax.bar(np.arange(len(eps)) - 0.2, fh, 0.4, label="HSR", color=figstyle.SEQ_BLUE[5])
    ax.bar(np.arange(len(eps)) + 0.2, fr, 0.4, label="rule", color=figstyle.CATEGORICAL[1])
    ax.set_xticks(range(len(eps)), eps, rotation=60, fontsize=7); ax.set_ylabel("macro-F1"); ax.legend(fontsize=8)
    m = Bj["macro_f1"]
    ax.set_title(f"per-episode macro-F1 — mean HSR {m['hsr_mean']:.3f} · rule {m['rule_mean']:.3f} · "
                 f"diff {m['diff_mean']:+.3f} [{m['diff_ci95'][0]:+.3f}, {m['diff_ci95'][1]:+.3f}]", fontsize=9)
    figstyle.style_axes(fig, [ax]); fig.tight_layout(); fig.savefig(o / "graph_episode_f1.png", dpi=150); plt.close(fig)

    # 4. scene strip: one episode, head camera at the truth phase changes
    k = "ep1804"
    z = np.load(T55 / f"resim/test/{k}__E0.npz", allow_pickle=False)
    e = Bj["episodes"][k]; y = e["truth"]
    idx = [0] + [t for t in range(1, len(y)) if y[t] != y[t - 1]]
    fig, axes = plt.subplots(1, len(idx), figsize=(2.4 * len(idx), 2.9))
    for ax, t in zip(np.atleast_1d(axes), idx):
        ax.imshow(z["image_cam_high"][t]); ax.axis("off")
        ax.set_title(f"f{t}\ntruth {NAMES[y[t]]}\nHSR {NAMES[e['hsr'][t]]}\nrule {NAMES[e['rule'][t]]}", fontsize=7)
    fig.tight_layout(); fig.savefig(o / f"scene_{k}_phase_changes.png", dpi=140); plt.close(fig)

    # 5. tables (json + md)
    tab = {"partA": {"val_macro_f1": A["six"]["val"]["macro_f1"], "decision": A["decision"], "C": A["six"]["C"],
                     "per_class": A["six"]["val"]["per_class"]},
           "partB": {"macro_f1": Bj["macro_f1"], "per_class_f1": Bj["per_class_f1"], "lead": Bj["lead"],
                     "n_failure_frames": Bj["n_failure_frames"]}}
    (o / "tables.json").write_text(json.dumps(tab, indent=1, default=float))
    lines = ["| 단계 | expert val F1 (Part A) | closed-loop HSR F1 | closed-loop rule F1 | 정답 frame 수 |", "|---|---|---|---|---|"]
    for n in NAMES[:6]:
        lines.append(f"| {n} | {A['six']['val']['per_class'][n]['f1']:.3f} | {Bj['per_class_f1'][n]['hsr']:.3f} | "
                     f"{Bj['per_class_f1'][n]['rule']:.3f} | {Bj['per_class_f1'][n]['n']} |")
    (o / "table_phase_f1.md").write_text("\n".join(lines) + "\n")
    print("->", o)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
