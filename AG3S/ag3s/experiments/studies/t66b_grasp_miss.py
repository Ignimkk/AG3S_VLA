"""T66b — why the final configuration still misses the apple near neighbours / the occluding post (diagnosis, read-only on
session A's records).

Inputs are the per-run JSONs of the scratch diagnosis (`pushdiag` rows per request: palm shift refined − VLA reference,
initial violation, worst pair and its nearest GT object; `closepose`: palm position at the first closing command, AG3S
and VLA-alone, against the apple at t=0; `agg`: per-run window summary). This script only draws:
  scene_hand_path.png    top and side view of the palm path up to closing, VLA alone vs AG3S, with GT objects
  graph_push_vs_outcome.png   per run: largest TrajOpt palm shift in the last 60 mm, and palm–apple distance at closing
  table_t66b.md

usage: MUJOCO_GL=osmesa SN=<snapshot> t66b_grasp_miss.py DIAG_DIR OUTPUTS_DIR   (.venv-openpi-live)
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
import figstyle  # noqa: E402

CE = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261028T224856+0900_b3b69311f948_8784a1265958"
                  "/verify/AG3S-CORE-EVAL")
SCENE_RUNS = ("hurdle954_k0", "hurdle211_k0", "hurdle1345_k0")


def gt_objects(lab):
    sys.path.insert(0, os.environ["SN"])
    from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as tde
    items = {line.split(":")[1]: line.strip().split(":") for line in open(CE / "hurdle/hurdle_items.txt")}
    ep, k = lab[6:].split("_")
    it = items[ep]
    gt = tde.scene_truth(os.environ["SN"] + "/robots/rby1/description/models/rby1a/mujoco/"
                         "model_transport_pick_place_obstacles.xml", (it[4], it[5]))
    st0 = sorted((CE / f"hurdle/ag3s/{k}/runs/{it[0]}/ag3s/run_0000").glob("step_*.npz"),
                 key=lambda p: int(np.load(p)["t_step"]))[0]
    samples, _ = gt.at(np.asarray(np.load(st0)["qpos"], float))
    return {gt.kind[i][1]: p for i, p in enumerate(samples) if len(p) and gt.kind[i][1] not in ("table", "crate")}


def main(diag: pathlib.Path, o: pathlib.Path) -> None:
    o.mkdir(parents=True, exist_ok=True)
    cp = json.load(open(diag / "closepose.json"))
    agg = json.load(open(diag / "agg.json"))
    cols = figstyle.CATEGORICAL

    # scene: palm path top (x-y) and side (x-z), VLA alone vs AG3S
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    side = {}
    for c, lab in enumerate(SCENE_RUNS):
        objs = gt_objects(lab)
        rec = cp[lab]
        apple = np.asarray(rec["apple0"])
        for r, (i, j, name) in enumerate(((0, 1, "y"), (0, 2, "z"))):
            ax = axes[r, c]
            for nm, p in objs.items():
                near = np.linalg.norm(p - apple, axis=1) < 0.25
                if not near.any():
                    continue
                kind = "target apple" if nm == "apple" else ("occluding post" if "post" in nm else
                                                             ("hurdle" if "obstacle" in nm else f"neighbour {nm}"))
                col = cols[2] if nm == "apple" else (figstyle.INK_2 if "obstacle" in nm else cols[3])
                ax.scatter(p[near, i], p[near, j], s=1, color=col, alpha=0.35, label=kind)
            for w, (who, ls) in enumerate((("VLA", "-"), ("AG3S", "-"))):
                if not rec.get(who):
                    continue
                P = np.asarray(rec[who]["path"] + [rec[who]["ee"]])
                ax.plot(P[:, i], P[:, j], ls, color=cols[0] if who == "VLA" else cols[1], lw=1.6,
                        label=f"{'VLA alone' if who == 'VLA' else 'AG3S final'} palm path")
                e = rec[who]["ee"]
                ax.scatter([e[i]], [e[j]], marker="x", s=60, color=cols[0] if who == "VLA" else cols[1],
                           label=f"closing starts ({rec[who]['dist_mm']:.0f} mm from apple centre)")
            ax.set_xlim(apple[i] - 0.16, apple[i] + 0.10)
            ax.set_ylim(apple[j] - 0.10, apple[j] + (0.10 if name == "y" else 0.16))
            ax.set_aspect("equal")
            ax.set_xlabel("x (m, robot base)")
            ax.set_ylabel(f"{name} (m)")
            ax.set_title(f"{lab.replace('hurdle', 'scene ').replace('_k', ', draw ')}: "
                         f"{'top view' if name == 'y' else 'side view'}", fontsize=9)
            if c == 0:
                h, lb = ax.get_legend_handles_labels()
                d = dict(zip(lb, h))
                ax.legend(d.values(), d.keys(), fontsize=6, markerscale=4, loc="lower left")
        side[lab] = {w: (rec[w] or {}).get("dist_mm") for w in ("VLA", "AG3S")}
    figstyle.style_axes(fig, list(axes.ravel()))
    fig.suptitle("Palm path up to the first closing command (same scene and seed): VLA alone vs AG3S final", fontsize=10)
    fig.tight_layout()
    fig.savefig(o / "scene_hand_path.png", dpi=150)
    plt.close(fig)
    (o / "scene_hand_path.json").write_text(json.dumps(side, indent=1))

    # graph: per run push in the last 60 mm and closing distance
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    runs = [x["run"] for x in agg]
    lab_en = [r.replace("hurdle", "hurdle ").replace("회귀", "regr ").replace("가림", "occl ").replace("_k", " d") for r in runs]
    colr = [cols[7] if x["grasp_failed"] else cols[0] for x in agg]
    y = np.arange(len(runs))
    axes[0].barh(y, [x["max_shift_win_mm"] for x in agg], color=colr)
    axes[0].set_yticks(y, lab_en, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("largest palm shift by TrajOpt in one request,\nlast 60 mm before closing (mm)")
    axes[1].barh(y, [x["close_dist_mm"] or 0 for x in agg], color=colr)
    vla = [x["vla_close_dist_mm"] for x in agg]
    axes[1].scatter([v if v else np.nan for v in vla], y, marker="|", s=120, color=figstyle.INK, label="VLA alone, same scene")
    axes[1].set_yticks(y, lab_en, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("palm to apple centre when closing starts (mm)")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_title("red = grasp missed (apple not lifted), blue = grasped", fontsize=9)
    figstyle.style_axes(fig, list(axes))
    fig.tight_layout()
    fig.savefig(o / "graph_push_vs_outcome.png", dpi=150)
    plt.close(fig)
    (o / "graph_push_vs_outcome.json").write_text(json.dumps(agg, ensure_ascii=False, indent=1))

    lines = ["| run | 결과 | 마지막 60 mm 요청 수 | 한 요청 최대 밀림 (mm) | 8 mm 이상 밀린 요청 | 대체 solver | 닫기 시작 손–사과 (mm) | VLA 단독 (mm) | 밀어낸 제약의 가장 가까운 물체 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for x in agg:
        lines.append(f"| {x['run']} | {'쥐지 못함' if x['grasp_failed'] else '쥠'} | {x['n_win']} | {x['max_shift_win_mm']:.1f} | "
                     f"{x['n_push8_win']} | {x['n_ms_win']} | {x['close_dist_mm']} | {x['vla_close_dist_mm'] or '-'} | "
                     f"{', '.join(f'{k} {v}' for k, v in x['blockers'].items()) or '-'} |")
    (o / "table_t66b.md").write_text("\n".join(lines) + "\n")
    print("->", o)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
