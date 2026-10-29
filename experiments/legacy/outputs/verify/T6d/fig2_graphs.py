"""T6d figure 2 — 그래프 넷.

(a) 보고된 최악 clearance (세 실행, 75 chunk) + HOLD 표시
(b) clearance 에서 되짚은 ESDF 거리 대 참값 최근접 씬 표면 (같은 질의점, 같은 frame)
(c) shadow: 참값 clearance (사과 제외) — reference 대 refined, 실행된 앞 8 스텝
(d) 실행 창에서 왼손끝이 사과 표면까지: shadow reference · shadow refined · 두 closed-loop
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t6d/t6d-trend.png'))
SIDE = OUT.with_suffix(".json")
EXEC_H = 8
LBL = {"closed_loop_approach": "closed-loop / approach",
       "shadow_approach": "shadow / approach",
       "closed_loop_grasp": "closed-loop / grasp"}
COL = {"closed_loop_approach": figstyle.CATEGORICAL[0],
       "shadow_approach": figstyle.CATEGORICAL[1],
       "closed_loop_grasp": figstyle.CATEGORICAL[2]}


def main() -> None:
    gt = json.loads((V / "m2c_gt_unsaturated.json").read_text())
    m3c = json.loads((V / "m3c_clearance_split.json").read_text())
    z3 = np.load(V / "m3_curves.npz")
    z3c = np.load(V / "m3c_curves.npz")

    figstyle.use_korean()
    fig, ax = plt.subplots(2, 2, figsize=(14.6, 9.0))
    figstyle.style_axes(fig, ax)
    side: dict = {}

    # ---- (a) 보고된 최악 clearance -----------------------------------------------------
    a = ax[0, 0]
    for run, rows in gt["runs"].items():
        seq = [r["seq"] for r in rows]
        cl = [r["clearance_mm"] for r in rows]
        a.plot(seq, cl, lw=1.4, color=COL[run], label=LBL[run])
        h = [(r["seq"], r["clearance_mm"]) for r in rows if r["hold_reason"]]
        if h:
            a.scatter([x for x, _ in h], [y for _, y in h], s=48, marker="v",
                      color=COL[run], zorder=5, edgecolor=figstyle.SURFACE, lw=0.8)
        side.setdefault("a_reported_clearance_mm", {})[run] = cl
    a.axhline(0.0, color=figstyle.INK_2, lw=0.9, ls="--")
    a.set_xlabel("chunk seq", fontsize=9, color=figstyle.INK_2)
    a.set_ylabel("reported worst clearance  [mm]", fontsize=9, color=figstyle.INK_2)
    a.set_title("(a) what the safety layer reported — triangles are HOLD (0 / 6 / 1)",
                fontsize=10, color=figstyle.INK)
    a.legend(fontsize=8, frameon=False)

    # ---- (b) 되짚은 ESDF 거리 대 참값 --------------------------------------------------
    b = ax[0, 1]
    for run, rows in gt["runs"].items():
        seq = [r["seq"] for r in rows]
        imp = [r["implied_esdf_distance_mm"] for r in rows]
        g = [r["gt_scene_min_mm"] for r in rows]
        b.plot(seq, imp, lw=1.4, color=COL[run], label=f"{LBL[run]} — ESDF (implied)")
        b.plot(seq, g, lw=1.2, ls=":", color=COL[run], label=f"{LBL[run]} — ground truth")
        side.setdefault("b_implied_esdf_mm", {})[run] = imp
        side.setdefault("b_gt_scene_min_mm", {})[run] = g
    b.set_xlabel("chunk seq", fontsize=9, color=figstyle.INK_2)
    b.set_ylabel("distance from the query point  [mm]", fontsize=9, color=figstyle.INK_2)
    b.set_title("(b) ESDF distance read back from the reported clearance, next to ground truth\n"
                "solid = clearance + sphere radius + margin · dotted = nearest ground-truth "
                "scene surface", fontsize=10, color=figstyle.INK)
    b.legend(fontsize=6.6, frameon=False, ncol=2)

    # ---- (c) shadow 참값 clearance, 사과 제외, 실행 창 ---------------------------------
    c = ax[1, 0]
    rows = m3c["shadow_chunks"]
    seq = [r["seq"] for r in rows]
    r0 = [r["reference_no_apple_min_mm_exec8"] for r in rows]
    r1 = [r["refined_no_apple_min_mm_exec8"] for r in rows]
    c.plot(seq, r0, lw=1.5, color=figstyle.CATEGORICAL[3], label="reference (executed 8 steps)")
    c.plot(seq, r1, lw=1.5, color=figstyle.CATEGORICAL[0], label="refined (executed 8 steps)")
    c.fill_between(seq, r0, r1, where=np.asarray(r1) >= np.asarray(r0),
                   color=figstyle.CATEGORICAL[0], alpha=0.14)
    c.axhline(0.0, color=figstyle.INK_2, lw=0.9, ls="--")
    c.set_xlabel("chunk seq", fontsize=9, color=figstyle.INK_2)
    c.set_ylabel("ground-truth clearance, apple excluded  [mm]", fontsize=9, color=figstyle.INK_2)
    s = m3c["shadow_no_apple_exec8"]
    c.set_title(f"(c) apple excluded — refined below reference in "
                f"{s['n_refined_worse']} of {s['n_chunks']} chunks\n"
                f"global min {s['reference_global_min_mm']:+.2f} -> "
                f"{s['refined_global_min_mm']:+.2f} mm", fontsize=10, color=figstyle.INK)
    c.legend(fontsize=8, frameon=False)
    side["c_reference_no_apple_exec8_mm"] = r0
    side["c_refined_no_apple_exec8_mm"] = r1

    # ---- (d) 손끝에서 사과 표면까지, 실행 창 -------------------------------------------
    dax = ax[1, 1]
    series = {
        "shadow reference (executed)": (np.nanmin(z3["shadow_tip_reference_mm"][:, :EXEC_H], 1),
                                        figstyle.CATEGORICAL[3], "-"),
        "shadow refined (executed)": (np.nanmin(z3["shadow_tip_refined_mm"][:, :EXEC_H], 1),
                                      figstyle.CATEGORICAL[1], "-"),
        "closed-loop approach (executed)": (
            np.nanmin(z3["closed_loop_approach_tip_mm"][:, :EXEC_H], 1),
            figstyle.CATEGORICAL[0], "-"),
        "closed-loop grasp (executed)": (
            np.nanmin(z3["closed_loop_grasp_tip_mm"][:, :EXEC_H], 1),
            figstyle.CATEGORICAL[2], "-"),
        "closed-loop approach (full 50-step plan)": (
            np.nanmin(z3["closed_loop_approach_tip_mm"], 1), figstyle.CATEGORICAL[0], ":"),
    }
    for name, (v, colr, ls) in series.items():
        dax.plot(np.arange(1, v.size + 1), v, lw=1.4, ls=ls, color=colr, label=name)
        side.setdefault("d_tip_to_apple_mm", {})[name] = [float(x) for x in v]
    dax.axhline(0.0, color=figstyle.INK_2, lw=0.9, ls="--")
    for xv, txt in ((8, "diff > 50 mm from chunk 8"), (10, "reference first below 0 at chunk 10")):
        dax.axvline(xv, color=figstyle.GRID_INK, lw=1.0)
        dax.annotate(txt, (xv + 0.6, 455), fontsize=7, color=figstyle.INK_2, rotation=90,
                     va="top")
    dax.set_xlabel("chunk seq", fontsize=9, color=figstyle.INK_2)
    dax.set_ylabel("left fingertip to apple surface  [mm]", fontsize=9, color=figstyle.INK_2)
    dax.set_title("(d) only 8 of each 50-step chunk were ever executed\n"
                  "(step_in_chunk observed 0..7 in all three runs)",
                  fontsize=10, color=figstyle.INK)
    dax.legend(fontsize=7.2, frameon=False)

    fig.suptitle("T6d — where the three stages stand, measured on three recordings "
                 "(75 chunks each, no new capture)", fontsize=12.5, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
