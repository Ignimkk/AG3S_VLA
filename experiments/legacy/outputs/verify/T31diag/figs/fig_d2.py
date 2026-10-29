"""T31-diag D2 figures: per-chunk ablation deviations and a counts table."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
KO = fs.use_korean(); C = fs.CATEGORICAL
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t31'))
S = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d2/d2_summary.json')).read_text())
VAR = ["replay", "no_collision", "no_limits", "no_continuity", "no_smooth", "replay_3iter"]
runs = list(S["runs"])
fig, axes = plt.subplots(len(runs), 1, figsize=(16, 3.4 * len(runs)))
side = {"source": "outputs/verify/T31diag/d2/d2_summary.json", "runs": {}}
for ax, r in zip(axes, runs):
    ch = S["runs"][r]["chunks"]
    t = [c["t"] for c in ch]
    ax.plot(t, [c["recorded_dev_deg"] for c in ch], color=fs.INK, lw=2.2, marker="o", ms=4, label="recorded (server)" if not KO else "기록 (서버)")
    for i, v in enumerate(VAR):
        ax.plot(t, [c["ablation_dev_deg"][v] if c["ablation_dev_deg"][v] is not None else np.nan for c in ch], color=C[i], lw=1.1, marker=".", ms=5, label=v)
    for c in ch:
        w = c["ref_worst"]
        mk = "^" if (w["link"] or "").startswith("attached") else ("s" if w["link"] == "ee_finger_l1" else "D")
        ax.scatter([c["t"]], [18.5], marker=mk, s=40, facecolors=(C[7] if w["gt_nearest"] == "apple" else C[6]) if w["tier"] == "fine" else "none",
                   edgecolors=C[7] if w["gt_nearest"] == "apple" else C[6], lw=1.2)
        if c["executed"] is not None and c["n_control"] and c["executed"] < c["n_control"]:
            ax.axvspan(c["t"] - 3, c["t"] + 3, color=fs.GRID_INK, alpha=0.7, lw=0)
    ax.set_ylim(-0.5, 20); ax.set_ylabel("max |traj − reference| (deg)")
    ax.set_title(f"{r}  (fidelity: reconstructed worst row = recorded in {S['runs'][r]['fidelity_same_row']}/{S['runs'][r]['n_chunks']}, max |Δ| {S['runs'][r]['fidelity_max_abs_diff_mm']:.3f} mm;"
                 f" replay vs recorded refined median {S['runs'][r]['replay_vs_recorded_max_deg_median']:.2f}°)", fontsize=9.5)
    side["runs"][r] = [{"t": c["t"], "recorded": c["recorded_dev_deg"], **c["ablation_dev_deg"], "ref_worst": c["ref_worst"]} for c in ch]
axes[0].legend(ncol=7, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.93))
fs.style_axes(fig, list(axes))
fig.suptitle(("D2 — per-term ablation on fixed recorded inputs (fresh solver, clock off, iterations = recorded). Top markers: reference worst row "
              "(square ee_finger_l1, diamond ee_finger_l2, triangle held point; filled = fine tier, hollow = coarse; red = GT nearest apple, "
              "purple = crate). Grey = chunk not executed (HOLD)") if not KO else
             ("D2 — 고정 입력 위 항별 ablation (새 solver, 시계 끔, 반복 수 = 기록). 위 표식: reference 최악 행 (네모 ee_finger_l1, 마름모 ee_finger_l2, "
              "세모 쥔 점; 채움 = fine, 빈 = coarse; 빨강 = GT 최근접 사과, 보라 = crate). 회색 = 실행 안 된 청크 (HOLD)"), fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(F / "t31-d2-ablation.png", dpi=110)
(F / "t31-d2-ablation.json").write_text(json.dumps(side, indent=1, default=float))
# table
cols = runs
rowdefs = [("window t", lambda R: f"{R['window_t'][0]}–{R['window_t'][1]}"),
           ("chunks", lambda R: R["n_chunks"]),
           ("recorded dev ≥ 8°", lambda R: R["n_recorded_dev_ge_8deg"]),
           ("recorded dev = 0", lambda R: R["n_recorded_dev_eq_0"]),
           ("SQP iterations", lambda R: R["sqp_iterations_counts"]),
           ("returned", lambda R: R["returned_counts"]),
           ("plan-join disc. ≥ 0.14 rad", lambda R: R["n_plan_join_discontinuity_ge_0p14rad"]),
           ("ref worst link", lambda R: R["ref_worst_link_counts"]),
           ("ref worst tier", lambda R: R["ref_worst_tier_counts"]),
           ("ref worst GT nearest", lambda R: R["ref_worst_gt_nearest_counts"]),
           ("final worst GT nearest", lambda R: R["final_worst_gt_nearest_counts"]),
           ("ref beyond TO pos. limit (chunks)", lambda R: f"{R['n_chunks_ref_beyond_TO_position_limits']} {R['TO_limit_joints']}"),
           ("max overshoot (rad)", lambda R: f"{R['max_ref_beyond_TO_limit_rad']:.4f}"),
           ("ref beyond MuJoCo range (chunks)", lambda R: R["n_chunks_ref_beyond_mujoco_range"]),
           ] + [(f"ablation median dev: {v}", (lambda v: lambda R: f"{R['ablation_dev_deg_median'][v]:.2f}° (≥8°: {R['ablation_n_ge_8deg'][v]})")(v)) for v in VAR]
cell = [[str(f(S["runs"][r])).replace("'", "") for r in cols] for _, f in rowdefs]
fig, ax = plt.subplots(figsize=(22, 0.46 * len(rowdefs) + 1.2))
ax.axis("off")
tb = ax.table(cellText=cell, rowLabels=[n for n, _ in rowdefs], colLabels=cols, loc="center", cellLoc="left")
tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.45)
for (i, j), cc in tb.get_celld().items():
    cc.set_edgecolor(fs.GRID_INK)
    if i == 0:
        cc.set_facecolor(fs.SEQ_BLUE[0])
fig.patch.set_facecolor(fs.SURFACE)
ax.set_title("D2 — rows and terms, grasp → release window (counts over chunks; source d2_summary.json)" if not KO else
             "D2 — 행과 항, grasp → release 구간 (청크 개수; 출처 d2_summary.json)", fontsize=12)
fig.savefig(F / "t31-d2-rows-table.png", dpi=110, bbox_inches="tight")
(F / "t31-d2-rows-table.json").write_text(json.dumps({"rows": [n for n, _ in rowdefs], "cols": cols, "cells": cell}, indent=1, ensure_ascii=False))
print("ok")
