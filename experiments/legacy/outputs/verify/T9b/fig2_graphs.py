"""T9b figure 2 — N2 의 mm 차이 시계열과 N4 의 관절별 기여.

(1) chunk 축 · 손끝-사과 거리: 원본 대 TO 출력 (실행 창 최소)
(2) chunk 축 · 손끝 위치 차이 mm (l1) 와 성분 x·y·z
(3) 실행 창 step 축 · 평균 차이 (TO 가 실행 창을 따라 얼마나 벌리는가)
(4) N4 관절별 손끝 변위 기여 mm 대 관절별 편차 deg — 같은 축에 나란히
(5) N4 기여 점유율
(6) N3 chunk 14 원본 50 step 손끝-사과 거리
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

T9B = pathlib.Path(_ag3s_workspace_path('outputs/verify/T9b'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t9b/t9b-graphs.png'))
SIDE = OUT.with_suffix(".json")
EXEC_H = 8


def main() -> None:
    z = np.load(T9B / "n1to5_curves.npz", allow_pickle=True)
    ana = json.loads((T9B / "n1to5.json").read_text())
    n1, n2, n3, n4 = (ana["n1_link_to_apple"], ana["n2_fingertip_divergence"],
                      ana["n3_reference_full_horizon"], ana["n4_joint_contribution_mm"])
    ta_r = np.asarray(z["tip_apple_reference"], np.float64).min(axis=1)
    ta_t = np.asarray(z["tip_apple_refined"], np.float64).min(axis=1)
    tip_r = np.asarray(z["tip_xyz_reference"], np.float64)[:, :, 0, :]
    tip_t = np.asarray(z["tip_xyz_refined"], np.float64)[:, :, 0, :]
    diff = (tip_r - tip_t) * 1000.0
    norm = np.linalg.norm(diff, axis=2)
    jc = np.asarray(z["jac_contrib"], np.float64)
    joints = [str(x) for x in z["joints"]]
    F = ta_r.size
    x = np.arange(F)

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 3, figsize=(18.6, 9.8))
    figstyle.style_axes(fig, axes)
    side: dict = {"exec_window": EXEC_H, "panels": {}}

    # (1)
    ax = axes[0, 0]
    ax.plot(x, ta_r, color=figstyle.CATEGORICAL[2], lw=1.8, label="policy reference")
    ax.plot(x, ta_t, color=figstyle.CATEGORICAL[1], lw=1.8, label="TO output")
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    ax.axhline(10.0, color=figstyle.CATEGORICAL[7], lw=1.0, ls="--")
    ax.annotate("esdf margin 10 mm", (F - 1, 10.0), ha="right", va="bottom", fontsize=7.4,
                color=figstyle.CATEGORICAL[7])
    ic = n1["global_min_chunk_reference"]
    ax.plot([ic], [ta_r[ic]], marker="*", ms=15, color=figstyle.CATEGORICAL[2], zorder=9)
    ax.annotate(f"chunk {ic}: reference {ta_r[ic]:.2f} mm, TO {ta_t[ic]:.2f} mm",
                (ic, ta_r[ic]), textcoords="offset points", xytext=(10, 14), fontsize=8.0,
                color=figstyle.INK)
    ax.set_ylim(-20, 460)
    ax.set_xlabel("chunk (75 planning records)", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("left fingertip to apple surface  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title("N1/N2 - how close each chunk gets (min over the 8 exec steps)\n"
                 "clearance = surface distance - sphere radius, PRE-margin",
                 fontsize=9.2, color=figstyle.INK)
    ax.legend(fontsize=7.8, loc="upper left", framealpha=0.9)
    side["panels"]["tip_to_apple"] = {"reference_mm": [round(float(v), 4) for v in ta_r],
                                      "refined_mm": [round(float(v), 4) for v in ta_t]}

    # (2)
    ax = axes[0, 1]
    ax.plot(x, norm.max(axis=1), color=figstyle.CATEGORICAL[0], lw=1.8,
            label="|reference - TO| (max over exec window)")
    for k, (lab, c) in enumerate((("x", 3), ("y", 4), ("z", 5))):
        ax.plot(x, diff[:, :, k].mean(axis=1), lw=1.0, ls="-.",
                color=figstyle.CATEGORICAL[c], label=f"{lab} component (mean, signed)")
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    ax.set_xlabel("chunk", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("fingertip position difference  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title(f"N2 - where the fingertip wanted to go vs where it went\n"
                 f"max {n2['max_abs_mm']:.2f} mm (chunk {n2['max_abs_chunk']}) · "
                 f"median of per-chunk max {n2['median_over_chunks_of_max_mm']:.2f} mm",
                 fontsize=9.2, color=figstyle.INK)
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.9)
    side["panels"]["fingertip_diff"] = {
        "max_over_exec_mm": [round(float(v), 4) for v in norm.max(axis=1)],
        "mean_signed_x_mm": [round(float(v), 4) for v in diff[:, :, 0].mean(axis=1)],
        "mean_signed_y_mm": [round(float(v), 4) for v in diff[:, :, 1].mean(axis=1)],
        "mean_signed_z_mm": [round(float(v), 4) for v in diff[:, :, 2].mean(axis=1)]}

    # (3)
    ax = axes[0, 2]
    ax.plot(range(EXEC_H), n2["per_step_mean_mm"], marker="o", ms=5, lw=2.0,
            color=figstyle.CATEGORICAL[0])
    ax.fill_between(range(EXEC_H), 0, n2["per_step_mean_mm"], color=figstyle.CATEGORICAL[0],
                    alpha=0.12)
    for s, v in enumerate(n2["per_step_mean_mm"]):
        ax.annotate(f"{v:.2f}", (s, v), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7.6, color=figstyle.INK_2)
    ax.set_xlabel("step inside the execution window", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("mean |reference - TO| fingertip  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title("N2 - the gap opens along the execution window\n"
                 "mean over all 75 chunks, both fingertip bodies",
                 fontsize=9.2, color=figstyle.INK)
    side["panels"]["per_step_mean_mm"] = n2["per_step_mean_mm"]

    # (4) N4 mm vs deg
    ax = axes[1, 0]
    w = 0.38
    mm = [n4["mean_contribution_mm"][j] for j in joints]
    deg = [n4["degree_vs_mm"]["max_abs_deg_by_joint"][j] for j in joints]
    ax.bar(np.arange(len(joints)) - w / 2, mm, width=w, color=figstyle.CATEGORICAL[0],
           label="fingertip displacement contribution (mean, mm)")
    ax2 = ax.twinx()
    ax2.bar(np.arange(len(joints)) + w / 2, deg, width=w, color=figstyle.CATEGORICAL[3],
            label="joint deviation (max, deg)")
    ax2.set_ylabel("joint deviation  [deg]", fontsize=8.6, color=figstyle.CATEGORICAL[3])
    ax2.tick_params(labelsize=8, colors=figstyle.CATEGORICAL[3])
    ax.set_xticks(range(len(joints)))
    ax.set_xticklabels([j.replace("left_arm_", "L") for j in joints], fontsize=9)
    ax.set_ylabel("fingertip contribution  [mm]", fontsize=8.6, color=figstyle.CATEGORICAL[0])
    ax.set_title(f"N4 - degrees and millimetres rank the joints differently\n"
                 f"largest in mm: {n4['largest_mean_joint'].replace('left_arm_','L')} · "
                 f"largest in deg: L6 (5.80 deg but only "
                 f"{n4['mean_contribution_mm']['left_arm_6']:.2f} mm)",
                 fontsize=9.2, color=figstyle.INK)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.2, loc="upper right", framealpha=0.9)
    side["panels"]["n4_mm_vs_deg"] = {"joints": joints, "mean_contribution_mm": mm,
                                      "max_abs_deg": deg}

    # (5) 점유율
    ax = axes[1, 1]
    share = [n4["total_contribution_share"][j] * 100 for j in joints]
    ax.barh(range(len(joints)), share,
            color=[figstyle.SEQ_BLUE[min(12, int(2 + s / 2.2))] for s in share])
    ax.set_yticks(range(len(joints)))
    ax.set_yticklabels([j.replace("left_arm_", "L") for j in joints], fontsize=9)
    for k, s in enumerate(share):
        ax.annotate(f"{s:.1f} %", (s, k), textcoords="offset points", xytext=(5, -3),
                    fontsize=8.0, color=figstyle.INK_2)
    ax.set_xlabel("share of the total fingertip displacement contribution  [%]",
                  fontsize=8.6, color=figstyle.INK_2)
    ax.set_xlim(0, max(share) * 1.25)
    lc = n4["linearisation_check"]
    ax.set_title(f"N4 - who moved the fingertip\n"
                 f"Jacobian check: predicted {lc['mean_predicted_mm']:.2f} mm vs actual "
                 f"{lc['mean_actual_mm']:.2f} mm (max residual {lc['max_abs_residual_mm']:.2f})",
                 fontsize=9.2, color=figstyle.INK)
    side["panels"]["n4_share_pct"] = dict(zip(joints, [round(s, 4) for s in share]))

    # (6) N3
    ax = axes[1, 2]
    sr = np.asarray(n3["series_reference_mm"], np.float64)
    st = np.asarray(n3["series_refined_mm"], np.float64)
    ax.plot(range(sr.size), sr, color=figstyle.CATEGORICAL[2], lw=2.0,
            label="policy reference, all 50 steps")
    ax.plot(range(st.size), st, color=figstyle.CATEGORICAL[1], lw=1.2, ls="--",
            label="TO output, all 50 steps")
    ax.axvspan(0, EXEC_H - 1, color=figstyle.CATEGORICAL[0], alpha=0.08)
    ax.annotate("execution window\n(the only steps TO changes)", (EXEC_H - 1, sr.max() * 0.8),
                fontsize=7.6, color=figstyle.INK_2)
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    ax.plot([n3["reference_argmin_step"]], [n3["reference_min_mm_over_50_steps"]], marker="*",
            ms=15, color=figstyle.CATEGORICAL[2], zorder=9)
    ax.annotate(f"{n3['reference_min_mm_over_50_steps']:+.2f} mm at step "
                f"{n3['reference_argmin_step']}",
                (n3["reference_argmin_step"], n3["reference_min_mm_over_50_steps"]),
                textcoords="offset points", xytext=(8, -14), fontsize=8.2, color=figstyle.INK)
    ax.set_xlabel("step in the chunk (0..49)", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("left fingertip to apple surface  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title(f"N3 - follow the policy's own chunk {n3['chunk']} to the end\n"
                 f"{n3['reference_below_0mm_steps']} of 50 steps are at or below the apple "
                 f"surface", fontsize=9.2, color=figstyle.INK)
    ax.legend(fontsize=7.6, loc="upper right", framealpha=0.9)
    side["panels"]["n3"] = {"chunk": n3["chunk"], "reference_mm": n3["series_reference_mm"],
                            "refined_mm": n3["series_refined_mm"]}

    fig.suptitle("T9b - policy reference vs TO output: how far the fingertip moved, which joint "
                 "moved it, and where the policy was heading", fontsize=12.6,
                 color=figstyle.INK, y=0.985)
    fig.text(0.5, 0.010,
             "all values come from the recorded qpos plus the two action chunks, solved with "
             "forward kinematics - no rollout · clearances are PRE-margin "
             "(surface distance - sphere radius) · N4 uses mujoco mj_jacBody at the reference pose",
             ha="center", fontsize=8.3, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.024, 1, 0.958))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
