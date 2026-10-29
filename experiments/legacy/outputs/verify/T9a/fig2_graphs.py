"""T9a figure 2 — 한 시간축 위의 손–사과 거리 · 사과 높이 · 손 높이.

위: 손끝(왼쪽)–사과 표면 거리. T7a 의 두 기준선(closed-loop +89.86, shadow -17.96)을 같이.
가운데: 사과 변위 (참값, mm) 와 gripper 명령.
아래: 손 높이(EE_BODY_L z) 와 crate 까지 최소 clearance, crate 위 구간 음영.
오른쪽: chunk 별 max_violation_pair clearance 와 link 분포.
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

T9A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T9a'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t9a/t9a-graphs.png'))
SIDE = OUT.with_suffix(".json")
GRIP_CLOSE = 0.85


def main() -> None:
    c = np.load(T9A / "m1to5_curves.npz", allow_pickle=True)
    ana = json.loads((T9A / "m1to5_analysis.json").read_text())
    z0 = np.load(T9A / "m0_state_tensor.npz", allow_pickle=True)
    t = np.asarray(c["t_step"], np.int64)
    kind = np.asarray([str(k) for k in c["kind"]])
    ctrl = kind == "control"
    dfl = np.asarray(c["finger_left_to_apple_mm"], np.float64)
    dpl = np.asarray(c["palm_left_to_apple_mm"], np.float64)
    disp = np.asarray(c["apple_disp_mm"], np.float64)
    hz = np.asarray(c["hand_z_mm"], np.float64)
    grip = np.asarray(c["grip_cmd"], np.float64)
    cr = np.asarray(c["min_clearance_crate_mm"], np.float64)
    tb = np.asarray(c["min_clearance_table_mm"], np.float64)
    crate_top = float(z0["crate_top_z_m"]) * 1000.0
    m1, m2, m3, m4, m5 = (ana["m1_finger_to_apple"], ana["m2_gripper_close"],
                          ana["m3_apple_ground_truth"], ana["m4_descent_over_crate"],
                          ana["m5_max_violation_pair"])
    t_grip = m2["left"]["t_step"]
    t_over = m4["first_over_crate_t_step"]

    figstyle.use_korean()
    fig, axes = plt.subplots(3, 2, figsize=(17.4, 11.4),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    figstyle.style_axes(fig, axes)
    side: dict = {"t_step": [int(v) for v in t], "panels": {}}

    def marks(ax):
        ax.axvline(t_grip, color=figstyle.CATEGORICAL[7], lw=1.2, ls="--")
        if t_over is not None:
            ax.axvline(t_over, color=figstyle.CATEGORICAL[4], lw=1.2, ls=":")

    # --- 1. 손끝 - 사과 -----------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(t[ctrl], dfl[ctrl], color=figstyle.CATEGORICAL[2], lw=1.7,
            label="left fingertip -> apple surface")
    ax.plot(t[ctrl], dpl[ctrl], color=figstyle.CATEGORICAL[1], lw=1.2, ls="-.",
            label="ee_left palm -> apple surface")
    ax.axhline(m1["comparison_mm"]["prev_closed_loop_T7a"], color=figstyle.CATEGORICAL[0],
               lw=1.3, ls="--")
    ax.annotate(f"previous closed-loop best  +{m1['comparison_mm']['prev_closed_loop_T7a']:.2f} mm",
                (t.max(), m1["comparison_mm"]["prev_closed_loop_T7a"]), ha="right", va="bottom",
                fontsize=7.6, color=figstyle.CATEGORICAL[0])
    ax.axhline(m1["comparison_mm"]["successful_shadow_T7a"], color=figstyle.CATEGORICAL[7],
               lw=1.3, ls="--")
    ax.annotate(f"successful shadow  {m1['comparison_mm']['successful_shadow_T7a']:.2f} mm",
                (t.max(), m1["comparison_mm"]["successful_shadow_T7a"]), ha="right", va="top",
                fontsize=7.6, color=figstyle.CATEGORICAL[7])
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    ax.plot([m1["left_min_t_step"]], [m1["left_min_mm"]], marker="*", ms=15,
            color=figstyle.CATEGORICAL[2], zorder=9)
    ax.annotate(f"min {m1['left_min_mm']:.2f} mm at t={m1['left_min_t_step']}",
                (m1["left_min_t_step"], m1["left_min_mm"]), textcoords="offset points",
                xytext=(8, 10), fontsize=8.0, color=figstyle.CATEGORICAL[2])
    marks(ax)
    ax.set_ylim(-40, 420)
    ax.set_ylabel("distance  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title("M1 - left hand to apple surface (clearance = surface distance - sphere radius)",
                 fontsize=9.6, color=figstyle.INK)
    ax.legend(fontsize=7.4, loc="upper left", framealpha=0.9)
    side["panels"]["m1"] = {"finger_left_mm": [round(float(v), 4) for v in dfl],
                            "palm_left_mm": [round(float(v), 4) for v in dpl],
                            "min_mm": m1["left_min_mm"], "min_t_step": m1["left_min_t_step"]}

    # --- 2. 사과 변위 + gripper -----------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(t[ctrl], disp[ctrl, 2], color=figstyle.CATEGORICAL[7], lw=1.8,
            label="apple z displacement (ground truth)")
    ax.plot(t[ctrl], np.linalg.norm(disp[ctrl], axis=1), color=figstyle.CATEGORICAL[3], lw=1.2,
            ls="-.", label="apple total displacement")
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    ax.set_ylim(-0.6, 0.6)
    ax.set_ylabel("apple displacement  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ax2 = ax.twinx()
    ax2.plot(t[ctrl], grip[ctrl, 0], color=figstyle.CATEGORICAL[0], lw=1.5,
             label="gripper command, left column 7")
    ax2.axhline(GRIP_CLOSE, color=figstyle.CATEGORICAL[0], lw=1.0, ls=":")
    ax2.set_ylabel("gripper command", fontsize=8.6, color=figstyle.CATEGORICAL[0])
    ax2.tick_params(labelsize=8, colors=figstyle.CATEGORICAL[0])
    ax2.set_ylim(0.3, 1.1)
    marks(ax)
    ax.set_title(f"M3 - the apple never moved: max lift {m3['max_lift_mm']:.4f} mm, "
                 f"max total displacement {m3['max_total_displacement_mm']:.4f} mm   |   "
                 f"M2 - gripper first below {GRIP_CLOSE} at t={t_grip}",
                 fontsize=9.6, color=figstyle.INK)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.4, loc="upper left", framealpha=0.9)
    side["panels"]["m3_m2"] = {"apple_dz_mm": [round(float(v), 6) for v in disp[:, 2]],
                               "apple_total_mm": [round(float(v), 6)
                                                  for v in np.linalg.norm(disp, axis=1)],
                               "gripper_left": [None if not np.isfinite(v) else round(float(v), 6)
                                                for v in grip[:, 0]]}

    # --- 3. 손 높이 + crate ---------------------------------------------------------------
    ax = axes[2, 0]
    ax.plot(t[ctrl], hz[ctrl], color=figstyle.CATEGORICAL[0], lw=1.8, label="EE_BODY_L z")
    ax.axhline(crate_top, color=figstyle.INK_2, lw=1.2, ls="--")
    ax.annotate(f"crate rim top z = {crate_top:.1f} mm", (t.min(), crate_top),
                textcoords="offset points", xytext=(4, 4), fontsize=7.6, color=figstyle.INK_2)
    if t_over is not None:
        ax.axvspan(t_over, t.max(), color=figstyle.CATEGORICAL[4], alpha=0.07)
        ax.annotate(f"hand over the crate from t={t_over}\n"
                    f"lowest EE_BODY_L z = {m4['hand_z_mm_min_while_over_crate']:.1f} mm\n"
                    f"= {m4['hand_above_crate_rim_mm_min']:.1f} mm above the rim",
                    (0.985, 0.95), xycoords="axes fraction", ha="right", va="top",
                    fontsize=8.0, color=figstyle.INK,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=figstyle.SURFACE,
                              edgecolor=figstyle.GRID_INK, alpha=0.93))
    ax.set_ylabel("height  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ax3 = ax.twinx()
    ax3.plot(t[ctrl], cr[ctrl], color=figstyle.CATEGORICAL[2], lw=1.3, ls="-.",
             label="min clearance to crate (any sphere)")
    ax3.plot(t[ctrl], tb[ctrl], color=figstyle.CATEGORICAL[5], lw=1.0, ls=":",
             label="min clearance to table (any sphere)")
    ax3.axhline(10.0, color=figstyle.CATEGORICAL[7], lw=1.0, ls="--")
    ax3.set_ylabel("clearance  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ax3.tick_params(labelsize=8, colors=figstyle.INK_2)
    ax3.set_ylim(0, 320)
    marks(ax)
    ax.set_xlabel("t_step (control records, 312)", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title("M4 - hand height and what was closest\n"
                 "dashed horizontal = esdf margin 10 mm; these clearances are PRE-margin so "
                 "this is the comparison that applies", fontsize=9.2, color=figstyle.INK)
    h1, l1 = ax.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h3, l1 + l3, fontsize=7.4, loc="lower left", framealpha=0.9)
    side["panels"]["m4"] = {"hand_z_mm": [round(float(v), 4) for v in hz],
                            "min_clearance_crate_mm": [round(float(v), 4) for v in cr],
                            "min_clearance_table_mm": [round(float(v), 4) for v in tb],
                            "crate_rim_top_mm": round(crate_top, 4)}

    # --- 4. M5 chunk 별 mvp clearance ------------------------------------------------------
    ax = axes[0, 1]
    pc = m5["per_chunk"]
    xs = [r["t_step"] for r in pc]
    ys = [r["clearance_mm"] for r in pc]
    isatt = [str(r["link"] or "").startswith("attached:") for r in pc]
    ax.bar([x for x, a in zip(xs, isatt) if a], [y for y, a in zip(ys, isatt) if a],
           width=5.5, color=figstyle.CATEGORICAL[1], label="attached:* row")
    ax.bar([x for x, a in zip(xs, isatt) if not a], [y for y, a in zip(ys, isatt) if not a],
           width=5.5, color=figstyle.CATEGORICAL[0], label="robot link row")
    ax.axhline(0.0, color=figstyle.INK, lw=1.2)
    ax.annotate("0 = the margin is exactly used up; below 0 would be a violation",
                (t.max(), 0.0), ha="right", va="bottom", fontsize=7.4, color=figstyle.INK_2)
    marks(ax)
    ax.set_xlabel("t_step", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("max_violation_pair clearance, POST-margin  [mm]\n"
                  "d_esdf - r_robot - margin", fontsize=8.4, color=figstyle.INK_2)
    ax.set_title(f"M5 - tightest recorded row per chunk (39 chunks, unsafe 0)\n"
                 f"post-margin min {m5['clearance_mm_min']:.2f} · "
                 f"median {m5['clearance_mm_median']:.2f} · "
                 f"max {m5['clearance_mm_max']:.2f} mm  (margin 10 mm already subtracted)",
                 fontsize=9.4, color=figstyle.INK)
    ax.legend(fontsize=7.4, loc="upper left", framealpha=0.9)
    side["panels"]["m5_per_chunk"] = pc

    # --- 5. M5 link 분포 -------------------------------------------------------------------
    ax = axes[1, 1]
    lc = m5["link_counts"]
    names = list(lc)[::-1]
    vals = [lc[n] for n in names]
    cols = [figstyle.CATEGORICAL[1] if str(n).startswith("attached:") else figstyle.CATEGORICAL[0]
            for n in names]
    ax.barh(range(len(names)), vals, color=cols)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.4)
    ax.set_xlabel("chunks (of 39)", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title("M5 - which link held the tightest row\n"
                 f"ee_left appears {m5['ee_left_in_mvp']}x, ee_right {m5['ee_right_in_mvp']}x",
                 fontsize=9.6, color=figstyle.INK)
    side["panels"]["m5_link_counts"] = lc

    # --- 6. M1 비교 막대 ------------------------------------------------------------------
    ax = axes[2, 1]
    names = ["previous\nclosed-loop", "this run\nexecute_ep1807", "successful\nshadow"]
    vals = [m1["comparison_mm"]["prev_closed_loop_T7a"], m1["left_min_mm"],
            m1["comparison_mm"]["successful_shadow_T7a"]]
    cols = [figstyle.CATEGORICAL[0], figstyle.CATEGORICAL[2], figstyle.CATEGORICAL[7]]
    ax.bar(range(3), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.2f} mm", (i, v), textcoords="offset points",
                    xytext=(0, 5 if v >= 0 else -14), ha="center", fontsize=9.0,
                    color=figstyle.INK)
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=8.2)
    ax.set_ylabel("closest left fingertip to apple surface  [mm]", fontsize=8.6,
                  color=figstyle.INK_2)
    ax.set_title(f"M1 - where this run sits between the two references\n"
                 f"{m1['comparison_mm']['fraction_of_the_gap_closed']*100:.1f} % of the gap "
                 f"closed (sphere radius differs, see sidecar)", fontsize=9.6, color=figstyle.INK)
    side["panels"]["m1_comparison"] = m1["comparison_mm"] | {
        "note_radius_changed": m1["note_radius_changed"],
        "this_run_centre_to_surface_mm": m1["left_min_centre_to_apple_surface_mm"]}

    fig.suptitle("T9a - execute_ep1807: 39 chunks all safe, unsafe 0 - how close the hand got, "
                 "whether the apple moved, and where the descent stopped",
                 fontsize=12.6, color=figstyle.INK, y=0.985)
    fig.text(0.5, 0.010,
             "orange dashed vertical = gripper first commanded below 0.85 · pink dotted vertical "
             "= hand first over the crate · all curves come from the recorded qpos (312 control "
             "rows), not from a rollout\n"
             "the offline clearances (M1, M4) are PRE-margin (surface distance - sphere radius); "
             "the recorded max_violation_pair clearance (M5) is POST-margin "
             "(d_esdf - r - margin, linearize.py:489) - the two are not directly comparable",
             ha="center", fontsize=8.2, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.036, 1, 0.958))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
