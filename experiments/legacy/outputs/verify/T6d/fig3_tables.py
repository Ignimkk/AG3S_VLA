"""T6d figure 3 — 표·도식.

왼쪽 위: **재는 것 1** — `authorized`(ClearancePolicy) 와 `links`(builder.sphere_link_names) 두 집합.
왼쪽 아래: **재는 것 2** — HOLD 가 난 frame 의 질의점에서 참값까지.
오른쪽 위: **재는 것 3** — shadow 의 refined 대 reference.
오른쪽 아래: 세 단계 도식에 측정값을 붙인 것.
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
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t6d/t6d-table.png'))
SIDE = OUT.with_suffix(".json")


def table(ax, title, cols, rows, widths, fontsize=8.2, head=9.0):
    ax.set_axis_off()
    ax.set_title(title, fontsize=10.5, color=figstyle.INK, loc="left", pad=10)
    x0 = np.concatenate([[0.0], np.cumsum(widths)])
    y = 0.94
    dy = min(0.072, 0.88 / max(len(rows) + 1, 1))
    for j, c in enumerate(cols):
        ax.text(x0[j], y, c, fontsize=head, color=figstyle.INK, va="top", weight="normal")
    ax.plot([0, 1], [y - dy * 0.42] * 2, color=figstyle.GRID_INK, lw=1.0)
    for i, row in enumerate(rows):
        yy = y - dy * (i + 1)
        for j, cell in enumerate(row):
            colr = figstyle.INK if j == 0 else figstyle.INK_2
            ax.text(x0[j], yy, str(cell), fontsize=fontsize, color=colr, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def main() -> None:
    m1 = json.loads((V / "m1_authorized_vs_spheres.json").read_text())
    m1b = json.loads((V / "m1b_query_index_identity.json").read_text())
    gt = json.loads((V / "m2c_gt_unsaturated.json").read_text())
    m3 = json.loads((V / "m3_to_refined_vs_reference.json").read_text())
    m3c = json.loads((V / "m3c_clearance_split.json").read_text())
    m3d = json.loads((V / "m3d_divergence.json").read_text())

    figstyle.use_korean()
    fig, ax = plt.subplots(2, 2, figsize=(16.0, 10.2))
    fig.patch.set_facecolor(figstyle.SURFACE)
    for a in ax.ravel():
        a.set_facecolor(figstyle.SURFACE)
    side: dict = {}

    # ---------------- 1. 두 집합 -------------------------------------------------------
    arms = m1["constraint_models"]["arms (ARM_LINKS)"]
    allm = m1["constraint_models"]["all (link_filter=None)"]
    rows = []
    for phase in ("transit", "approach", "pre_grasp", "grasp"):
        k = f"{phase}|left"
        e = arms["contexts"][k]
        f = allm["contexts"][k]
        rows.append([phase,
                     ", ".join(e["authorized_links"]),
                     f"{e['n_overlap']} / {e['n_authorized_links']}  ({e['n_spheres_authorized']} spheres)",
                     f"{e['target_margin_unique_mm']}",
                     f"{f['n_overlap']} / {f['n_authorized_links']}  ({f['n_spheres_authorized']})"])
    table(ax[0, 0],
          "1. AG3S -> cuRobo — is_authorized is exact string match (clearance.py:280)",
          ["phase (left)", "authorized_links", "overlap (arms, 120)",
           "TARGET margin", "overlap (all, 218)"],
          rows, [0.115, 0.315, 0.235, 0.175, 0.16])
    ax[0, 0].text(0.0, 0.42,
                  "sphere_link_names (arms model, 120 spheres) carries:\n  "
                  + ", ".join(arms["sphere_link_names_unique"][:9]) + ",\n  "
                  + ", ".join(arms["sphere_link_names_unique"][9:])
                  + f"\n\nee_finger_l1 / l2: 11 spheres each   ·   ee_left: 0 spheres in this model "
                    f"(12 in the all model)\nrecorded (query, link) pairs match the arms model "
                    f"({m1b['which_model_matches']['arms']['consistent']}) and not the all model "
                    f"({m1b['which_model_matches']['all']['consistent']})",
                  fontsize=8.0, color=figstyle.INK_2, va="top")
    side["overlap"] = {p: arms["contexts"][f"{p}|left"] for p in
                       ("transit", "approach", "pre_grasp", "grasp")}

    # ---------------- 2. 위반 좌표에 무엇이 있나 ---------------------------------------
    rows = []
    for run, rs in gt["runs"].items():
        for r in rs:
            if not r["hold_reason"] and r["seq"] != 1:
                continue
            rows.append([f"{run.replace('closed_loop_', 'cl-').replace('shadow_', 'sh-')} s{r['seq']}",
                         r["link"],
                         f"{r['clearance_mm']:+.2f}",
                         f"{r['sphere_radius_mm']:.2f} + 50 = {r['required_free_radius_mm']:.2f}",
                         f"{r['implied_esdf_distance_mm']:.1f}",
                         f"{r['gt_scene_min_mm']:.1f} ({r['gt_scene_min_body']})",
                         f"{r['gt_robot_min_mm']:.1f} ({r['gt_robot_min_body']})"])
    table(ax[1, 0],
          "2. cuRobo — at the worst-clearance query point, measured against ground truth",
          ["run / seq", "link", "clearance", "needs r+margin", "ESDF d",
           "nearest GT scene", "nearest GT robot"],
          rows, [0.15, 0.145, 0.095, 0.165, 0.095, 0.185, 0.165], fontsize=7.4, head=8.0)
    ax[1, 0].text(0.0, 0.10,
                  "esdf.unknown_policy = 'free' (config.py:443, serve_safe passes no override) · "
                  "esdf.time_decay = 1.0 and frustum_decay = 1.0 (decay off)\n"
                  "field tiers in every record: coarse 20 mm 128^3 + fine 5 mm 128^3 · "
                  "the field itself is not stored in the recordings, so ESDF d is read back from "
                  "clearance + sphere radius + margin",
                  fontsize=7.6, color=figstyle.INK_2, va="top")
    side["hold_rows"] = rows

    # ---------------- 3. TO ------------------------------------------------------------
    jd = m3["joint_delta"]
    top = sorted(jd["per_column"], key=lambda r: -r["max_abs_deg"])[:5]
    rows = [[r["joint"], f"{r['max_abs_deg']:.3f}", f"{r['rms_deg']:.4f}"] for r in top]
    table(ax[0, 1], "3. TO — refined minus reference (shadow run, 75 chunks x 50 steps)",
          ["column", "max |delta| [deg]", "rms [deg]"], rows, [0.30, 0.22, 0.20])
    e8 = m3c["shadow_no_apple_exec8"]
    f50 = m3c["shadow_no_apple_full50"]
    ax[0, 1].text(0.0, 0.50,
                  "ground-truth clearance, apple excluded (constraint spheres vs ground-truth geoms)\n"
                  f"  executed window (steps 0-7): refined below reference in "
                  f"{e8['n_refined_worse']} / 75 chunks, above in {e8['n_refined_better']}\n"
                  f"      global min {e8['reference_global_min_mm']:+.2f} mm (reference) -> "
                  f"{e8['refined_global_min_mm']:+.2f} mm (refined), best gain "
                  f"{e8['best_improvement_mm']:+.2f} mm\n"
                  f"  full 50-step plan: refined below reference in {f50['n_refined_worse']} / 75, "
                  f"global min {f50['reference_global_min_mm']:+.2f} -> "
                  f"{f50['refined_global_min_mm']:+.2f} mm\n\n"
                  "left fingertip to apple surface, executed window\n"
                  f"  shadow reference {m3d['executed_window_best_tip_mm']['shadow_reference']:+.2f} mm · "
                  f"shadow refined {m3d['executed_window_best_tip_mm']['shadow_refined']:+.2f} mm\n"
                  f"  closed-loop approach {m3d['executed_window_best_tip_mm']['closed_loop_approach']:+.2f} mm · "
                  f"closed-loop grasp {m3d['executed_window_best_tip_mm']['closed_loop_grasp']:+.2f} mm\n"
                  f"  refined further from the apple than reference in "
                  f"{m3d['whole_run']['n_refined_further_from_apple']} / 75 chunks "
                  f"(max {m3d['whole_run']['max_push_mm']:+.2f} mm); "
                  f"{m3d['pre_touch']['n_refined_further_from_apple']} of those in chunks 1-10\n"
                  f"  closed-loop vs shadow reference: first chunk with a gap > 50 mm = "
                  f"{m3d['closed_loop_approach_vs_shadow_reference']['first_chunk_diff_gt_50mm']}",
                  fontsize=8.0, color=figstyle.INK_2, va="top")
    side["to"] = {"top_joints": top, "exec8": e8, "full50": f50, "divergence": m3d}

    # ---------------- 도식 -------------------------------------------------------------
    a = ax[1, 1]
    a.set_axis_off()
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_title("the three stages, with what T6d measured on each",
                fontsize=10.5, color=figstyle.INK, loc="left", pad=10)
    boxes = [
        (0.68, "1  AG3S -> cuRobo", figstyle.CATEGORICAL[0],
         "ClearancePolicy.margin_matrix -> manipulated_link_margin\n"
         "authorized(left) = ee_left, ee_finger_l1, ee_finger_l2\n"
         "sphere_link_names(arms) has 2 of those 3 -> 22 of 120 spheres\n"
         "ee_left contributes 0 spheres in the arms model\n"
         "TARGET margin on those 22: 50 -> 20 mm (approach), 0 mm (grasp)\n"
         "worst pair in all 225 frames is link_*_arm_5 / _6, never authorized"),
        (0.35, "2  cuRobo (field)", figstyle.CATEGORICAL[1],
         "link_*_arm_5 sphere r = 81.25 mm, esdf margin 50 mm\n"
         "  -> the row demands 131.25 mm of free space at the centre\n"
         "at the one grasp HOLD: implied ESDF d = 106.07 mm,\n"
         "  nearest ground-truth scene surface 122.61 mm (apple),\n"
         "  nearest ground-truth robot surface 56.66 mm (ee_finger_l1)\n"
         "unknown_policy = free, decay off, tiers 20 mm + 5 mm"),
        (0.02, "3  TO (SQP refine)", figstyle.CATEGORICAL[2],
         "refined never below reference on ground-truth clearance\n"
         "  (apple excluded): 0 of 75 chunks, executed window\n"
         "  global min -24.61 -> +25.61 mm\n"
         "largest per-joint change: left_arm_0, max 21.289 deg\n"
         "executed window is 8 of each 50-step chunk (step_in_chunk 0..7)\n"
         "fingertip to apple there: reference -17.96, refined +89.86 / +85.76 mm"),
    ]
    for y0, title, colr, body in boxes:
        a.add_patch(Rectangle((0.01, y0), 0.98, 0.29, facecolor=colr, alpha=0.08,
                              edgecolor=colr, lw=1.2))
        a.text(0.03, y0 + 0.245, title, fontsize=9.6, color=colr)
        a.text(0.03, y0 + 0.205, body, fontsize=7.8, color=figstyle.INK_2, va="top")
    for y0 in (0.31, 0.64):
        a.add_patch(FancyArrowPatch((0.5, y0 + 0.028), (0.5, y0 + 0.002),
                                    arrowstyle="-|>", mutation_scale=11,
                                    color=figstyle.INK_2, lw=1.0))

    fig.suptitle("T6d — the three stages side by side, numbers only",
                 fontsize=12.5, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
