"""T7a figure 1 — 실제 씬. 성공한 shadow 와 실패한 closed-loop 를 **같은 chunk** 에서.

고른 chunk: 9 (손끝이 사과 표면을 처음 파고드는 chunk, `s3`) 와 24 (손끝 간격이 가장 크게
벌어지는 chunk, 134.7 mm).

위 줄: MuJoCo 렌더 (osmesa). 기록의 `qpos` 에 **그 chunk 의 실행된 action 행**(rank 1 행의 step)
을 팔 관절에 얹은 자세.
아래 줄: 같은 자세를 base frame XZ 로 투영 —
  · 참값 물체와 crate·table
  · 왼팔 제약 구 (link_left_arm_5 · 손끝 · 나머지)
  · rank 1..3 활성 행의 구와 그것이 요구하는 자유 반지름 (r + required_margin 50 mm)
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T6d'))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from m3_to_refined_vs_reference import ACT_COL, JOINT_ORDER, MODEL_XML, QADR  # noqa: E402

S1 = json.load(open(_ag3s_workspace_path('outputs/verify/T7a/s1_active_rows.json')))
S3 = json.load(open(_ag3s_workspace_path('outputs/verify/T7a/s3_divergence_context.json')))
PANELS = [
    ("shadow / executed reference", "shadow_approach", "reference", 9),
    ("closed-loop / executed refined", "closed_loop_approach", "refined", 9),
    ("shadow / executed reference", "shadow_approach", "reference", 24),
    ("closed-loop / executed refined", "closed_loop_approach", "refined", 24),
]
SEL = "no_target__left_spheres"
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t7a/t7a-scene.png'))
SIDE = OUT.with_suffix(".json")
MARGIN = 0.050
FRUITS = ("apple", "orange", "pear", "banana")
FRUIT_R = {"apple": 0.0333, "orange": 0.0307, "pear": 0.0432, "banana": 0.0226}


def main() -> None:
    import mujoco

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    scene = TransportScene.attach(m, d)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    names = list(resolve_link_names(robot, robot.n_spheres))
    _, radii = robot.sphere_centers_numeric(np.zeros(robot.nq))
    radii = np.asarray(radii, np.float64).reshape(-1)

    raw = {r: [json.loads(l) for l in open(S1["runs"][r]["path"])
               if json.loads(l).get("kind") == "planning"] for r in {p[1] for p in PANELS}}

    renderer = mujoco.Renderer(m, height=430, width=520)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.52, 0.18, 0.92]
    cam.distance, cam.azimuth, cam.elevation = 1.05, 128.0, -18.0

    ok = figstyle.use_korean()
    fig, axes = plt.subplots(2, 4, figsize=(19.2, 9.6),
                             gridspec_kw={"height_ratios": [1.0, 1.05]})
    figstyle.style_axes(fig, axes[1])
    side: dict = {"required_margin_mm": MARGIN * 1000.0, "selection": SEL,
                  "chunk_choice_reason_numbers": {
                      "first_chunk_shadow_tip_below_0mm": S3["first_chunk_shadow_tip_below_0mm"],
                      "chunk_of_max_gap_tip_mm": S3["chunk_of_max_gap_tip_mm"],
                      "max_gap_tip_mm": S3["max_gap_tip_mm"]},
                  "panels": []}

    for col, (label, run, variant, ch) in enumerate(PANELS):
        rec = raw[run][ch]
        s = S1["runs"][run]["chunks"][ch]["variants"][variant][SEL]
        top = s["top_rows_exec8"]
        step = int(top[0]["step"])
        chunk = np.asarray(rec["actions_reference"] if variant == "reference" else rec["actions"],
                           np.float64)
        qpos = np.asarray(rec["qpos"], np.float64)
        d.qpos[: qpos.size] = qpos
        for n, col_i in ACT_COL.items():
            d.qpos[QADR[n]] = float(chunk[step, col_i])
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        renderer.update_scene(d, cam)
        ax = axes[0, col]
        ax.imshow(renderer.render())
        ax.set_axis_off()
        ax.set_title(f"{label}\nchunk {ch} (seq {rec['seq']}) · t={rec['t_step']} · plan step {step}"
                     f" · {rec['verdict']['trajopt_status']}",
                     fontsize=9.5, color=figstyle.INK)

        ax = axes[1, col]
        q = np.asarray([qpos[QADR[n]] if n.startswith("torso_") else float(chunk[step, ACT_COL[n]])
                        for n in JOINT_ORDER], np.float64)
        centres = np.asarray(robot.sphere_centers_numeric(q)[0], np.float64).reshape(-1, 3)
        objs = rec["object_poses"]
        for k, name in enumerate(FRUITS):
            p = np.asarray(objs[name]["pos"], np.float64)
            ax.add_patch(Circle((p[0], p[2]), FRUIT_R[name], facecolor=figstyle.CATEGORICAL[k],
                                edgecolor="none", alpha=0.85, zorder=4))
            ax.annotate(name, (p[0], p[2] + FRUIT_R[name] + 0.012), ha="center", fontsize=7.5,
                        color=figstyle.INK_2)
        cr = np.asarray(objs["crate"]["pos"], np.float64)
        ax.add_patch(plt.Rectangle((cr[0] - 0.09, cr[2] - 0.004), 0.18, 0.112, facecolor="none",
                                   edgecolor=figstyle.INK_2, lw=1.0, zorder=3))
        ax.annotate("crate", (cr[0], cr[2] + 0.12), ha="center", fontsize=7.5, color=figstyle.INK_2)
        tb = np.asarray(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")], np.float64)
        ax.axhline(tb[2] + 0.02, color=figstyle.GRID_INK, lw=1.4, zorder=1)

        for i, n in enumerate(names):
            if "left" not in n and not n.startswith("ee_finger_l"):
                continue
            if n == "link_left_arm_5":
                fc, a, z = figstyle.CATEGORICAL[7], 0.30, 5
            elif n.startswith("ee_finger_l"):
                fc, a, z = figstyle.CATEGORICAL[2], 0.85, 6
            else:
                fc, a, z = figstyle.SEQ_BLUE[3], 0.20, 2
            ax.add_patch(Circle((centres[i, 0], centres[i, 2]), radii[i], facecolor=fc,
                                edgecolor="none", alpha=a, zorder=z))

        for rank, row in enumerate(top[:3]):
            if int(row["step"]) != step:
                continue
            si = int(row["sphere_index"])
            need = float(radii[si]) + MARGIN
            ax.add_patch(Circle((centres[si, 0], centres[si, 2]), need, facecolor="none",
                                edgecolor=figstyle.CATEGORICAL[1 if rank == 0 else 6],
                                lw=1.6 if rank == 0 else 1.0,
                                ls="--" if rank == 0 else ":", zorder=7))
            ax.plot([centres[si, 0]], [centres[si, 2]], marker="x", ms=8, mew=2.0,
                    color=figstyle.CATEGORICAL[1 if rank == 0 else 6], zorder=8)

        t0 = top[0]
        ax.annotate(
            f"rank 1 active row  (step {t0['step']})\n"
            f"{t0['link']}  sphere {t0['sphere_index']}  r {t0['sphere_radius_mm']:.2f} mm\n"
            f"vs {t0['obstacle_body']}  geom {t0['geom_id']}\n"
            f"clearance {t0['clearance_mm']:+.2f}  -  margin 50.00\n"
            f"= slack {t0['margin_slack_mm']:+.2f} mm\n"
            f"joints {', '.join(t0['arm_joints'])}",
            (0.02, 0.98), xycoords="axes fraction", fontsize=7.8, color=figstyle.INK, va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=figstyle.SURFACE,
                      edgecolor=figstyle.GRID_INK, alpha=0.93))
        tip = (S3["shadow_reference_tip_mm"] if variant == "reference"
               else S3["closed_loop_refined_tip_mm"])[ch]
        ax.annotate(f"fingertip to apple surface {tip:+.1f} mm", (0.98, 0.03),
                    xycoords="axes fraction", ha="right", fontsize=8, color=figstyle.CATEGORICAL[2])
        ax.set_xlim(0.22, 0.90)
        ax.set_ylim(0.78, 1.28)
        ax.set_aspect("equal")
        ax.set_xlabel("base x  [m]", fontsize=8.5, color=figstyle.INK_2)
        if col == 0:
            ax.set_ylabel("base z  [m]", fontsize=8.5, color=figstyle.INK_2)
        side["panels"].append({
            "label": label, "run": run, "variant": variant, "chunk": ch, "seq": int(rec["seq"]),
            "t_step": int(rec["t_step"]), "render_plan_step": step,
            "trajopt_status": rec["verdict"]["trajopt_status"],
            "recorded_max_violation_m": rec["verdict"].get("max_violation_m"),
            "recorded_max_violation_pair": rec.get("max_violation_pair"),
            "min_margin_slack_mm_exec8": s["min_margin_slack_mm_exec8"],
            "top_rows_exec8": top,
            "fingertip_to_apple_mm": tip,
        })

    fig.suptitle("T7a — active collision rows on the left arm: successful shadow vs failed "
                 "closed-loop at the same chunk", fontsize=12.5, color=figstyle.INK, y=0.985)
    fig.text(0.5, 0.012,
             "green = left fingertip spheres · pale orange = link_left_arm_5 spheres (r 81.25 mm) · "
             "blue = other left-arm spheres · dashed circle = free radius the rank-1 row demands "
             "(sphere r + required_margin 50 mm) · rows measured against ground-truth scene "
             "surfaces, apple excluded (authorized target)",
             ha="center", fontsize=8.3, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("korean font:", ok)
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
