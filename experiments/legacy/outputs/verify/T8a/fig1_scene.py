"""T8a figure 1 — 실제 씬. 사과를 빼는 것이 어느 구·어느 순간의 이야기인가.

네 칸은 m0 tensor 의 극값 행 네 개다 (모두 shadow `actions_reference`):
  A  chunk 21 step 7  sphere 119 ee_finger_l2  — 손끝이 사과로 가장 깊이 (-17.45 mm)
  B  chunk 23 step 6  sphere  41 link_left_arm_5 — 팔뚝이 사과로 가장 깊이 (-78.30 mm)
  C  chunk 24 step 17 sphere  41 link_left_arm_5 — 사과를 뺀 뒤 남는 최악 행 (crate -24.78 mm)
  D  chunk  5 step 48 sphere  41 link_left_arm_5 — 사과가 음수인 질의점에서 table 최악 (-15.80 mm)

위 줄: MuJoCo 렌더 (osmesa). 기록 `qpos` 에 그 chunk·그 step 의 action 을 팔 관절에 얹은 자세.
아래 줄: 같은 자세의 base frame XZ 투영. 초점 구는 margin 10 mm 고리를 함께 그린다.
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
from m3_to_refined_vs_reference import (  # noqa: E402
    ACT_COL, JOINT_ORDER, MODEL_XML, QADR, SHADOW, load_planning)

NPZ = _ag3s_workspace_path('outputs/verify/T8a/m0_clearance_tensor.npz')
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8a/t8a-scene.png'))
SIDE = OUT.with_suffix(".json")
MARGIN_MM = 10.0
AUTH = ("ee_finger_l1", "ee_finger_l2", "link_left_arm_5")
PANELS = [
    ("A  fingertip deepest in apple", 21, 7, 119, "apple"),
    ("B  link_left_arm_5 deepest in apple", 23, 6, 41, "apple"),
    ("C  worst row left after removing apple", 24, 17, 41, "crate"),
    ("D  table at an apple-negative query point", 5, 48, 41, "table"),
]
FRUITS = ("apple", "orange", "pear", "banana")
FRUIT_R = {"apple": 0.0333, "orange": 0.0307, "pear": 0.0432, "banana": 0.0226}


def main() -> None:
    import mujoco

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    z = np.load(NPZ, allow_pickle=True)
    clear_mm = z["clearance_m"].astype(np.float64) * 1000.0
    bodies = [str(b) for b in z["bodies"]]

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    scene = TransportScene.attach(m, d)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    names = list(resolve_link_names(robot, robot.n_spheres))
    _, radii = robot.sphere_centers_numeric(np.zeros(robot.nq))
    radii = np.asarray(radii, np.float64).reshape(-1)
    recs = load_planning(SHADOW)

    renderer = mujoco.Renderer(m, height=430, width=520)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.52, 0.18, 0.92]
    cam.distance, cam.azimuth, cam.elevation = 1.05, 128.0, -18.0

    ok = figstyle.use_korean()
    fig, axes = plt.subplots(2, 4, figsize=(19.2, 9.6),
                             gridspec_kw={"height_ratios": [1.0, 1.05]})
    figstyle.style_axes(fig, axes[1])
    side: dict = {"margin_mm": MARGIN_MM, "authorized_links": list(AUTH),
                  "source_tensor": NPZ, "panels": []}

    for col, (label, ch, step, si, focus) in enumerate(PANELS):
        rec = recs[ch]
        chunk = np.asarray(rec["actions_reference"], np.float64)
        qpos = np.asarray(rec["qpos"], np.float64)
        d.qpos[: qpos.size] = qpos
        for n, ci in ACT_COL.items():
            d.qpos[QADR[n]] = float(chunk[step, ci])
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        renderer.update_scene(d, cam)
        ax = axes[0, col]
        ax.imshow(renderer.render())
        ax.set_axis_off()
        ax.set_title(f"{label}\nchunk {ch} (seq {rec['seq']}) · t={rec['t_step']} · plan step {step}"
                     f" · sphere {si} {names[si]}", fontsize=9.5, color=figstyle.INK)

        ax = axes[1, col]
        q = np.asarray([qpos[QADR[n]] if n.startswith("torso_") else float(chunk[step, ACT_COL[n]])
                        for n in JOINT_ORDER], np.float64)
        centres = np.asarray(robot.sphere_centers_numeric(q)[0], np.float64).reshape(-1, 3)
        objs = rec["object_poses"]
        for k, name in enumerate(FRUITS):
            p = np.asarray(objs[name]["pos"], np.float64)
            ax.add_patch(Circle((p[0], p[2]), FRUIT_R[name],
                                facecolor=figstyle.CATEGORICAL[k], edgecolor="none",
                                alpha=0.9 if name == "apple" else 0.6, zorder=4))
            dx, dy, ha = {"apple": (0.0, FRUIT_R[name] + 0.016, "center"),
                          "pear": (0.0, -FRUIT_R[name] - 0.026, "center"),
                          "orange": (-FRUIT_R[name] - 0.012, -0.006, "right"),
                          "banana": (FRUIT_R[name] + 0.010, -0.004, "left")}[name]
            ax.annotate(name + (" (target)" if name == "apple" else ""),
                        (p[0] + dx, p[2] + dy), ha=ha, fontsize=7.5, zorder=9,
                        color=figstyle.CATEGORICAL[k],
                        bbox=dict(boxstyle="round,pad=0.12", facecolor=figstyle.SURFACE,
                                  edgecolor="none", alpha=0.8))
        cr = np.asarray(objs["crate"]["pos"], np.float64)
        ax.add_patch(plt.Rectangle((cr[0] - 0.09, cr[2] - 0.004), 0.18, 0.112, facecolor="none",
                                   edgecolor=figstyle.INK_2, lw=1.2, zorder=3))
        ax.annotate("crate (kept)", (cr[0] + 0.095, cr[2] + 0.055), ha="left", fontsize=7.5,
                    color=figstyle.INK_2, zorder=9)
        tbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
        top_z = max(float(d.geom_xpos[g][2]) + float(m.geom_size[g][2])
                    for g in range(m.ngeom) if int(m.geom_bodyid[g]) == tbid)
        ax.axhline(top_z, color=figstyle.GRID_INK, lw=1.6, zorder=1)
        ax.annotate(f"table top (kept)  z = {top_z:.3f} m", (0.245, top_z + 0.008),
                    fontsize=7.5, color=figstyle.INK_2)

        for i, n in enumerate(names):
            if "left" not in n and not n.startswith("ee_finger_l"):
                continue
            if n in AUTH:
                fc, a, zo = figstyle.CATEGORICAL[1], 0.32, 5
            else:
                fc, a, zo = figstyle.SEQ_BLUE[3], 0.18, 2
            ax.add_patch(Circle((centres[i, 0], centres[i, 2]), radii[i], facecolor=fc,
                                edgecolor="none", alpha=a, zorder=zo))
        # 초점 구와 그 구가 요구하는 자유 반지름 (r + margin 10 mm)
        ax.add_patch(Circle((centres[si, 0], centres[si, 2]), radii[si], facecolor="none",
                            edgecolor=figstyle.CATEGORICAL[1], lw=1.8, zorder=7))
        ax.add_patch(Circle((centres[si, 0], centres[si, 2]), radii[si] + MARGIN_MM / 1000.0,
                            facecolor="none", edgecolor=figstyle.CATEGORICAL[6], lw=1.2, ls="--",
                            zorder=7))
        ax.plot([centres[si, 0]], [centres[si, 2]], marker="x", ms=8, mew=2.0,
                color=figstyle.CATEGORICAL[1], zorder=8)

        row = {b: float(clear_mm[ch, step, si, bi]) for bi, b in enumerate(bodies)}
        ax.annotate(
            f"sphere {si}  {names[si]}  r {radii[si]*1000:.2f} mm\n"
            f"clearance at this query point  [mm]\n"
            f"  apple  {row['apple']:+9.2f}   (removed for this link)\n"
            f"  table  {row['table']:+9.2f}\n"
            f"  crate  {row['crate']:+9.2f}\n"
            f"  orange {row['orange']:+9.2f}\n"
            f"margin {MARGIN_MM:.0f} mm  ->  slack = clearance - 10",
            (0.985, 0.985), xycoords="axes fraction", fontsize=7.6, color=figstyle.INK,
            va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=figstyle.SURFACE,
                      edgecolor=figstyle.GRID_INK, alpha=0.93))
        ax.set_xlim(0.22, 0.90)
        ax.set_ylim(0.75, 1.28)
        ax.set_aspect("equal")
        ax.set_xlabel("base x  [m]", fontsize=8.5, color=figstyle.INK_2)
        if col == 0:
            ax.set_ylabel("base z  [m]", fontsize=8.5, color=figstyle.INK_2)
        side["panels"].append({
            "label": label, "chunk": ch, "seq": int(rec["seq"]), "t_step": int(rec["t_step"]),
            "plan_step": step, "sphere_index": si, "link": names[si],
            "sphere_radius_mm": round(float(radii[si]) * 1000.0, 4),
            "focus_body": focus,
            "clearance_mm_by_body": {k: round(v, 4) for k, v in row.items()},
            "table_top_z_m": round(top_z, 5),
            "trajopt_status": rec["verdict"]["trajopt_status"],
            "recorded_max_violation_pair": rec.get("max_violation_pair"),
        })

    fig.suptitle("T8a - shadow actions_reference: where the apple rows are, and what is left "
                 "once they are removed", fontsize=12.5, color=figstyle.INK, y=0.985)
    fig.text(0.5, 0.012,
             "orange fill = spheres of the authorized links (ee_finger_l1, ee_finger_l2, "
             "link_left_arm_5) whose apple rows are removed · blue fill = other left-arm spheres "
             "(apple rows kept) · solid ring = focus sphere radius · dashed ring = radius + "
             "margin 10 mm · clearance = analytic surface distance - sphere radius, "
             "no mj_geomDistance",
             ha="center", fontsize=8.3, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("korean font:", ok)
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
