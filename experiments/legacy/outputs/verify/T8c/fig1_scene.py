"""T8c figure 1 — 실제 씬. 구를 줄이면 그 행이 어떻게 되는가, 그리고 안 되는 행은 왜 안 되는가.

세 칸은 s3 가 역산한 임계 행 셋 (모두 shadow `actions_reference`, 사과는 authorized 셋에서 제외):
  A chunk 24 step 17 sphere 41  link_left_arm_5 vs crate   중심-표면 56.47 mm -> r_crit 46.47 mm
  B chunk  5 step 48 sphere 108 ee_finger_l1   vs table    중심-표면  8.34 mm -> r_crit -1.66 mm
  C chunk 12 step  5 sphere 108 ee_finger_l1   vs table    중심-표면 17.75 mm -> r_crit  7.75 mm

위 줄: MuJoCo 렌더. 아래 줄: base frame XZ. 각 초점 구를 **원래 반지름**(옅은 채움)과
**r_crit**(진한 테두리) 로 겹쳐 그리고, margin 10 mm 고리를 함께 그린다.
반지름이 바뀌어도 중심은 안 움직인다 (s1 에서 drift 0.0 m 확인).
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

S3 = json.loads(pathlib.Path(
    _ag3s_workspace_path('outputs/verify/T8c/s3_critical_radius.json')).read_text())
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8c/t8c-scene.png'))
SIDE = OUT.with_suffix(".json")
MARGIN_MM = 10.0
PANELS = [
    ("A  link_left_arm_5 vs crate  (worst remaining row)", "full50",
     "link_left_arm_5__r81250.0000mm"),
    ("B  ee_finger_l1 vs table  (full50: r_crit < 0)", "full50",
     "ee_finger_l1__r18698.0760mm"),
    ("C  ee_finger_l1 vs table  (exec8: r_crit 7.75 mm)", "exec8",
     "ee_finger_l1__r18698.0760mm"),
]
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
    recs = load_planning(SHADOW)

    renderer = mujoco.Renderer(m, height=430, width=560)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.52, 0.18, 0.92]
    cam.distance, cam.azimuth, cam.elevation = 1.05, 128.0, -18.0

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.6),
                             gridspec_kw={"height_ratios": [1.0, 1.05]})
    figstyle.style_axes(fig, axes[1])
    side: dict = {"margin_mm": MARGIN_MM, "source": str(S3["definition"]), "panels": []}

    for col, (label, win, gkey) in enumerate(PANELS):
        g = S3["groups"][win][gkey]
        ch, step, si = g["argmin_chunk"], g["argmin_step"], g["argmin_sphere"]
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
        ax.set_title(f"{label}\nchunk {ch} (seq {rec['seq']}) · plan step {step} · sphere {si} "
                     f"{names[si]} · window {win}", fontsize=9.5, color=figstyle.INK)

        ax = axes[1, col]
        q = np.asarray([qpos[QADR[n]] if n.startswith("torso_") else float(chunk[step, ACT_COL[n]])
                        for n in JOINT_ORDER], np.float64)
        centres = np.asarray(robot.sphere_centers_numeric(q)[0], np.float64).reshape(-1, 3)
        objs = rec["object_poses"]
        for k, name in enumerate(FRUITS):
            p = np.asarray(objs[name]["pos"], np.float64)
            ax.add_patch(Circle((p[0], p[2]), FRUIT_R[name], facecolor=figstyle.CATEGORICAL[k],
                                edgecolor="none", alpha=0.85, zorder=4))
        cr = np.asarray(objs["crate"]["pos"], np.float64)
        ax.add_patch(plt.Rectangle((cr[0] - 0.09, cr[2] - 0.004), 0.18, 0.112, facecolor="none",
                                   edgecolor=figstyle.INK_2, lw=1.3, zorder=3))
        ax.annotate("crate", (cr[0] + 0.095, cr[2] + 0.06), fontsize=7.5, color=figstyle.INK_2)
        tbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
        top_z = max(float(d.geom_xpos[g_][2]) + float(m.geom_size[g_][2])
                    for g_ in range(m.ngeom) if int(m.geom_bodyid[g_]) == tbid)
        ax.axhline(top_z, color=figstyle.GRID_INK, lw=1.6, zorder=1)
        ax.annotate(f"table top  z = {top_z:.3f} m", (0.245, top_z + 0.006), fontsize=7.5,
                    color=figstyle.INK_2)

        gi = [i for i, n in enumerate(names) if i in g["sphere_indices"]]
        for i, n in enumerate(names):
            if "left" not in n and not n.startswith("ee_finger_l"):
                continue
            ax.add_patch(Circle((centres[i, 0], centres[i, 2]), radii[i],
                                facecolor=figstyle.SEQ_BLUE[3], edgecolor="none",
                                alpha=0.16, zorder=2))
        r_old = float(radii[si])
        r_new = max(g["r_crit_mm"], 0.0) / 1000.0
        for i in gi:
            ax.add_patch(Circle((centres[i, 0], centres[i, 2]), radii[i],
                                facecolor=figstyle.CATEGORICAL[1], edgecolor="none",
                                alpha=0.22, zorder=4))
            ax.add_patch(Circle((centres[i, 0], centres[i, 2]), r_new, facecolor="none",
                                edgecolor=figstyle.CATEGORICAL[0], lw=1.3, zorder=6))
        ax.add_patch(Circle((centres[si, 0], centres[si, 2]), r_old, facecolor="none",
                            edgecolor=figstyle.CATEGORICAL[1], lw=1.8, zorder=7))
        ax.add_patch(Circle((centres[si, 0], centres[si, 2]), r_new + MARGIN_MM / 1000.0,
                            facecolor="none", edgecolor=figstyle.CATEGORICAL[7], lw=1.2,
                            ls="--", zorder=7))
        ax.plot([centres[si, 0]], [centres[si, 2]], marker="x", ms=9, mew=2.2,
                color=figstyle.CATEGORICAL[1], zorder=8)

        # --- 확대 inset: 초점 구 주변 ---------------------------------------------------
        half = 0.085
        cx, cz = float(centres[si, 0]), float(centres[si, 2])
        iax = ax.inset_axes([0.03, 0.04, 0.40, 0.40], zorder=12)
        iax.set_facecolor(figstyle.SURFACE)
        iax.patch.set_alpha(1.0)
        for k, name in enumerate(FRUITS):
            pp = np.asarray(objs[name]["pos"], np.float64)
            iax.add_patch(Circle((pp[0], pp[2]), FRUIT_R[name],
                                 facecolor=figstyle.CATEGORICAL[k], edgecolor="none",
                                 alpha=0.85, zorder=4))
        iax.add_patch(plt.Rectangle((cr[0] - 0.09, cr[2] - 0.004), 0.18, 0.112,
                                    facecolor="none", edgecolor=figstyle.INK_2, lw=1.3, zorder=3))
        iax.axhline(top_z, color=figstyle.GRID_INK, lw=1.6, zorder=1)
        for i, n in enumerate(names):
            if "left" not in n and not n.startswith("ee_finger_l"):
                continue
            iax.add_patch(Circle((centres[i, 0], centres[i, 2]), radii[i],
                                 facecolor=figstyle.SEQ_BLUE[3], edgecolor="none",
                                 alpha=0.16, zorder=2))
        for i in gi:
            iax.add_patch(Circle((centres[i, 0], centres[i, 2]), radii[i],
                                 facecolor=figstyle.CATEGORICAL[1], edgecolor="none",
                                 alpha=0.22, zorder=4))
            iax.add_patch(Circle((centres[i, 0], centres[i, 2]), r_new, facecolor="none",
                                 edgecolor=figstyle.CATEGORICAL[0], lw=1.3, zorder=6))
        iax.add_patch(Circle((cx, cz), r_old, facecolor="none",
                             edgecolor=figstyle.CATEGORICAL[1], lw=1.8, zorder=7))
        iax.add_patch(Circle((cx, cz), r_new + MARGIN_MM / 1000.0, facecolor="none",
                             edgecolor=figstyle.CATEGORICAL[7], lw=1.2, ls="--", zorder=7))
        iax.plot([cx], [cz], marker="x", ms=7, mew=2.0, color=figstyle.CATEGORICAL[1], zorder=8)
        iax.set_xlim(cx - half, cx + half)
        iax.set_ylim(cz - half, cz + half)
        iax.set_aspect("equal")
        iax.tick_params(labelsize=6.0, colors=figstyle.INK_2, length=2)
        for sp in iax.spines.values():
            sp.set_color(figstyle.GRID_INK)
        iax.annotate("zoom x170 mm", (0.03, 0.95), xycoords="axes fraction", fontsize=6.6,
                     color=figstyle.INK_2, va="top")
        ax.indicate_inset_zoom(iax, edgecolor=figstyle.GRID_INK)

        sc = g["scale_crit"]
        ax.annotate(
            f"sphere {si}  {names[si]}\n"
            f"centre to {g['argmin_body']} surface   {g['min_surface_distance_mm']:8.3f} mm\n"
            f"margin                        -{MARGIN_MM:7.3f} mm\n"
            f"= r_crit (largest feasible r) {g['r_crit_mm']:8.3f} mm\n"
            f"r at scale 1.0                {g['r_at_scale_1_mm']:8.3f} mm\n"
            f"sphere-chain floor            {g['floor_mm']:8.3f} mm\n"
            f"scale_crit  " + ("undefined - r_crit <= 0" if sc is None else f"{sc:.6f}"),
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
            "label": label, "window": win, "group": gkey, "chunk": ch, "seq": int(rec["seq"]),
            "plan_step": step, "sphere_index": si, "link": names[si],
            "table_top_z_m": round(top_z, 5),
            "r_at_scale_1_mm": g["r_at_scale_1_mm"], "r_crit_mm": g["r_crit_mm"],
            "scale_crit": g["scale_crit"], "floor_mm": g["floor_mm"],
            "min_surface_distance_mm": g["min_surface_distance_mm"],
            "argmin_body": g["argmin_body"]})

    fig.suptitle("T8c - shrinking the constraint spheres: the exact radius each capsule group "
                 "would need, at the row that sets it", fontsize=12.5, color=figstyle.INK, y=0.985)
    fig.text(0.5, 0.012,
             "pale orange = the capsule group's spheres at scale 1.0 · blue outline = the same "
             "spheres at r_crit · dashed ring = r_crit + margin 10 mm · pale blue = other "
             "left-arm spheres · sphere centres do not move with the scale (verified drift 0.0 m)",
             ha="center", fontsize=8.3, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.962))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
