"""T10a figure 1 — 실제 씬. 손가락과 **그 최근접 body** 를 함께.

왼쪽: t=96 — 기록의 가장 빡빡한 행이 있던 자리 (ESDF 는 13.45 mm 라고 했다).
오른쪽: t=192 — 사과를 뺀 뒤 gripper 가 어떤 body 에 가장 가까웠던 자리 (crate).
위 줄 MuJoCo 렌더, 아래 줄 base frame XZ. 각 gripper 구에서 **가장 가까운 body 표면까지**
선을 긋고 mm 를 적는다. 괄호는 사과를 뺐을 때의 최근접이다.
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
from m3_to_refined_vs_reference import ACT_COL, MODEL_XML, QADR  # noqa: E402

REC = _ag3s_workspace_path('outputs/live_test/20260928_t10/execute_ep1807/frames.jsonl')
T10A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T10a'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t10a/t10a-scene.png'))
SIDE = OUT.with_suffix(".json")
MARGIN_MM = 10.0
APPLE_R = 0.0333
GRIPPER_L = ("ee_finger_l1", "ee_finger_l2", "ee_left")
PANELS = [(96, "A  t=96 — the tightest recorded row of the approach"),
          (192, "B  t=192 — closest gripper sphere to any non-apple body")]


def main() -> None:
    import mujoco

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    ana = json.loads((T10A / "p1to4.json").read_text())
    z = np.load(T10A / "p1to4_curves.npz", allow_pickle=True)
    T = z["T10__tensor"]
    tt = np.asarray(z["T10__t"], np.int64)
    bodies = [str(b) for b in z["bodies"]]
    link_names = [str(n) for n in z["link_names"]]
    radii = np.asarray(z["radii_m"], np.float64)
    grip = np.asarray([n in GRIPPER_L for n in link_names], bool)
    ai = bodies.index("apple")

    recs = [json.loads(l) for l in open(REC)]
    P = [r for r in recs if r["kind"] == "planning"]
    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    scene = TransportScene.attach(m, d)
    robot = build_constraint_robot_model(
        scene, link_filter=ARM_LINKS, sphere_options=ana["method"]["sphere_options"])
    names2 = list(resolve_link_names(robot, robot.n_spheres))
    assert names2 == link_names

    renderer = mujoco.Renderer(m, height=470, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.6),
                             gridspec_kw={"height_ratios": [1.0, 1.12]})
    figstyle.style_axes(fig, axes[1])
    side: dict = {"margin_mm": MARGIN_MM, "panels": []}
    rows_named = {x["t_step"]: x for x in
                  ana["p1_what_blocks_the_fingers"]["T10"]
                  ["recorded_worst_row_named_offline"]["rows"]}

    for col, (tstep, label) in enumerate(PANELS):
        i = int(np.nonzero(tt == tstep)[0][0])
        rec = P[i]
        act = np.asarray(rec["actions"], np.float64)
        qpos = np.asarray(rec["qpos"], np.float64)
        # chunk 안에서 gripper 가 가장 가까웠던 step 을 고른다 (사과 제외 기준)
        sub = T[i, 1][:, grip, :]
        sel_na = np.arange(len(bodies)) != ai
        st = int(np.unravel_index(int(np.nanargmin(sub[:, :, sel_na])), sub[:, :, sel_na].shape)[0])
        d.qpos[: qpos.size] = qpos
        for n, c in ACT_COL.items():
            d.qpos[QADR[n]] = float(act[st, c])
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        apple = np.asarray(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")],
                           np.float64)
        crate = np.asarray(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")],
                           np.float64)
        cbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")
        crate_top = max(float(d.geom_xpos[g][2]) + float(m.geom_size[g][2])
                        for g in range(m.ngeom) if int(m.geom_bodyid[g]) == cbid)
        tbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
        table_top = max(float(d.geom_xpos[g][2]) + float(m.geom_size[g][2])
                        for g in range(m.ngeom) if int(m.geom_bodyid[g]) == tbid)
        centres = np.asarray(robot.sphere_centers_numeric(np.asarray(
            [qpos[QADR[n]] if n.startswith("torso_") else float(act[st, ACT_COL[n]])
             for n in __import__("m3_to_refined_vs_reference").JOINT_ORDER], np.float64))[0],
            np.float64).reshape(-1, 3)

        cam.lookat[:] = [float(apple[0]), float(apple[1]), float(apple[2]) + 0.06]
        cam.distance, cam.azimuth, cam.elevation = 0.75, 128.0, -16.0
        renderer.update_scene(d, cam)
        ax = axes[0, col]
        ax.imshow(renderer.render())
        ax.set_axis_off()
        mvp = rec.get("max_violation_pair") or {}
        ax.set_title(f"{label}\nchunk seq {rec['seq']} · exec step {st} · safe={rec.get('safe')} "
                     f"· recorded worst row: {mvp.get('link')} "
                     f"{(mvp.get('clearance_m') or 0)*1000:.2f} mm post-margin, "
                     f"obstacle={mvp.get('obstacle')}", fontsize=9.2, color=figstyle.INK)

        ax = axes[1, col]
        ax.add_patch(Circle((apple[0], apple[2]), APPLE_R, facecolor=figstyle.CATEGORICAL[7],
                            edgecolor="none", alpha=0.9, zorder=4))
        ax.annotate("apple (carved out for the gripper links)",
                    (apple[0], apple[2] + APPLE_R + 0.014), ha="center", fontsize=7.6,
                    color=figstyle.CATEGORICAL[7], zorder=9)
        ax.add_patch(plt.Rectangle((crate[0] - 0.09, crate[2] - 0.004), 0.18,
                                   crate_top - crate[2] + 0.004, facecolor="none",
                                   edgecolor=figstyle.CATEGORICAL[0], lw=1.5, zorder=3))
        ax.annotate("crate", (crate[0] + 0.095, crate_top), fontsize=7.8,
                    color=figstyle.CATEGORICAL[0], zorder=9)
        ax.axhline(table_top, color=figstyle.GRID_INK, lw=1.6, zorder=1)
        ax.annotate(f"table top z={table_top*1000:.0f} mm", (0.30, table_top + 0.006),
                    fontsize=7.6, color=figstyle.INK_2)
        for k, n in enumerate(link_names):
            if n in ("ee_finger_l1", "ee_finger_l2"):
                fc, al, zo = figstyle.CATEGORICAL[2], 0.9, 7
            elif n == "ee_left":
                fc, al, zo = figstyle.CATEGORICAL[1], 0.6, 6
            elif n.startswith("link_left"):
                fc, al, zo = figstyle.SEQ_BLUE[3], 0.18, 2
            else:
                continue
            ax.add_patch(Circle((centres[k, 0], centres[k, 2]), radii[k], facecolor=fc,
                                edgecolor="none", alpha=al, zorder=zo))
        # 최근접 구와 그 body
        row = T[i, 1, st][grip]
        gi = np.nonzero(grip)[0]
        k_all = np.unravel_index(int(np.nanargmin(row)), row.shape)
        rna = row.copy()
        rna[:, ai] = np.inf
        k_na = np.unravel_index(int(np.nanargmin(rna)), rna.shape)
        s_all, b_all = int(gi[k_all[0]]), bodies[int(k_all[1])]
        s_na, b_na = int(gi[k_na[0]]), bodies[int(k_na[1])]
        for sidx, bname, colr in ((s_all, b_all, figstyle.CATEGORICAL[7]),
                                  (s_na, b_na, figstyle.CATEGORICAL[0])):
            ax.add_patch(Circle((centres[sidx, 0], centres[sidx, 2]),
                                radii[sidx] + MARGIN_MM / 1000.0, facecolor="none",
                                edgecolor=colr, lw=1.3, ls="--", zorder=8))
            ax.plot([centres[sidx, 0]], [centres[sidx, 2]], marker="x", ms=8, mew=2.0,
                    color=colr, zorder=9)
        nr = rows_named.get(tstep)
        txt = (f"gripper spheres ({', '.join(GRIPPER_L)})\n"
               f"nearest body, all bodies      {b_all:7s} {row[k_all]:8.2f} mm  "
               f"({link_names[s_all]})\n"
               f"nearest body, apple excluded  {b_na:7s} {rna[k_na]:8.2f} mm  "
               f"({link_names[s_na]})\n"
               f"margin {MARGIN_MM:.0f} mm  ->  both are outside it\n")
        if nr and not nr["is_attached_query_point"]:
            txt += (f"\nrecorded worst row at this chunk\n"
                    f"  link {nr['sphere_link_offline']}  sphere {nr['sphere_index']}\n"
                    f"  ESDF   d - r = {nr['esdf_d_minus_r_mm']:8.2f} mm "
                    f"(clearance_m {nr['recorded_clearance_mm_post_margin']:+.2f} + margin 10)\n"
                    f"  ground truth nearest = {nr['nearest_body_all']} "
                    f"{nr['nearest_mm_all']:.2f} mm\n"
                    f"  difference {nr['offline_minus_esdf_mm_all_bodies']:+.2f} mm")
        ax.annotate(txt, (0.985, 0.985), xycoords="axes fraction", ha="right", va="top",
                    fontsize=7.4, family="monospace", color=figstyle.INK,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=figstyle.SURFACE,
                              edgecolor=figstyle.GRID_INK, alpha=0.94))
        lo = np.array([min(apple[0], crate[0]) - 0.16, table_top - 0.03])
        hi = np.array([max(apple[0], crate[0]) + 0.20, apple[2] + 0.28])
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xlabel("base x  [m]", fontsize=8.6, color=figstyle.INK_2)
        if col == 0:
            ax.set_ylabel("base z  [m]", fontsize=8.6, color=figstyle.INK_2)
        side["panels"].append({
            "t_step": tstep, "chunk_index": i, "seq": int(rec["seq"]), "exec_step": st,
            "nearest_body_all": b_all, "nearest_mm_all": round(float(row[k_all]), 4),
            "nearest_link_all": link_names[s_all],
            "nearest_body_apple_excluded": b_na,
            "nearest_mm_apple_excluded": round(float(rna[k_na]), 4),
            "nearest_link_apple_excluded": link_names[s_na],
            "crate_rim_top_z_m": round(crate_top, 5), "table_top_z_m": round(table_top, 5),
            "apple_pos_m": [round(float(v), 6) for v in apple],
            "recorded_worst_row": nr, "recorded_max_violation_pair": mvp,
            "safe": rec.get("safe")})

    fig.suptitle("T10a - what is actually near the gripper during the approach "
                 "(ground-truth MuJoCo surfaces, 266-sphere model)",
                 fontsize=12.4, color=figstyle.INK, y=0.984)
    fig.text(0.5, 0.010,
             "green = left fingertip spheres · orange = ee_left palm · blue = rest of the "
             "left arm\nred dashed ring = radius + margin 10 mm on the nearest sphere counting "
             "the apple; blue dashed = the same with the apple excluded\n"
             "clearances are PRE-margin ground truth (MuJoCo surfaces), not the ESDF",
             ha="center", fontsize=8.2, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.048, 1, 0.960))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
