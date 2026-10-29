"""T8c figure 4 — 실제 씬. density 를 올리면 arm_5 구 사슬이 어떻게 달라지는가.

왼쪽: MuJoCo 렌더 (chunk 24 step 17 — 남은 최악 행이 있는 자세).
오른쪽 네 칸: 같은 자세의 base frame XZ 에 `link_left_arm_5` 구 사슬을
  n = 5 (현재) · 8 · 13 · 21 로 겹쳐 그린 것. 각 칸은
  (a) 배율 1.0 의 구 (옅은 채움)  (b) 그 density 에서 exec8 위반 0 에 닿는 배율의 구 (테두리)
  (c) `ee_finger_l1` 뿌리 위치 (x 표시) — arm_5 capsule 이 이것을 삼키는지 눈으로.
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

S4 = json.loads(pathlib.Path(
    _ag3s_workspace_path('outputs/verify/T8c/s4_density_scale_sweep.json')).read_text())
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8c/t8c-density-scene.png'))
SIDE = OUT.with_suffix(".json")
CHUNK, STEP = 24, 17
MARGIN_MM = 10.0


def main() -> None:
    import mujoco

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    scene = TransportScene.attach(m, d)
    recs = load_planning(SHADOW)
    rec = recs[CHUNK]
    chunk = np.asarray(rec["actions_reference"], np.float64)
    qpos = np.asarray(rec["qpos"], np.float64)
    d.qpos[: qpos.size] = qpos
    for n, ci in ACT_COL.items():
        d.qpos[QADR[n]] = float(chunk[STEP, ci])
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    q = np.asarray([qpos[QADR[n]] if n.startswith("torso_") else float(chunk[STEP, ACT_COL[n]])
                    for n in JOINT_ORDER], np.float64)

    finger_root = np.asarray(
        d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_finger_l1")], np.float64)
    ee_left = np.asarray(
        d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_left")], np.float64)
    tbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
    top_z = max(float(d.geom_xpos[g][2]) + float(m.geom_size[g][2])
                for g in range(m.ngeom) if int(m.geom_bodyid[g]) == tbid)

    figstyle.use_korean()
    fig = plt.figure(figsize=(21.0, 5.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.15, 1, 1, 1, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    axs = [fig.add_subplot(gs[0, i + 1]) for i in range(4)]
    figstyle.style_axes(fig, np.asarray(axs))

    renderer = mujoco.Renderer(m, height=430, width=520)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.52, 0.18, 0.92]
    cam.distance, cam.azimuth, cam.elevation = 1.05, 128.0, -18.0
    renderer.update_scene(d, cam)
    ax0.imshow(renderer.render())
    ax0.set_axis_off()
    ax0.set_title(f"chunk {CHUNK} (seq {rec['seq']}) · plan step {STEP}\n"
                  f"the pose that sets the worst remaining row", fontsize=9.5, color=figstyle.INK)

    side: dict = {"chunk": CHUNK, "plan_step": STEP, "seq": int(rec["seq"]),
                  "table_top_z_m": round(top_z, 5),
                  "ee_finger_l1_root_world": [round(float(v), 5) for v in finger_root],
                  "ee_left_world": [round(float(v), 5) for v in ee_left],
                  "panels": []}

    for k, dn in enumerate(S4["densities"]):
        ax = axs[k]
        spacing, cap = dn["sphere_spacing"], dn["max_spheres_per_capsule"]
        # exec8 위반 0 에 닿는 가장 큰 arm5 배율 (finger 0.35 계열에서)
        feas = [c for c in dn["combos"] if c["windows"]["exec8"]["feasible_at_10mm"]]
        pick = max(feas, key=lambda c: c["arm5_scale"]) if feas else None
        base = build_constraint_robot_model(
            scene, link_filter=ARM_LINKS,
            sphere_options={"sphere_spacing": spacing, "max_spheres_per_capsule": cap})
        names = list(resolve_link_names(base, base.n_spheres))
        c1, r1 = base.sphere_centers_numeric(q)
        c1 = np.asarray(c1, np.float64).reshape(-1, 3)
        r1 = np.asarray(r1, np.float64).reshape(-1)
        if pick is not None:
            by = {"link_left_arm_5": pick["arm5_scale"], "link_right_arm_5": pick["arm5_scale"],
                  "ee_finger_l1": pick["finger_scale"], "ee_finger_l2": pick["finger_scale"]}
            by = {kk: vv for kk, vv in by.items() if vv != 1.0}
            rob2 = build_constraint_robot_model(
                scene, link_filter=ARM_LINKS,
                sphere_options={"sphere_spacing": spacing, "max_spheres_per_capsule": cap,
                                **({"capsule_radius_scale_by_link": by} if by else {})})
            r2 = np.asarray(rob2.sphere_centers_numeric(q)[1], np.float64).reshape(-1)
        else:
            r2 = None

        ax.axhline(top_z, color=figstyle.GRID_INK, lw=1.5, zorder=1)
        cr = np.asarray(rec["object_poses"]["crate"]["pos"], np.float64)
        ax.add_patch(plt.Rectangle((cr[0] - 0.09, cr[2] - 0.004), 0.18, 0.112, facecolor="none",
                                   edgecolor=figstyle.INK_2, lw=1.2, zorder=3))
        ap = np.asarray(rec["object_poses"]["apple"]["pos"], np.float64)
        ax.add_patch(Circle((ap[0], ap[2]), 0.0333, facecolor=figstyle.CATEGORICAL[7],
                            edgecolor="none", alpha=0.85, zorder=5))
        for i, n in enumerate(names):
            if n == "link_left_arm_5":
                ax.add_patch(Circle((c1[i, 0], c1[i, 2]), r1[i], facecolor=figstyle.CATEGORICAL[1],
                                    edgecolor="none", alpha=0.17, zorder=2))
                if r2 is not None:
                    ax.add_patch(Circle((c1[i, 0], c1[i, 2]), r2[i], facecolor="none",
                                        edgecolor=figstyle.CATEGORICAL[0], lw=1.1, zorder=6))
            elif n.startswith("ee_finger_l"):
                ax.add_patch(Circle((c1[i, 0], c1[i, 2]), r1[i], facecolor=figstyle.CATEGORICAL[2],
                                    edgecolor="none", alpha=0.5, zorder=7))
        ax.plot([finger_root[0]], [finger_root[2]], marker="x", ms=9, mew=2.2,
                color=figstyle.CATEGORICAL[6], zorder=9)
        ax.annotate("ee_finger_l1 root", (finger_root[0], finger_root[2]),
                    textcoords="offset points", xytext=(7, -10), fontsize=7.2,
                    color=figstyle.CATEGORICAL[6])
        ax.plot([ee_left[0]], [ee_left[2]], marker="+", ms=9, mew=2.0,
                color=figstyle.CATEGORICAL[4], zorder=9)
        r1_a5 = float(r1[[i for i, n in enumerate(names) if n == "link_left_arm_5"][0]]) * 1000.0
        txt = (f"n = {dn['arm5_n_spheres']}  (spacing {spacing:g}, cap {cap})\n"
               f"arm5 r at scale 1.0   {r1_a5:6.2f} mm\n"
               f"sphere-chain floor    {250.0 / (dn['arm5_n_spheres'] - 1) / 2:6.2f} mm\n"
               f"total spheres         {dn['n_spheres_total']:6d}")
        if pick is not None:
            e8 = pick["windows"]["exec8"]
            f50 = pick["windows"]["full50"]
            txt += (f"\nexec8 feasible at arm5 x{pick['arm5_scale']:g}"
                    f" finger x{pick['finger_scale']:g}\n"
                    f"  arm5 r {pick['radius_mm_link_left_arm_5']:6.2f} mm · "
                    f"min slack {e8['global_min_slack_mm']:+.2f} mm\n"
                    f"  same combo on full50: {f50['n_rows_slack_le0']} rows, "
                    f"min {f50['global_min_slack_mm']:+.2f} mm")
        else:
            txt += "\nno combo in the grid reaches 0 in exec8"
        ax.annotate(txt, (0.02, 0.98), xycoords="axes fraction", va="top", fontsize=7.2,
                    family="monospace", color=figstyle.INK,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=figstyle.SURFACE,
                              edgecolor=figstyle.GRID_INK, alpha=0.93))
        ax.set_xlim(0.26, 0.78)
        ax.set_ylim(0.80, 1.24)
        ax.set_aspect("equal")
        ax.set_xlabel("base x  [m]", fontsize=8.5, color=figstyle.INK_2)
        if k == 0:
            ax.set_ylabel("base z  [m]", fontsize=8.5, color=figstyle.INK_2)
        side["panels"].append({
            "sphere_spacing": spacing, "max_spheres_per_capsule": cap,
            "arm5_n_spheres": dn["arm5_n_spheres"], "n_spheres_total": dn["n_spheres_total"],
            "arm5_radius_mm_at_scale_1": round(r1_a5, 4),
            "arm5_floor_mm": round(250.0 / (dn["arm5_n_spheres"] - 1) / 2, 4),
            "exec8_feasible_pick": None if pick is None else {
                "key": pick["key"], "arm5_scale": pick["arm5_scale"],
                "finger_scale": pick["finger_scale"],
                "arm5_radius_mm": pick["radius_mm_link_left_arm_5"],
                "exec8": {kk: pick["windows"]["exec8"][kk] for kk in
                          ("n_rows_slack_le0", "n_chunks_slack_le0", "global_min_slack_mm")},
                "full50": {kk: pick["windows"]["full50"][kk] for kk in
                           ("n_rows_slack_le0", "n_chunks_slack_le0", "global_min_slack_mm")},
                "worst_shortfall_mm": pick["worst_shortfall_mm"]}})

    fig.suptitle("T8c - the density axis: link_left_arm_5 sphere chain at n = 5, 8, 13, 21, "
                 "and the radius each one can reach", fontsize=12.5, color=figstyle.INK, y=0.975)
    fig.text(0.5, 0.012,
             "pale orange = link_left_arm_5 spheres at scale 1.0 · blue outline = the same "
             "spheres at the smallest grid scale that clears exec8 · green = left fingertip "
             "spheres · x = ee_finger_l1 root, + = ee_left (palm): both sit inside the arm_5 "
             "chain at every density",
             ha="center", fontsize=8.3, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.945))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
