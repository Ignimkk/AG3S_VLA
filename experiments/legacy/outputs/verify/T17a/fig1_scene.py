"""T17a figure 1 — 실제 씬에 **접근 구간 손끝 경로**와 **그때 가장 빡빡한 표면**.

왼쪽: crate 가 미는 구간의 한 chunk. 오른쪽: 사과가 미는 구간의 한 chunk.
위 줄 MuJoCo 렌더. 가운데 XZ, 아래 XY.
경로는 `ee_finger_l1` body 원점이 접근 구간(t=0..120) 동안 지난 자리이고,
별표는 각 chunk 의 가장 빡빡한 행이 묶인 표면점(거리장 기울기를 d 만큼 걸어 얻은 것),
색은 그 점이 가장 가까운 참값 body 다.
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
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from m3_to_refined_vs_reference import ACT_COL, MODEL_XML, QADR  # noqa: E402

T17A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T17a'))
ROOT = pathlib.Path(_ag3s_workspace_path('outputs/live_test/20260928_t17'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t17a/t17a-scene.png'))
SIDE = OUT.with_suffix(".json")
APPLE_R = 33.3
BODY_COL = {"apple": figstyle.CATEGORICAL[7], "crate": figstyle.CATEGORICAL[0],
            "table": figstyle.CATEGORICAL[3], "orange": figstyle.CATEGORICAL[1],
            "pear": figstyle.CATEGORICAL[2], "banana": figstyle.CATEGORICAL[4]}


def main() -> None:
    import mujoco

    a = json.loads((T17A / "r1to4.json").read_text())
    rows = {r["t_step"]: r for r in a["r3_if_not_the_apple"]["rows"]}
    per = {r["t_step"]: r for r in a["per_chunk"]}
    P = [json.loads(l) for l in open(ROOT / "shadow_ep1807" / "frames.jsonl")
         if json.loads(l).get("kind") == "planning"]
    by_t = {int(r["t_step"]): r for r in P}
    APP = [t for t in sorted(rows) if t <= a["approach_window_t_step"][1]]

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_finger_l1")

    def tip_path():
        """접근 구간 동안 실행된 8 step 마다의 손끝 자리 (shadow 는 reference 를 실행한다)."""
        pts, ts = [], []
        for t in APP:
            rec = by_t[t]
            ref = np.asarray(rec["actions_reference"], np.float64)
            q = np.asarray(rec["qpos"], np.float64)
            for st in range(8):
                d.qpos[: q.size] = q
                for n, c in ACT_COL.items():
                    d.qpos[QADR[n]] = float(ref[st, c])
                d.qvel[:] = 0.0
                mujoco.mj_forward(m, d)
                pts.append(np.asarray(d.xpos[tip_id], np.float64) * 1000.0)
                ts.append(t + st)
        return np.asarray(pts), np.asarray(ts)

    path, pts_t = tip_path()
    picks = []
    for t in APP:
        if rows[t]["esdf_surface_body"] == "crate" and len(picks) == 0:
            picks.append((t, "the crate is the tightest surface"))
    for t in APP:
        if rows[t]["esdf_surface_body"] == "apple":
            picks.append((t, "the apple is the tightest surface"))
            break

    renderer = mujoco.Renderer(m, height=470, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    figstyle.use_korean()
    fig, axes = plt.subplots(3, 2, figsize=(14.8, 16.2),
                             gridspec_kw={"height_ratios": [1.0, 1.15, 1.15]})
    figstyle.style_axes(fig, axes[1:])
    side: dict = {"approach_t": a["approach_window_t_step"], "panels": []}

    for col, (t, label) in enumerate(picks):
        rec = by_t[t]
        r, pr = rows[t], per[t]
        ref = np.asarray(rec["actions_reference"], np.float64)
        q = np.asarray(rec["qpos"], np.float64)
        st = int(r.get("esdf_gradient_tier") is not None) * 0
        d.qpos[: q.size] = q
        for n, c in ACT_COL.items():
            d.qpos[QADR[n]] = float(ref[0, c])
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        apple = np.asarray(rec["object_poses"]["apple"]["pos"], np.float64) * 1000.0
        crate = np.asarray(rec["object_poses"]["crate"]["pos"], np.float64) * 1000.0
        cbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")
        cg = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == cbid]
        crate_top = max(float(d.geom_xpos[g][2]) + float(m.geom_size[g][2])
                        for g in cg) * 1000.0
        # crate 의 실제 XY 범위를 geom 에서 읽는다 (눈대중 상자를 쓰지 않는다)
        c_lo = np.min([np.asarray(d.geom_xpos[g], np.float64)
                       - np.asarray(m.geom_size[g], np.float64)[:3] for g in cg],
                      axis=0) * 1000.0
        c_hi = np.max([np.asarray(d.geom_xpos[g], np.float64)
                       + np.asarray(m.geom_size[g], np.float64)[:3] for g in cg],
                      axis=0) * 1000.0
        tbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
        table_top = max(float(d.geom_xpos[g][2]) + float(m.geom_size[g][2])
                        for g in range(m.ngeom) if int(m.geom_bodyid[g]) == tbid) * 1000.0
        qpt = np.asarray(pr["mvp_point_mm"], np.float64)
        surf = (None if r.get("esdf_gradient_surface_mm") is None
                else np.asarray(r["esdf_gradient_surface_mm"], np.float64))

        cam.lookat[:] = [apple[0] / 1000, apple[1] / 1000, apple[2] / 1000 + 0.06]
        cam.distance, cam.azimuth, cam.elevation = 0.95, 128.0, -16.0
        renderer.update_scene(d, cam)
        ax = axes[0, col]
        ax.imshow(renderer.render())
        ax.set_axis_off()
        ax.set_title(f"{label}\nt_step {t} · {pr['mvp_link']} · clearance "
                     f"{pr['mvp_clearance_mm_post_margin']:+.2f} mm post-margin · "
                     f"ESDF says {pr['esdf_distance_mm']:.2f} mm · tier "
                     f"{r.get('esdf_gradient_tier')}", fontsize=9.2, color=figstyle.INK)

        for row_i, (ia, ib, alab, blab, vname) in enumerate(
                ((0, 2, "base x  [mm]", "base z  [mm]", "side view (XZ)"),
                 (0, 1, "base x  [mm]", "base y  [mm]", "top view (XY)"))):
            ax = axes[1 + row_i, col]
            if row_i == 0:
                ax.axhline(table_top, color=figstyle.GRID_INK, lw=1.6, zorder=1)
                ax.annotate("table top", (apple[ia] - 240, table_top + 6), fontsize=7.6,
                            color=figstyle.INK_2)
                ax.add_patch(Rectangle((c_lo[0], c_lo[2]), c_hi[0] - c_lo[0],
                                       c_hi[2] - c_lo[2], facecolor="none",
                                       edgecolor=BODY_COL["crate"], lw=1.6, zorder=3))
            else:
                ax.add_patch(Rectangle((c_lo[0], c_lo[1]), c_hi[0] - c_lo[0],
                                       c_hi[1] - c_lo[1], facecolor="none",
                                       edgecolor=BODY_COL["crate"], lw=1.6, zorder=3))
            ax.annotate("crate", (crate[ia] + 95, crate[ib]), fontsize=7.8,
                        color=BODY_COL["crate"], zorder=9)
            ax.add_patch(Circle((apple[ia], apple[ib]), APPLE_R, facecolor=BODY_COL["apple"],
                                edgecolor="none", alpha=0.95, zorder=5))
            ax.annotate("apple", (apple[ia], apple[ib] + APPLE_R + 14), ha="center",
                        fontsize=8.0, color=BODY_COL["apple"], zorder=9)
            sc = ax.scatter(path[:, ia], path[:, ib], c=pts_t, cmap="viridis", s=7, zorder=6)
            ax.plot(path[:, ia], path[:, ib], color=figstyle.INK_2, lw=0.7, alpha=0.5, zorder=5)
            for tt in APP:
                rr = rows[tt]
                if rr.get("esdf_gradient_surface_mm") is None:
                    continue
                sp = np.asarray(rr["esdf_gradient_surface_mm"], np.float64)
                ax.plot([sp[ia]], [sp[ib]], marker="*", ms=11,
                        color=BODY_COL.get(rr["esdf_surface_body"], figstyle.INK),
                        markeredgecolor=figstyle.INK, markeredgewidth=0.4, zorder=8)
            ax.plot([qpt[ia]], [qpt[ib]], marker="x", ms=11, mew=2.6,
                    color=figstyle.CATEGORICAL[6], zorder=9)
            ax.annotate(f"query point\n{pr['mvp_link']}", (qpt[ia], qpt[ib]),
                        textcoords="offset points", xytext=(9, 9), fontsize=7.6,
                        color=figstyle.CATEGORICAL[6], zorder=9)
            if surf is not None:
                ax.annotate("", xy=(surf[ia], surf[ib]), xytext=(qpt[ia], qpt[ib]),
                            arrowprops=dict(arrowstyle="->", lw=1.4,
                                            color=BODY_COL.get(r["esdf_surface_body"],
                                                               figstyle.INK)), zorder=9)
            A = np.vstack([path[:, [ia, ib]], [[apple[ia], apple[ib]], [crate[ia], crate[ib]],
                                               [qpt[ia], qpt[ib]]]])
            pad = 80
            ax.set_xlim(A[:, 0].min() - pad, A[:, 0].max() + pad)
            ax.set_ylim(A[:, 1].min() - pad, A[:, 1].max() + pad)
            ax.set_aspect("equal")
            ax.set_xlabel(alab, fontsize=8.6, color=figstyle.INK_2)
            ax.set_ylabel(blab, fontsize=8.6, color=figstyle.INK_2)
            ax.set_title(vname + "   ·   stars = the surface each chunk's tightest row was "
                                 "bound to", fontsize=8.6, color=figstyle.INK_2)
            if row_i == 0 and col == 0:
                cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
                cb.set_label("t_step along the approach", fontsize=7.6, color=figstyle.INK_2)
                cb.ax.tick_params(labelsize=7)
        side["panels"].append({
            "t_step": t, "label": label, "mvp_link": pr["mvp_link"],
            "mvp_clearance_mm_post_margin": pr["mvp_clearance_mm_post_margin"],
            "esdf_distance_mm": pr["esdf_distance_mm"],
            "esdf_surface_body": r["esdf_surface_body"],
            "esdf_surface_body_distance_mm": r["esdf_surface_body_distance_mm"],
            "esdf_gradient_tier": r.get("esdf_gradient_tier"),
            "query_point_mm": pr["mvp_point_mm"],
            "gt_distance_by_body_mm": r["gt_distance_by_body_mm"],
            "grounding_centroid_nearest": pr.get("centroid_nearest_object"),
            "apple_in_fine_window": pr.get("apple_in_fine_window"),
            "ball_contains_apple_centre": pr.get("ball_contains_apple_centre"),
            "crate_extent_mm": {"lo": [round(float(v), 2) for v in c_lo],
                                "hi": [round(float(v), 2) for v in c_hi]},
            "table_top_z_mm": round(table_top, 2)})
    side["tip_path_mm"] = [[round(float(v), 2) for v in p] for p in path]
    side["tip_path_t_step"] = [int(v) for v in pts_t]
    side["surface_points"] = {str(t): {"body": rows[t]["esdf_surface_body"],
                                       "point_mm": rows[t].get("esdf_gradient_surface_mm"),
                                       "tier": rows[t].get("esdf_gradient_tier")}
                              for t in APP}

    fig.suptitle("T17a - the successful shadow approach: where the fingertip went and what the "
                 "tightest row was bound to", fontsize=12.4, color=figstyle.INK, y=0.988)
    fig.text(0.5, 0.008,
             "dots = the ee_finger_l1 origin through the executed 8 steps of every approach "
             "chunk, coloured by t_step\n"
             "stars = the surface point reached by walking the recorded distance field down its "
             "gradient, coloured by the ground-truth body it lands on (blue crate, red apple)\n"
             "x = the query point of that chunk's tightest row · crate outline from its own "
             "collision geoms",
             ha="center", fontsize=8.2, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.032, 1, 0.968))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
