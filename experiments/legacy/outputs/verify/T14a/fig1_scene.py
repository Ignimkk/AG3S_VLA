"""T14a figure 1 — 실제 씬 위에 **미세 계층 창**과 사과·손끝·target centroid·공.

왼쪽: 사과가 fine 창 **안**에 있는 chunk. 오른쪽: **밖**으로 나간 chunk.
위 줄  MuJoCo 렌더 (기록된 qpos).
가운데 base frame **XY** (위에서 본 것) — 창이 벗어나는 축이 y 다.
아래 줄 base frame **XZ**.
창은 `esdf_origin + esdf_shape * esdf_voxel_size` 로 그린 것이고, coarse 창도 함께 그린다.
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
from m3_to_refined_vs_reference import MODEL_XML  # noqa: E402

T14A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T14a'))
ROOT = pathlib.Path(_ag3s_workspace_path('outputs/live_test/20260928_t14'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t14a/t14a-scene.png'))
SIDE = OUT.with_suffix(".json")
APPLE_R = 33.3


def main() -> None:
    import mujoco

    ana = json.loads((T14A / "q1to4.json").read_text())
    q1 = ana["q1_fine_window"]["T14"]["per_chunk"]
    inside = [r for r in q1 if r["apple_in_fine_window"] is True]
    outside = [r for r in q1 if r["apple_in_fine_window"] is False]
    picks = [(inside[len(inside) // 2], "apple INSIDE the fine window"),
             (outside[0], "apple OUTSIDE the fine window")]

    P = [json.loads(l) for l in open(ROOT / "execute_ep1807" / "frames.jsonl")
         if json.loads(l).get("kind") == "planning"]
    plan_by_t = {int(r["t_step"]): r for r in P}
    files = {int(json.loads(str(np.load(f, allow_pickle=True)["summary_json"]))["chunk_index"]): f
             for f in sorted((ROOT / "server_constraints" / "run_0000").glob("chunk_*.npz"))}

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=470, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    figstyle.use_korean()
    fig, axes = plt.subplots(3, 2, figsize=(14.6, 16.0),
                             gridspec_kw={"height_ratios": [1.0, 1.15, 1.15]})
    figstyle.style_axes(fig, axes[1:])
    side: dict = {"panels": []}

    for col, (r, label) in enumerate(picks):
        t = r["t_step"]
        rec = plan_by_t[t]
        z = np.load(files[r["chunk_index"]], allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        qpos = np.asarray(rec["qpos"], np.float64)
        d.qpos[: qpos.size] = qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        apple = np.asarray(r["apple_mm"], np.float64)
        lo = np.asarray(r["fine_origin_mm"], np.float64)
        hi = np.asarray(r["fine_max_mm"], np.float64)
        clo = (None if r["coarse_origin_mm"] is None
               else np.asarray(r["coarse_origin_mm"], np.float64))
        chi = (None if clo is None
               else clo + np.asarray([128, 128, 128]) * float(r["coarse_voxel_mm"]))
        cen = None if r["target_centroid_mm"] is None else np.asarray(r["target_centroid_mm"])
        tip = None if r["tip_mm"] is None else np.asarray(r["tip_mm"], np.float64)
        tpts = (np.asarray(z["target_points"], np.float64) * 1000.0
                if "target_points" in z.files else None)

        cam.lookat[:] = [apple[0] / 1000.0, apple[1] / 1000.0, apple[2] / 1000.0 + 0.06]
        cam.distance, cam.azimuth, cam.elevation = 0.85, 128.0, -16.0
        renderer.update_scene(d, cam)
        ax = axes[0, col]
        ax.imshow(renderer.render())
        ax.set_axis_off()
        ax.set_title(f"{label}\nchunk {r['chunk_index']} · local t_step {t} · server seq "
                     f"{s['t_step']} · grounding {s.get('grounding_status')} · "
                     f"has_target {s.get('has_target')}", fontsize=9.4, color=figstyle.INK)

        for row, (ia, ib, alab, blab, vname) in enumerate(
                ((0, 1, "base x  [mm]", "base y  [mm]", "top view (XY)"),
                 (0, 2, "base x  [mm]", "base z  [mm]", "side view (XZ)"))):
            ax = axes[1 + row, col]
            if clo is not None:
                ax.add_patch(Rectangle((clo[ia], clo[ib]), chi[ia] - clo[ia], chi[ib] - clo[ib],
                                       facecolor=figstyle.SEQ_BLUE[1], edgecolor=figstyle.INK_2,
                                       lw=1.0, ls=":", alpha=0.25, zorder=1))
                ax.annotate(f"coarse layer, {r['coarse_voxel_mm']:.0f} mm voxels",
                            (clo[ia] + 8, chi[ib] - 26), fontsize=7.4, color=figstyle.INK_2,
                            zorder=9)
            ax.add_patch(Rectangle((lo[ia], lo[ib]), hi[ia] - lo[ia], hi[ib] - lo[ib],
                                   facecolor=figstyle.CATEGORICAL[0], edgecolor=figstyle.
                                   CATEGORICAL[0], lw=2.0, alpha=0.13, zorder=2))
            ax.annotate(f"fine layer, {r['fine_voxel_mm']:.0f} mm voxels\n"
                        f"{alab[5]} [{lo[ia]:.0f}, {hi[ia]:.0f}]  "
                        f"{blab[5]} [{lo[ib]:.0f}, {hi[ib]:.0f}] mm",
                        (lo[ia] + 8, hi[ib] - 8), fontsize=7.6,
                        color=figstyle.CATEGORICAL[0], va="top", zorder=9)
            if tpts is not None:
                ax.scatter(tpts[:, ia], tpts[:, ib], s=1.2, color=figstyle.CATEGORICAL[4],
                           alpha=0.35, zorder=4, label="target points (grounding)")
            ax.add_patch(Circle((apple[ia], apple[ib]), APPLE_R,
                                facecolor=figstyle.CATEGORICAL[7], edgecolor="none",
                                alpha=0.95, zorder=6))
            ax.annotate("apple", (apple[ia], apple[ib] + APPLE_R + 14), ha="center",
                        fontsize=8.0, color=figstyle.CATEGORICAL[7], zorder=9)
            if cen is not None:
                ax.plot([cen[ia]], [cen[ib]], marker="*", ms=15,
                        color=figstyle.CATEGORICAL[3], zorder=8)
                ax.annotate("target centroid", (cen[ia], cen[ib]), textcoords="offset points",
                            xytext=(8, -12), fontsize=7.8, color=figstyle.CATEGORICAL[3],
                            zorder=9)
                ax.annotate("", xy=(apple[ia], apple[ib]), xytext=(cen[ia], cen[ib]),
                            arrowprops=dict(arrowstyle="<->", color=figstyle.CATEGORICAL[3],
                                            lw=1.0, ls="--"), zorder=7)
            if tip is not None:
                ax.plot([tip[ia]], [tip[ib]], marker="x", ms=10, mew=2.4,
                        color=figstyle.CATEGORICAL[2], zorder=8)
                ax.annotate("fingertip", (tip[ia], tip[ib]), textcoords="offset points",
                            xytext=(8, 8), fontsize=7.8, color=figstyle.CATEGORICAL[2],
                            zorder=9)
            pts = [apple, lo, hi]
            if cen is not None:
                pts.append(cen)
            if tip is not None:
                pts.append(tip)
            A = np.asarray(pts)
            pad = 90
            ax.set_xlim(A[:, ia].min() - pad, A[:, ia].max() + pad)
            ax.set_ylim(A[:, ib].min() - pad, A[:, ib].max() + pad)
            ax.set_aspect("equal")
            ax.set_xlabel(alab, fontsize=8.6, color=figstyle.INK_2)
            ax.set_ylabel(blab, fontsize=8.6, color=figstyle.INK_2)
            ax.set_title(vname + ("   ·   apple is inside" if r["apple_in_fine_window"]
                                  else f"   ·   apple is {r['apple_distance_outside_fine_mm']:.0f}"
                                       " mm outside the fine window"),
                         fontsize=8.8,
                         color=figstyle.INK if r["apple_in_fine_window"]
                         else figstyle.CATEGORICAL[7])
            if row == 0 and col == 0 and tpts is not None:
                ax.legend(fontsize=7.2, loc="lower left", framealpha=0.9)
        side["panels"].append({
            "label": label, "chunk_index": r["chunk_index"], "local_t_step": t,
            "server_seq": int(s["t_step"]), "grounding_status": s.get("grounding_status"),
            "has_target": bool(s.get("has_target")),
            "fine_origin_mm": r["fine_origin_mm"], "fine_max_mm": r["fine_max_mm"],
            "fine_voxel_mm": r["fine_voxel_mm"],
            "coarse_origin_mm": r["coarse_origin_mm"], "coarse_voxel_mm": r["coarse_voxel_mm"],
            "apple_mm": r["apple_mm"], "apple_in_fine_window": r["apple_in_fine_window"],
            "apple_distance_outside_fine_mm": r["apple_distance_outside_fine_mm"],
            "target_centroid_mm": r["target_centroid_mm"],
            "centroid_minus_apple_mm": r["centroid_minus_apple_mm"],
            "tip_mm": r["tip_mm"], "tip_in_fine_window": r["tip_in_fine_window"],
            "n_target_points": int(s.get("target_n_points") or 0)})

    fig.suptitle("T14a - the fine ESDF window against the apple, the fingertip and the target "
                 "centroid", fontsize=12.6, color=figstyle.INK, y=0.988)
    fig.text(0.5, 0.008,
             "blue box = the 5 mm fine layer window (origin + shape x voxel) · pale box = the "
             "20 mm coarse layer · pink dots = the grounding's target points · star = the target "
             "centroid the window is centred on · the dashed arrow is centroid minus apple",
             ha="center", fontsize=8.2, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.016, 1, 0.968))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
