"""T31-diag D4 — ESDF slices (fine 5 mm / coarse 20 mm / target-free) for selected chunks of one run.

usage: MUJOCO_GL=osmesa PYTHONPATH=outputs/verify/T31diag/snap .venv-ag3s/bin/python d4_slices.py RUN T [T ...]
Frame: robot base frame (server field frame). Field values: recorded float16 grids, trilinear (EsdfField.distance), sampled
every 2.5 mm; a panel shows only its own layer (no tier composition). Planes: xy at the GT crate rim height, xy at the GT apple
centre height of that chunk (held apple), xz through the GT crate centre y. Spheres / held points: FK of the recorded
reference_chunk and refined_chunk, 8 steps (same reconstruction as d2_rows_ablation.py), projected onto each plane.
Crate outline: plane intersection of the 13 MuJoCo crate collision boxes. Apple: GT collision mesh vertices within 4 mm of the plane.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T31diag/d4'))
import fig_common as FC
from AG3S.ag3s.experiments.common import figstyle as fs

KO = fs.use_korean(); C = fs.CATEGORICAL
CONT = [(0.0, "-", 1.4), (0.005, "--", 1.0), (0.010, "-.", 1.0), (0.020, ":", 1.2)]


def panel(ax, layer, ch, axis, value, la, lb, R, title, sidecar_key, side, show_window=None):
    ua, ub, V = FC.slice_values(layer, axis, value, la, lb, 0.0025)
    cm = fs.sequential_cmap()
    ext = [ua[0], ua[-1], ub[0], ub[-1]]
    ax.imshow(np.clip(V, 0, 0.1), origin="lower", extent=ext, cmap=cm, vmin=0, vmax=0.1, interpolation="nearest")
    neg = np.where(V < 0, 1.0, np.nan)
    ax.imshow(neg, origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([C[7]]), alpha=0.55, interpolation="nearest")
    A, B = np.meshgrid(ua, ub)
    for lev, ls, lw in CONT:
        if np.nanmin(V) < lev < np.nanmax(V):
            ax.contour(A, B, V, levels=[lev], colors=fs.INK, linestyles=ls, linewidths=lw)
    other = [k for k in range(3) if k != axis]
    for bx in ch.boxes:
        poly = FC.plane_box_polygon(bx, axis, value)
        if poly is not None:
            ax.add_patch(Polygon(poly, closed=True, fill=False, ec=C[1], lw=1.3))
    am = ch.apple_mesh[np.abs(ch.apple_mesh[:, axis] - value) < 0.004]
    if len(am):
        ax.plot(am[:, other[0]], am[:, other[1]], ".", ms=1.5, color=C[2])
    for key, col, ls in (("reference", C[3], "--"), ("refined", C[0], "-")):
        cen = R.traj[key]["centres"]; rad = R.traj[key]["radii"]
        for k in range(cen.shape[0]):
            for q in R.finger_idx:
                ax.add_patch(Circle(cen[k, q, other], rad[q], fill=False, ec=col, lw=0.6, ls=ls, alpha=0.8))
            if cen.shape[1] > R.S:
                hp = cen[k, R.S:]
                ax.plot(hp[:, other[0]], hp[:, other[1]], "x" if key == "reference" else ".", ms=2.5, color=col, alpha=0.7)
    if show_window is not None:
        g = show_window.grid; lo = g.origin; hi = g.origin + (np.asarray(g.shape) - 1) * g.voxel_size
        if lo[axis] <= value <= hi[axis]:
            ax.add_patch(Rectangle((lo[other[0]], lo[other[1]]), hi[other[0]] - lo[other[0]], hi[other[1]] - lo[other[1]],
                                   fill=False, ec=fs.INK_2, ls=(0, (4, 3)), lw=0.9))
    ax.set_xlim(la); ax.set_ylim(lb); ax.set_aspect("equal")
    lab = "xyz"
    ax.set_xlabel(f"{lab[other[0]]} (m, base)"); ax.set_ylabel(f"{lab[other[1]]} (m, base)")
    ax.set_title(title, fontsize=9)
    finite = V[np.isfinite(V)]
    side[sidecar_key] = {"plane_axis": "xyz"[axis], "plane_value_m": float(value), "covered_fraction": float(np.isfinite(V).mean()),
                         "min_mm": float(1000 * finite.min()) if finite.size else None,
                         "n_samples_lt0": int((finite < 0).sum()), "n_samples_lt5mm": int((finite < 0.005).sum()), "n_samples": int(V.size)}


def wall_profile(layer, ch):
    """d along base-x through the crate centre at mid-wall height (crate centre z + 4 mm): widths where d<0 and d<5 mm (mm)."""
    x = np.arange(ch.crate_centre[0] - 0.15, ch.crate_centre[0] + 0.15, 0.0005)
    pts = np.stack([x, np.full_like(x, ch.crate_centre[1]), np.full_like(x, ch.crate_centre[2] + 0.004)], axis=1)
    g = layer.grid; last = g.origin + (np.asarray(g.shape) - 1) * g.voxel_size
    ins = np.all((pts >= g.origin) & (pts <= last), axis=1)
    if not ins.all():
        return {"covered": False, "covered_fraction": float(ins.mean())}
    v = np.asarray(layer.distance(pts))
    return {"covered": True, "width_lt0_mm": float(0.5 * (v < 0).sum()), "width_lt5mm_mm": float(0.5 * (v < 0.005).sum()),
            "width_lt10mm_mm": float(0.5 * (v < 0.010).sum()), "min_mm": float(1000 * v.min()),
            "gt_wall_thickness_mm_each": 8.0, "n_gt_walls_on_line": 2}


def main():
    run = sys.argv[1]; ts = [int(x) for x in sys.argv[2:]]
    R = FC.Run(run)
    for t in ts:
        ch = R.load(t)
        R.traj = ch.traj
        fine, coarse = ch.field.layers[1], ch.field.layers[0]
        free = ch.field.target_free_layers[0] if ch.field.target_free_layers else None
        cc = ch.crate_centre
        hz = ch.apple_centre[2]
        planes = [(2, ch.crate_rim_z - 0.005, (cc[0] - 0.3, cc[0] + 0.3), (cc[1] - 0.35, cc[1] + 0.45), "xy @ crate rim z − 5 mm" if not KO else "xy @ crate 테두리 −5 mm"),
                  (2, hz, (cc[0] - 0.3, cc[0] + 0.3), (cc[1] - 0.35, cc[1] + 0.45), "xy @ held-apple z" if not KO else "xy @ 쥔 사과 중심 높이"),
                  (1, cc[1], (cc[0] - 0.3, cc[0] + 0.3), (ch.table_top_z - 0.05, ch.table_top_z + 0.55), "xz @ crate centre y" if not KO else "xz @ crate 중심 y")]
        fig, axes = plt.subplots(3, 3, figsize=(16, 17))
        side = {"run": run, "t": t, "server_chunk": ch.c["server_chunk"], "attached": ch.attached,
                "crate_centre_base_m": cc.tolist(), "crate_rim_z_base_m": float(ch.crate_rim_z), "apple_centre_base_m": ch.apple_centre.tolist(),
                "fine_window": {"origin": np.asarray(fine.grid.origin).tolist(), "shape": list(fine.grid.shape), "voxel_m": fine.grid.voxel_size},
                "target_free_recorded": free is not None, "panels": {},
                "recorded_dev_deg": ch.c.get("refined_minus_reference_deg"),
                "wall_profile": {"fine": wall_profile(fine, ch), "coarse": wall_profile(coarse, ch)}}
        # fine window coverage of the crate boxes (fraction of box corners inside the fine window)
        allc = np.vstack(ch.boxes); g = fine.grid; last = g.origin + (np.asarray(g.shape) - 1) * g.voxel_size
        side["crate_box_corners_inside_fine_window"] = float(np.all((allc >= g.origin) & (allc <= last), axis=1).mean())
        for i, (axis, val, la, lb, pname) in enumerate(planes):
            panel(axes[i, 0], fine, ch, axis, val, la, lb, R, f"fine 5 mm · {pname}", f"fine_{i}", side["panels"])
            panel(axes[i, 1], coarse, ch, axis, val, la, lb, R, f"coarse 20 mm · {pname}", f"coarse_{i}", side["panels"], show_window=fine)
            if free is not None:
                panel(axes[i, 2], free, ch, axis, val, la, lb, R, f"target-free 5 mm · {pname}", f"free_{i}", side["panels"])
            else:
                axes[i, 2].axis("off")
                axes[i, 2].text(0.5, 0.5, ("target-free layer: not in this record\n(built only while nothing is held;\n"
                                           "attached from t=%s)" % (R.att and "grasp")) if not KO else
                                "target-free 층: 이 기록에 없음\n(쥔 것이 없을 때만 만든다 —\npipeline.py:1644-1650)", ha="center", va="center", fontsize=11, color=fs.INK_2)
        fs.style_axes(fig, axes.ravel().tolist())
        h = [plt.Line2D([], [], color=C[1], lw=1.3), plt.Line2D([], [], color=C[2], marker=".", ls=""),
             plt.Line2D([], [], color=C[3], ls="--"), plt.Line2D([], [], color=C[0]),
             matplotlib.patches.Patch(color=C[7], alpha=0.55)] + [plt.Line2D([], [], color=fs.INK, ls=ls, lw=lw) for _, ls, lw in CONT] + \
            [plt.Line2D([], [], color=fs.INK_2, ls=(0, (4, 3)))]
        labs = (["GT crate boxes", "GT apple mesh (plane ±4 mm)", "reference fingers / held pts (8 steps, projected)",
                 "refined fingers / held pts (8 steps, projected)", "d < 0"] if not KO else
                ["GT crate 상자", "GT 사과 mesh (면 ±4 mm)", "reference 손가락 구 / 쥔 점 (8 스텝, 투영)", "refined 손가락 구 / 쥔 점 (8 스텝, 투영)", "d < 0"]) + \
               ["d = 0", "d = 5 mm", "d = 10 mm", "d = 20 mm"] + (["fine window" if not KO else "fine 창"])
        fig.legend(h, labs, loc="lower center", ncol=5, fontsize=9, frameon=False)
        dev = ch.c.get("refined_minus_reference_deg")
        fig.suptitle(f"{run}  t={t}  ESDF slices (colour: d 0–100 mm, SEQ_BLUE)  refined−reference {dev:.2f}°  attached={ch.attached}", fontsize=12)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        out = FC.FIG / f"t31-d4-slices-{run}-t{t:03d}.png"
        fig.savefig(out, dpi=110); plt.close(fig)
        out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float))
        print(out, side["wall_profile"], side["crate_box_corners_inside_fine_window"], flush=True)


if __name__ == "__main__":
    main()
