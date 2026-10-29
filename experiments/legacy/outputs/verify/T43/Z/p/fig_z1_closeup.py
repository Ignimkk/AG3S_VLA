"""T43 Z1 figure t43z-held-sphere-closeup — equal-scale side (x-z) and top (x-y) close-ups, V4 vs V5, 3 pairs.
Drawn in the crate frame (mm) with the apple placed at the crate centre, its lowest mesh vertex on the floor top (z = -52 mm);
held query spheres (server query set at the attach chunk), their 20 mm destination-margin shells and the closing carve-out ball are placed
at their attach-row offsets from the apple centre (world axes; crate quaternion (1,0,0,0) in all these runs). Orthographic projection:
a sphere projects to a circle of the same radius. Crate inner walls / floor from the MJCF collision boxes. Numbers: raw/z1_held.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

Z = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/Z')); FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
D = json.load(open(Z / "raw/z1_held.json")); R = D["runs"]; MG = D["margin_mm"]; CI = D["crate_inner_mm"]
PAIRS = ["1925_s19253", "1807_s18071", "1968_s19681"]
C = figstyle.CATEGORICAL
APPLE = "#9a9a94"; WALL = "#b9b7b0"
XL = (-105, 105); ZL = (-70, 140); YL = (-160, 160)
kor = figstyle.use_korean()


def placed(o):
    off = np.array([0.0, 0.0, CI["floor_top_z"] - (o["apple_bottom_z_mm"] - o["apple_centre_mm"][2])])   # apple centre in crate frame
    V = np.array(o["apple_mesh_rel_mm"]) + off; F = np.array(o["apple_mesh_faces"])
    S = [(np.array(s["centre_rel_apple_mm"]) + off, s["radius_mm"]) for s in o["sets"][o["query_set"]]]
    b = o.get("carve_ball"); B = (np.array(b["ball_rel_apple_mm"]) + off, b["radius_mm"]) if b else None
    return off, V, F, S, B


def draw(ax, o, view):
    off, V, F, S, B = placed(o)
    i, j = (0, 2) if view == "side" else (0, 1)
    if view == "side":
        for x0, x1 in ((CI["half_x"], CI["half_x"] + 8), (-CI["half_x"] - 8, -CI["half_x"])):
            ax.add_patch(Rectangle((x0, -60), x1 - x0, 120, color=WALL, lw=0, zorder=1))
        ax.add_patch(Rectangle((-CI["half_x"] - 8, -60), 2 * CI["half_x"] + 16, 8, color=WALL, lw=0, zorder=1))
        ax.axhline(CI["wall_top_z"], color=figstyle.GRID_INK, lw=0.6, ls=":", zorder=0)
    else:
        for x0, x1 in ((CI["half_x"], CI["half_x"] + 8), (-CI["half_x"] - 8, -CI["half_x"])):
            ax.add_patch(Rectangle((x0, -CI["half_y"] - 8), x1 - x0, 2 * CI["half_y"] + 16, color=WALL, lw=0, zorder=1))
        for y0 in (CI["half_y"], -CI["half_y"] - 8):
            ax.add_patch(Rectangle((-CI["half_x"] - 8, y0), 2 * CI["half_x"] + 16, 8, color=WALL, lw=0, zorder=1))
    ax.add_collection(PolyCollection([V[f][:, [i, j]] for f in F], facecolor=APPLE, edgecolor=APPLE, lw=0.2, alpha=0.9, zorder=3))
    for k, (c, r) in enumerate(S):
        col = C[k % len(C)]
        ax.add_patch(Circle((c[i], c[j]), r, facecolor=col, alpha=0.18, edgecolor=col, lw=1.6, zorder=4))
        ax.add_patch(Circle((c[i], c[j]), r + MG, fill=False, edgecolor=col, lw=1.0, ls="--", zorder=4))
        ax.plot([c[i]], [c[j]], marker="+", color=col, ms=6, zorder=5)
    if B:
        ax.add_patch(Circle((B[0][i], B[0][j]), B[1], fill=False, edgecolor=figstyle.INK, lw=1.2, ls="-.", zorder=5))
    ax.set_xlim(*XL); ax.set_ylim(*(ZL if view == "side" else YL)); ax.set_aspect("equal")
    ax.set_xlabel("crate x (mm)", fontsize=8); ax.set_ylabel("crate z (mm)" if view == "side" else "crate y (mm)", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(True, color=figstyle.GRID_INK, lw=0.4, alpha=0.6)
    return off, S, B


figstyle.use_korean()
fig = plt.figure(figsize=(17, 20.5))
gs = fig.add_gridspec(len(PAIRS), 4, height_ratios=[1] * len(PAIRS), hspace=0.32, wspace=0.18, left=0.04, right=0.99, top=0.925, bottom=0.03)
side = {"definitions": __doc__, "margin_mm": MG, "crate_inner_mm": CI, "panels": {}}
for r_, pk in enumerate(PAIRS):
    for c_, (arm, view) in enumerate((("V4", "side"), ("V4", "top"), ("V5", "side"), ("V5", "top"))):
        ax = fig.add_subplot(gs[r_, c_]); figstyle.style_axes(fig, ax)
        o = R[f"{arm}/H/{arm}_ep{pk.replace('_', '_')}"]
        off, S, B = draw(ax, o, view)
        f = o["fit"][o["query_set"]]
        ax.set_title(f"{arm} ep{pk} · {'side (look along +y)' if view == 'side' else 'top (look down -z)'}", fontsize=10, color=figstyle.INK)
        if view == "side":
            txt = [f"attach t {o['attach_t']} · body cover '{o['body_cover_mode']}' · query set {o['query_set'].split('_')[0]}",
                   "r = " + " / ".join(f"{x[1]:.1f}" for x in S) + " mm" + (" (spheres 0 and 1 coincide)" if len(S) > 1 and np.linalg.norm(S[0][0] - S[1][0]) < 0.5 else "") + (f" · carve-out ball r {B[1]:.1f}" if B else ""),
                   f"apple half-extent x/y/z {' / '.join(f'{v:.1f}' for v in o['apple_half_extent_mm'])} mm",
                   f"lowest sphere bottom {-f['bottom_below_apple_mm']:+.1f} mm vs apple bottom",
                   f"+ margin: apple must be >= {f['apple_bottom_above_floor_needed_mm']:.1f} mm above floor"]
        else:
            txt = [f"x half-span {f['half_extent_x_mm']:.1f} + {MG:.0f} = {f['need_x_mm']:.1f} vs inner {CI['half_x']:.0f} ({f['spare_x_mm']:+.1f})",
                   f"y half-span {f['half_extent_y_mm']:.1f} + {MG:.0f} = {f['need_y_mm']:.1f} vs inner {CI['half_y']:.0f} ({f['spare_y_mm']:+.1f})",
                   f"cover radius about apple centre {f['cover_radius_about_apple_mm']:.1f} mm"]
        ax.text(0.01, 0.99, "\n".join(txt), transform=ax.transAxes, fontsize=7.2, va="top", color=figstyle.INK,
                bbox=dict(facecolor=figstyle.SURFACE, alpha=0.85, lw=0))
        side["panels"][f"{arm}_{pk}_{view}"] = {"key": o["key"], "attach_t": o["attach_t"], "query_set": o["query_set"], "body_cover_mode": o["body_cover_mode"],
                                                "apple_centre_crate_mm": off.tolist(), "spheres_crate_mm": [[c.tolist(), rr] for c, rr in S],
                                                "carve_ball_crate_mm": [B[0].tolist(), B[1]] if B else None, "fit": f}
h = [Line2D([], [], color=APPLE, lw=8, label="apple collision mesh (projected)"),
     Line2D([], [], color=C[0], lw=1.6, label="held query sphere 0 (body cover), r"), Line2D([], [], color=C[1], lw=1.6, label="held query sphere 1"),
     Line2D([], [], color=C[2], lw=1.6, label="held query sphere 2"), Line2D([], [], color=C[3], lw=1.6, label="held query sphere 3"),
     Line2D([], [], color=figstyle.INK_2, lw=1.0, ls="--", label=f"r + {MG:.0f} mm destination margin shell"),
     Line2D([], [], color=figstyle.INK, lw=1.2, ls="-.", label="closing carve-out ball (last closing chunk)"),
     Line2D([], [], color=WALL, lw=8, label="crate collision walls / floor (inner x +-82, y +-142, floor top -52, wall top 60)")]
fig.legend(handles=h, loc="upper center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("T43 Z1 — held query spheres at attach, equal scale (1 mm = 1 mm in every panel). Apple placed at the crate centre resting on the floor; "
             "spheres at their attach-row offsets from the apple", fontsize=11.5, color=figstyle.INK, y=0.995)
out = FIG / "t43z-held-sphere-closeup.png"
assert not out.exists() or "--overwrite" in sys.argv, out
fig.savefig(out, dpi=105, facecolor=figstyle.SURFACE)
side.update(figure=str(out), script=__file__, sources=["outputs/verify/T43/Z/raw/z1_held.json"])
json.dump(side, open(FIG / "t43z-held-sphere-closeup.json", "w"), indent=1, default=float)
print(out)
