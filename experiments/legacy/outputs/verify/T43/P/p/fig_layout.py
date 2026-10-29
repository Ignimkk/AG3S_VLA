"""T43 P figure 1 (layout, top-down): per episode, at the client's reset scene: crate collision geoms (walls, handles) and the reset
left gripper (EE_BODY_L + fingers; xy convex hull of 400 support points per geom), recorded T39 apple-centre paths [lift_t, place_t]
(E0, E3b), apple at t0 (dotted circle r 35 mm), the T40 F round-1 hurdle (dashed) and the T43 P hurdle (solid; relaxed-c variant dotted when it differs; ep1995: the trade-off candidate d_best_given_abc dash-dot): bar segment
between the post centres + foot circles (r 35 mm).  Text: bar height, a' (min robot clearance over reset/warmup/t0) and d.
-> AG3S/ag3s/docs/figures/t43/t43p-layout.png + .json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, pathlib, sys
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from scipy.spatial import ConvexHull
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p')); sys.path.insert(0, _ag3s_workspace_path(''))
import flib as F, gjk
from AG3S.ag3s.experiments.common import figstyle
KO = figstyle.use_korean()
RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P/raw')); OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
TAGS = F.TAGS
DIRS = gjk._fib(400)
C = figstyle.CATEGORICAL


def hull_xy(c, g):
    s = gjk.make_support(c.m, c.d, int(g)); P = np.array([s(v) for v in DIRS])
    h = ConvexHull(P[:, :2]); return P[h.vertices, :2], float(P[:, 2].min())


def hurdle(ax, e, col, ls, lw, label):
    u = np.array([math.cos(e["yaw"]), math.sin(e["yaw"])]); p = np.array([e["x"], e["y"]])
    a, b = p - 0.12 * u, p + 0.12 * u
    ax.plot([a[0], b[0]], [a[1], b[1]], color=col, ls=ls, lw=lw, label=label, solid_capstyle="round")
    for q in (a, b):
        ax.add_patch(Circle(q, 0.035, fill=False, ec=col, ls=ls, lw=1.0))


fig, axes = plt.subplots(1, 5, figsize=(24, 7.2)); figstyle.style_axes(fig, axes)
data = {"definitions": __doc__, "episodes": {}}
for ax, tag in zip(axes, TAGS):
    c = F.Ctx(tag); c.restore(); m, d = c.m, c.d
    sel = json.loads((RAW / f"select_p_{tag}.json").read_text()); old = json.loads((RAW / f"old_f_{tag}.json").read_text())
    rec = {"crate": [], "gripper": []}
    for g in c.CRATE:
        P, _ = hull_xy(c, g); ax.add_patch(Polygon(P, closed=True, fc=figstyle.GRID_INK, ec=figstyle.INK_2, lw=0.6, alpha=0.7))
        rec["crate"].append({"geom": c.gname(g), "hull_xy": P.round(4).tolist()})
    zmin = []
    for g in c.A_GRIP:
        P, z = hull_xy(c, g); zmin.append(z)
        ax.add_patch(Polygon(P, closed=True, fc=C[4], ec=C[4], lw=0.6, alpha=0.55))
        rec["gripper"].append({"geom": c.gname(g), "hull_xy": P.round(4).tolist(), "z_min_above_table_m": z - F.TABLE_Z})
    for k, cond in enumerate(("E0", "E3b")):
        R = c.runs[cond]; mk = (R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"]); A = R["Q"][mk][:, 38:41]
        ax.plot(A[:, 0], A[:, 1], color=C[k], lw=1.4, label=f"apple path {cond}")
        rec[f"path_{cond}"] = A[:, :2].round(4).tolist()
    a0 = c.runs["E0"]["Q"][0][38:40]; ax.add_patch(Circle(a0, 0.035, fill=False, ec=figstyle.INK_2, ls=":", lw=1.0))
    hurdle(ax, old, C[7], "--", 2.0, f"T40 F h={old['h']:.2f}  a'={old['a_min'] * 1000:.1f} d={old['d_min'] * 1000:.1f} mm")
    rec["old_F"] = {k: old[k] for k in ("x", "y", "yaw", "h", "a_min", "d_min", "b_crate_min")}
    lines = []
    for kind, ls in (("chosen", "-"), ("chosen_relaxed", ":")):
        e = sel.get(kind)
        if not e or (kind == "chosen_relaxed" and sel.get("chosen") and all(abs(sel["chosen"][q] - e[q]) < 1e-9 for q in ("x", "y", "yaw", "h"))):
            continue
        hurdle(ax, e, C[2] if kind == "chosen" else C[6], ls, 2.4,
               f"T43 P{'' if kind == 'chosen' else ' (c relaxed)'} h={e['h']:.2f}  a'={e['a_min'] * 1000:.1f} d={e['d_min'] * 1000:.1f} mm")
        rec[kind] = e
    if not sel.get("chosen"):
        lines.append("no placement passes a', b, c, d, e")
        e = sel["tradeoff"]["d_best_given_abc"]
        if e:
            hurdle(ax, e, C[3], "-.", 2.0, f"trade-off (passes a', b, c; d fails) h={e['h']:.2f}  a'={e['a_min'] * 1000:.1f} d={e['d_min'] * 1000:.1f} mm")
            rec["tradeoff_d_best_given_abc"] = e
    ax.set_xlim(0.33, 0.76); ax.set_ylim(-0.06, 0.44); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"ep{tag[:4]} s{tag[5:]}  gripper bottom {min(zmin) - F.TABLE_Z:.3f} m above table", fontsize=9, color=figstyle.INK)
    if lines:
        ax.text(0.34, -0.05, "\n".join(lines), fontsize=8, color=figstyle.INK_2)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    data["episodes"][tag] = rec
fig.suptitle("T43 P layout (top-down, reset scene): grey = crate + handles, pink = reset left gripper, dashed red = T40 F hurdle, "
             "green = T43 P hurdle (dotted purple: relaxed c; dash-dot yellow: ep1995 trade-off), circles = post feet r 35 mm, dotted circle = apple at t0", fontsize=10, color=figstyle.INK)
fig.tight_layout(); fig.savefig(OUT / "t43p-layout.png", dpi=130, facecolor=figstyle.SURFACE, bbox_inches="tight")
(OUT / "t43p-layout.json").write_text(json.dumps(data, indent=1, default=float)); print(OUT / "t43p-layout.png")
