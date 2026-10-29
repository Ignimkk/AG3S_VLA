"""T43 P2 figure (layout, top-down, reset scene): per episode: crate collision geoms (floor, walls, handles; grey), the reset
gripper of the used arm (EE_BODY + fingers, pink; xy convex hull of 400 support points per geom) and of the idle arm (light),
recorded apple-centre paths over [lift_t, place_t] (E0, E3b), apple at t0 (dotted circle r 35 mm), the T43 P hurdle (dashed
grey-violet, 5 T40 episodes) and the T43 P2 overbar (red: bar capsule drawn with its true width 2r between the leg axes, legs =
circles r 12 mm).  Text: bar height h (bar centre above the table), L, a' (min robot clearance over reset/warmup/t0), d, crate
clearance, drop gap, free height.  -> AG3S/ag3s/docs/figures/t43/t43p2-layout.png + .json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, pathlib, sys
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from scipy.spatial import ConvexHull
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P2/p')); sys.path.insert(0, _ag3s_workspace_path(''))
import p2lib as F, gjk
from AG3S.ag3s.experiments.common import figstyle
figstyle.use_korean()
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
DIRS = gjk._fib(400)
C = figstyle.CATEGORICAL
RED = "#d62728"


def hull_xy(c, g):
    s = gjk.make_support(c.m, c.d, int(g)); P = np.array([s(v) for v in DIRS])
    h = ConvexHull(P[:, :2]); return P[h.vertices, :2], float(P[:, 2].min())


def hurdle(ax, e, col):
    u = np.array([math.cos(e["yaw"]), math.sin(e["yaw"])]); p = np.array([e["x"], e["y"]])
    a, b = p - 0.12 * u, p + 0.12 * u
    ax.plot([a[0], b[0]], [a[1], b[1]], color=col, ls="--", lw=1.4, label=f"T43 P hurdle_0 (h {e['h']:.2f})")
    for q in (a, b):
        ax.add_patch(Circle(q, 0.035, fill=False, ec=col, ls="--", lw=0.8))


def overbar(ax, e):
    u = np.array([math.cos(e["yaw"]), math.sin(e["yaw"])]); n = np.array([-u[1], u[0]]); p = np.array([e["x"], e["y"]])
    a, b = p - e["L"] / 2 * u, p + e["L"] / 2 * u; r = e["r"]
    poly = np.array([a + r * n, b + r * n, b - r * n, a - r * n])
    ax.add_patch(Polygon(poly, closed=True, fc=RED, ec=RED, alpha=0.85, label=f"T43 P2 overbar_0 (h {e['h']:.2f}, L {e['L']:.2f}, r {r * 1000:.0f} mm)"))
    for q in (a, b):
        ax.add_patch(Circle(q, F.LEG_R, fc="white", ec=RED, lw=1.2))
    return a, b


tags = F.TAGS
fig, axes = plt.subplots(1, len(tags), figsize=(4.0 * len(tags), 7.4)); figstyle.style_axes(fig, axes)
data = {"definitions": __doc__, "episodes": {}}
for ax, tag in zip(axes, tags):
    c = F.Ctx(tag); c.restore(); m, d = c.m, c.d
    sel = json.loads((F.RAW / f"select_p2_{tag}.json").read_text()); e = sel.get("chosen")
    rec = {"crate": [], "gripper_used": [], "gripper_idle": []}
    for g in c.CRATE:
        P, _ = hull_xy(c, g); ax.add_patch(Polygon(P, closed=True, fc=figstyle.GRID_INK, ec=figstyle.INK_2, lw=0.6, alpha=0.7))
        rec["crate"].append({"geom": c.gname(g), "hull_xy": P.round(4).tolist()})
    S_used = "L" if c.arm == "left" else "R"; S_idle = "R" if S_used == "L" else "L"
    zmin = []
    for S, key, alpha in ((S_used, "gripper_used", 0.6), (S_idle, "gripper_idle", 0.25)):
        names = (f"EE_BODY_{S}", f"ee_finger_{S.lower()}1", f"ee_finger_{S.lower()}2")
        gg = [g for g in c.ROBOT_ALL if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) in names]
        for g in gg:
            P, z = hull_xy(c, g)
            if key == "gripper_used":
                zmin.append(z)
            ax.add_patch(Polygon(P, closed=True, fc=C[4], ec=C[4], lw=0.6, alpha=alpha))
            rec[key].append({"geom": c.gname(g), "hull_xy": P.round(4).tolist(), "z_min_above_table_m": z - F.TABLE_Z})
    for k, cond in enumerate(("E0", "E3b")):
        R = c.runs[cond]; mk = (R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"]); A = R["Q"][mk][:, 38:41]
        ax.plot(A[:, 0], A[:, 1], color=C[k], lw=1.4, label=f"apple path {cond}")
        rec[f"path_{cond}"] = A[:, :2].round(4).tolist()
    a0 = c.runs["E0"]["Q"][0][38:40]; ax.add_patch(Circle(a0, 0.035, fill=False, ec=figstyle.INK_2, ls=":", lw=1.0))
    old = F.RAW.parent.parent / "P" / "raw" / f"select_p_{tag}.json"
    if old.exists():
        o = json.loads(old.read_text()); oe = o.get("chosen") or (o.get("tradeoff") or {}).get("d_best_given_abc")
        if oe:
            hurdle(ax, oe, C[6]); rec["T43P_hurdle"] = {k: oe[k] for k in ("x", "y", "yaw", "h")}
    cx, cy = sel.get("crate_xy") or json.loads((F.RAW / f"p2search_{tag}.json").read_text())["crate_xy"]
    side = 1.0 if sel["side"] > 0 else -1.0
    if e:
        a, b = overbar(ax, e); rec["overbar"] = e; rec["legs_xy"] = [a.tolist(), b.tolist()]
        txt = (f"a' {e['a_min'] * 1000:.1f}  d {e['d_min'] * 1000:.1f}  crate {e['b_crate_min'] * 1000:.1f} mm\n"
               f"c E0/E3b {e['c_min_per_run']['E0'] * 1000:.1f} / {e['c_min_per_run']['E3b'] * 1000:.1f} mm\n"
               f"free height {e['free_height_over_bar_m'] * 1000:.0f} mm, drop gap {e['drop_gap_m'] * 1000:.0f} mm")
    else:
        txt = "no placement passes a', b, c, d, e"
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=7.5, color=figstyle.INK, va="bottom")
    ax.set_xlim(cx - 0.21, cx + 0.21)
    ax.set_ylim((cy - 0.20, cy + 0.42) if side > 0 else (cy - 0.42, cy + 0.20)); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"ep{tag[:4]} s{tag[5:]} ({c.arm} arm)\nused gripper bottom {min(zmin) - F.TABLE_Z:.3f} m above table", fontsize=9, color=figstyle.INK)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False)
    data["episodes"][tag] = rec
fig.suptitle("T43 P2 layout (top-down, reset scene): grey = crate + handles, pink = reset gripper of the used arm (light = idle arm), "
             "red = overbar_0 across the crate inside the handles (white circles = legs r 12 mm), dashed violet = T43 P hurdle, dotted circle = apple at t0",
             fontsize=10, color=figstyle.INK)
fig.tight_layout(); fig.savefig(OUT / "t43p2-layout.png", dpi=130, facecolor=figstyle.SURFACE, bbox_inches="tight")
(OUT / "t43p2-layout.json").write_text(json.dumps(data, indent=1, default=float)); print(OUT / "t43p2-layout.png")
