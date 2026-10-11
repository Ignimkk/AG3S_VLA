"""T40 A figures (rule A): real scene per episode, distance-vs-time graph, placement table + xy layout.
Writes AG3S/ag3s/docs/figures/t40/t40a-*.png with a .json sidecar each.
usage: fig_t40a.py TAG [TAG ...]   (TAG = <episode>_<seed>; reads raw/replay_<TAG>.json, raw/detour_<TAG>.json)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
import mujoco

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as F  # noqa: E402
import t40a_lib as L  # noqa: E402
import render_scene as RS  # noqa: E402

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/A/raw'))
REN = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/A/render'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t40'))
FIG.mkdir(parents=True, exist_ok=True)
KO = F.use_korean()
C_E0, C_E3, C_C = F.CATEGORICAL[0], F.CATEGORICAL[1], F.CATEGORICAL[2]
C_OBS = F.CATEGORICAL[3]


def sidecar(name, obj):
    (FIG / f"{name}.json").write_text(json.dumps(obj, indent=1))


def load(tags):
    allp = {f"{p['episode']}_{p['seed']}": p for p in json.load(open(RAW / "carry_paths_22pairs.json"))["pairs"]}
    D = {}
    for t in tags:
        rp = json.load(open(RAW / f"replay_{t}.json"))
        dt = json.load(open(RAW / f"detour_{t}.json")) if (RAW / f"detour_{t}.json").exists() else None
        D[t] = {"rp": rp, "dt": dt, "ent": allp[t],
                "runs": {c: L.load_run(allp[t]["runs"][c]["run_dir"]) for c in ("E0", "E3b")}}
    return D


def win_path(R, w):
    mk = (R["t"] >= w["lift_t"]) & (R["t"] <= w["place_t"])
    return R["q"][mk, 38:41]


def status(k):
    b = k["best"] or {}
    if b.get("blocks_E0") and b.get("blocks_E3b"):
        return "kept"
    if b.get("blocks_E0") or b.get("blocks_E3b"):
        return "blocks " + ("E0" if b.get("blocks_E0") else "E3b") + " only"
    return "not kept"


# ---------------------------------------------------------------- xy layout panel
def xy_panel(ax, D, t):
    d = D[t]; R = d["runs"]["E3b"]; L.set_q(R["q0"]); mujoco.mj_kinematics(L.m, L.d)
    for g in L.CR_CRATE:  # crate geoms as rotated rectangles (xy footprint)
        c = L.d.geom_xpos[g]; Rm = L.d.geom_xmat[g].reshape(3, 3); h = L.m.geom_size[g]
        corners = [c + Rm @ np.array([sx * h[0], sy * h[1], 0]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        ax.add_patch(Polygon([p[:2] for p in corners], closed=True, fc="#c9b79c", ec="#7a6a55", lw=0.6, alpha=0.6))
    for n in ("apple", "banana", "orange", "pear"):
        p = L.d.xpos[L.BODY(n)].copy()  # copy: matplotlib keeps a reference to the centre array
        ax.add_patch(Circle(tuple(p[:2]), 0.033, fc=F.GRID_INK, ec=F.INK if n == "apple" else F.INK_2, lw=1.0 if n == "apple" else 0.5))
        ax.text(p[0], p[1] + 0.042, n + (" (target)" if n == "apple" else ""), ha="center", fontsize=6, color=F.INK_2)
    for c, col in (("E0", C_E0), ("E3b", C_E3)):
        P = win_path(d["runs"][c], d["rp"]["windows"][c])
        ax.plot(P[:, 0], P[:, 1], color=col, lw=1.4, label=f"{c} apple path")
    for slot, k in d["rp"]["kinds"].items():
        b = k["best"]
        if b is None:
            continue
        x, y = b["xy"]; ls = "-" if status(k) == "kept" else ":"
        if slot == "bollard_0":
            ax.add_patch(Circle((x, y), 0.045, fc="none", ec=C_OBS, lw=0.8, ls=ls))
            ax.add_patch(Circle((x, y), 0.027, fc=C_OBS if ls == "-" else "none", ec=C_OBS, lw=1.0, ls=ls, alpha=0.9))
            ax.text(x + 0.05, y - 0.06, f"bollard ({status(k)})", fontsize=6, color=C_OBS, ha="center")
        else:
            yaw = b["yaw_rad"]; cy, sy = math.cos(yaw), math.sin(yaw)
            pts = [(x + cy * u - sy * v, y + sy * u + cy * v) for u, v in ((-0.104, -0.018), (0.104, -0.018), (0.104, 0.018), (-0.104, 0.018))]
            ax.add_patch(Polygon(pts, closed=True, fc="none", ec=F.CATEGORICAL[4], lw=1.0, ls=ls))
            ax.text(x, y - 0.035, f"divider ({status(k)})", fontsize=6, color=F.CATEGORICAL[4], ha="center")
    ax.set_aspect("equal"); ax.set_xlim(0.36, 0.80); ax.set_ylim(-0.30, 0.40)
    ax.invert_xaxis()  # robot at x = 0 is below/behind; view from the robot's front like the 'front' camera
    ax.set_xlabel("x world (m)"); ax.set_ylabel("y world (m)")
    ax.grid(color=F.GRID_INK, lw=0.4)


# ---------------------------------------------------------------- figure 1: scene
def fig_scene(D):
    side = {}
    for t, d in D.items():
        k = d["rp"]["kinds"]["bollard_0"]; b = k["best"]; slot = "bollard_0"
        paths = [(win_path(d["runs"][c], d["rp"]["windows"][c]), col) for c, col in (("E0", C_E0), ("E3b", C_E3))]
        marks = []
        if b is not None and "replay" in b:
            for c, col in (("E0", C_E0), ("E3b", C_E3)):
                s = b["replay"][c]["_series"]; w = d["rp"]["windows"][c]
                tt = np.array(s["t"]); B = np.array(s["B"]); mk = (tt >= w["lift_t"]) & (tt <= w["place_t"])
                i = int(np.argmin(np.where(mk, B, np.inf)))
                marks.append((d["runs"][c]["q"][i, 38:41], col))
        imgs = RS.render(d["runs"]["E3b"]["q0"], slot, b["xy"], b["yaw_rad"], paths, marks, str(REN / f"scene_{t}"))
        fig = plt.figure(figsize=(15, 4.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.33, 1.33, 1.0])
        for j, nm in enumerate(("front", "top")):
            ax = fig.add_subplot(gs[0, j]); ax.imshow(imgs[nm]); ax.set_axis_off()
            ax.set_title({"front": "front camera (pi05_infer --view front preset)", "top": "oblique top view"}[nm], fontsize=9, color=F.INK_2)
        ax = fig.add_subplot(gs[0, 2]); F.style_axes(fig, ax); xy_panel(ax, D, t)
        ax.legend(fontsize=6, loc="lower left", frameon=False)
        ep, seed = t.split("_")
        kd = d["rp"]["kinds"]["divider_0"]
        fig.suptitle(f"T40 A  ep {ep} seed {seed}  t = 0 scene, bollard_0 at ({b['xy'][0]:.4f}, {b['xy'][1]:.4f}) [{status(k)}];"
                     f"  E0 / E3b apple-centre path lift_t..place_t (blue / orange), dot = deepest row", fontsize=9, color=F.INK)
        F.style_axes(fig, [])
        fig.tight_layout()
        name = f"t40a-scene-{t}"
        fig.savefig(FIG / f"{name}.png", dpi=130, facecolor=F.SURFACE); plt.close(fig)
        side[t] = {"png": str(FIG / f"{name}.png"), "slot": slot, "xy": b["xy"], "yaw_rad": b["yaw_rad"], "status": status(k),
                   "divider_nearest_feasible": kd["best"] and {"xy": kd["best"]["xy"], "yaw_rad": kd["best"]["yaw_rad"], "status": status(kd)},
                   "windows": d["rp"]["windows"], "scene_qpos_source": "E3b planning row 0 qpos (== E0 planning row 0)",
                   "render": "MuJoCo osmesa; paths = apple centre (qpos 38:41) per control row in [lift_t, place_t]"}
        sidecar(name, side[t])
    return side


# ---------------------------------------------------------------- figure 2: distance graph
def fig_dist(D):
    tags = list(D)
    fig, axes = plt.subplots(len(tags), 2, figsize=(12, 2.3 * len(tags)), sharey=True, squeeze=False)
    F.style_axes(fig, axes)
    side = {}
    for i, t in enumerate(tags):
        d = D[t]; k = d["rp"]["kinds"]["bollard_0"]; b = k["best"]
        side[t] = {"slot": "bollard_0", "xy": b["xy"], "status": status(k), "series_mm": {}}
        for j, c in enumerate(("E0", "E3b")):
            ax = axes[i, j]; s = b["replay"][c]["_series"]; w = d["rp"]["windows"][c]
            tt = np.array(s["t"]); lo, hi = w["lift_t"] - 100, w["place_t"] + 40
            mk = (tt >= lo) & (tt <= hi)
            for key, col, lab in (("A", C_E0, "gripper (A)"), ("B", C_E3, "carried apple (B)"), ("C", C_C, "other robot (C)")):
                y = 1e3 * np.array(s[key])
                ax.plot(tt[mk], np.clip(y[mk], -40, 160), color=col, lw=1.1, label=lab)
            ax.axhline(0, color=F.INK, lw=0.6)
            ax.axvspan(w["lift_t"], w["place_t"], color=F.SEQ_BLUE[0], alpha=0.6, lw=0)
            ax.axvline(w["grasp_t"], color=F.INK_2, lw=0.6, ls="--")
            ax.set_ylim(-40, 160)
            ax.set_title(f"ep {t.split('_')[0]} seed {t.split('_')[1]}  {c}  bollard ({status(k)})", fontsize=8, color=F.INK)
            if j == 0:
                ax.set_ylabel("distance to obstacle (mm)", fontsize=7)
            if i == len(tags) - 1:
                ax.set_xlabel("control step t (15 Hz)", fontsize=7)
            side[t]["series_mm"][c] = {"t": tt[mk].tolist(), **{key: (1e3 * np.array(s[key])[mk]).round(2).tolist() for key in ("A", "B", "C")}}
            side[t].setdefault("windows", {})[c] = w
    axes[0, 0].legend(fontsize=6, loc="upper right", frameon=False)
    fig.suptitle("T40 A  replay of recorded T39 qpos (mj_kinematics only): signed distance obstacle <-> gripper / apple / other robot geoms;"
                 " shaded = [lift_t, place_t], dashed = grasp_t, clipped to [-40, 160] mm", fontsize=9, color=F.INK)
    fig.tight_layout()
    fig.savefig(FIG / "t40a-distance.png", dpi=120, facecolor=F.SURFACE); plt.close(fig)
    sidecar("t40a-distance", {"distance": "GJK signed distance (t40a_lib / gjk.py), negative = penetration depth", **side})


# ---------------------------------------------------------------- figure 3: table
def fig_table(D):
    cols = ["ep_seed", "crate xy (m)", "kind", "xy (m) / yaw", "status", "min A/B window E0 (mm)", "min A/B window E3b (mm)",
            "min A approach E0/E3b (mm)", "min C all E0/E3b (mm)", "t0 min clr (mm, item)", "free h over top (mm)",
            "post gap crate / near fruit (mm)"]
    rows, side = [], []
    for t, d in D.items():
        for slot in ("bollard_0", "divider_0"):
            k = d["rp"]["kinds"][slot]; b = k["best"]
            if b is None:
                rows.append([t, "", slot, "none", "-", "", "", "", "", "", "", ""]); continue
            r = b["replay"]
            wa = {c: (1e3 * r[c]["window_lift_to_place"]["A_gripper"]["min_m"], 1e3 * r[c]["window_lift_to_place"]["B_apple"]["min_m"]) for c in r}
            ap = {c: 1e3 * r[c]["approach"]["A_gripper"]["min_m"] if r[c]["approach"] else float("nan") for c in r}
            ca = {c: 1e3 * r[c]["all"]["C_robot"]["min_m"] for c in r}
            t0 = {kk: min(b["t0_clearance_m"][c][kk] for c in b["t0_clearance_m"]) for kk in b["t0_clearance_m"]["E0"]}
            it = min(t0, key=t0.get)
            dt = d["dt"]["kinds"][slot] if d["dt"] else None
            fh = 1e3 * dt["free_height_over_top_m"] if dt else float("nan")
            if dt:
                lat = dt["lateral_t0_tall_part_gap_m"]; nf = min((v, n) for n, v in lat.items() if n not in ("crate",))
                gap = f"{1e3 * lat['crate']:.0f} / {1e3 * nf[0]:.0f} ({nf[1]})"
            else:
                gap = ""
            cc = d["rp"]["crate_xy"]
            rows.append([t, f"{cc[0]:.3f}, {cc[1]:.3f}", slot.split("_")[0], f"{b['xy'][0]:.4f}, {b['xy'][1]:.4f} / {math.degrees(b['yaw_rad']):.1f}",
                         status(k), f"{wa['E0'][0]:.1f} / {wa['E0'][1]:.1f}", f"{wa['E3b'][0]:.1f} / {wa['E3b'][1]:.1f}",
                         f"{ap['E0']:.1f} / {ap['E3b']:.1f}", f"{ca['E0']:.1f} / {ca['E3b']:.1f}", f"{1e3 * t0[it]:.1f} ({it})",
                         f"{fh:.0f}", gap])
            side.append(dict(zip(cols, rows[-1])))
    fig, ax = plt.subplots(figsize=(20, 0.30 * (len(rows) + 3)))
    F.style_axes(fig, ax); ax.set_axis_off()
    tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 1.35)
    for (ri, ci), cell in tb.get_celld().items():
        cell.set_edgecolor(F.GRID_INK)
        if ri == 0:
            cell.set_facecolor(F.SEQ_BLUE[1]); cell.set_text_props(color=F.INK)
        elif rows[ri - 1][4] == "kept":
            cell.set_facecolor("#e8f4ec")
        else:
            cell.set_facecolor(F.SURFACE)
    ax.set_title("T40 A placement summary. A = left gripper (EE_BODY_L + 2 fingers), B = carried apple, C = other robot geoms; "
                 "window = [lift_t, place_t]\nkept = min(A, B) <= 0 mm in the windows of both E0 and E3b, with t0 clearance > 20 mm; "
                 "other rows = best of the 5 listed placements that pass the t0 filters", fontsize=8, color=F.INK)
    fig.tight_layout()
    fig.savefig(FIG / "t40a-table.png", dpi=130, facecolor=F.SURFACE); plt.close(fig)
    sidecar("t40a-table", {"columns": cols, "rows": side})


# ---------------------------------------------------------------- figure 3b: xy layout overview
def fig_layout(D):
    tags = list(D)
    fig, axes = plt.subplots(2, math.ceil(len(tags) / 2), figsize=(4.2 * math.ceil(len(tags) / 2), 9.5), squeeze=False)
    F.style_axes(fig, axes)
    side = {}
    for ax, t in zip(axes.ravel(), tags):
        xy_panel(ax, D, t); ax.set_title(f"ep {t.split('_')[0]} seed {t.split('_')[1]}", fontsize=9, color=F.INK)
        side[t] = {s: (D[t]["rp"]["kinds"][s]["best"] or {}).get("xy") for s in ("bollard_0", "divider_0")}
    axes.ravel()[0].legend(fontsize=6, loc="lower left", frameon=False)
    fig.suptitle("T40 A top-down layout at t = 0 (x axis inverted, robot side at right edge): crate geoms, fruits (r 33 mm), "
                 "apple paths, bollard post r27 / foot r45, divider 208 x 36 mm; solid = kept, dotted = not kept", fontsize=9, color=F.INK)
    fig.tight_layout()
    fig.savefig(FIG / "t40a-layout.png", dpi=120, facecolor=F.SURFACE); plt.close(fig)
    sidecar("t40a-layout", side)


if __name__ == "__main__":
    D = load(sys.argv[1:])
    fig_scene(D); fig_dist(D); fig_table(D); fig_layout(D)
    print("ok")
