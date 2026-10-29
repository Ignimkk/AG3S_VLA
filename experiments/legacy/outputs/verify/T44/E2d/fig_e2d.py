"""T44c-A6 figures (verifier): scene / graph / table + .json sidecars -> AG3S/ag3s/docs/figures/t44/t44c-a6-*."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, glob, pathlib, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle as FS
from AG3S.ag3s.experiments.live import exclusion_e2 as H

D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2d'))
E2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t44')); FIG.mkdir(parents=True, exist_ok=True)
FS.use_korean()
A = json.load(open(D / "agg_e2d.json"))
I = "gt_label"; MS = ("ball", "bo_d3"); VS = ("off", "guard", "plane")
VCOL = {"off": FS.CATEGORICAL[6] if len(FS.CATEGORICAL) > 6 else FS.INK_2, "guard": FS.CATEGORICAL[0], "plane": FS.CATEGORICAL[1]}
VLAB = {"off": "coarse off", "guard": "on + guard (A5)", "plane": "on + plane (A6)"}


def recs(d):
    out = []
    for f in sorted((D / d).glob("*.json")):
        r = json.load(open(f)); r["_npz"] = str(f)[:-5] + ".npz"; out.append(r)
    return out


def render_rgb(pf, cam):
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    import mujoco
    sc = TransportScene(_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml'), settle_steps=0)
    H.pose_posed(sc, pf)
    r = mujoco.Renderer(sc.model, 480, 640); r.update_scene(sc.data, camera=cam); img = r.render().copy()
    f = sc.capture(cam); K, T = np.asarray(f.camera_intrinsics), np.asarray(f.T_base_cam)
    c = H.body_in_base(sc, pf["target_fruit"]); pc = np.linalg.inv(T) @ np.r_[c, 1]
    h, w = f.depth.shape
    u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
    r.close(); sc.close()
    return img, (u * 640 / w, v * 480 / h)


def pick_frame(R):
    best = None
    for r in R:
        z = np.load(r["_npz"])
        k = f"{I}__T_d_ball|guard"
        if k not in z.files: continue
        g = int((z[k] < 0).sum()); p = int((z[f"{I}__T_d_ball|plane"] < 0).sum()); o = int((z[f"{I}__T_d_ball|off"] < 0).sum())
        s = g - p
        if best is None or s > best[0]: best = (s, r, (o, g, p))
    return best


def fig_scene():
    R = recs("raw_demo")
    s, r, (no, ng, np_) = pick_frame(R)
    z = np.load(r["_npz"]); ball = z["ball_gt_label"]; ax_ = z["sl_ax"].astype(np.float64); c0 = z["sl_centre"]
    sz = r["identities"][I]["ball"].get("support_plane_z_m")
    pfs = json.load(open(E2 / "e2_demo_posed_frames.json"))
    pf = next(p for p in pfs if p["episode_index"] == r["source"]["episode_index"] and p["frame"] == r["seq"])
    img, (u, v) = render_rgb(pf, "zed_left")
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.4)); FS.style_axes(fig, axes)
    ax = axes[0, 0]; cu, cv = int(np.clip(u, 120, 520)), int(np.clip(v, 90, 390))
    ax.imshow(img[cv - 90:cv + 90, cu - 120:cu + 120]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"MuJoCo zed_left, {r['key']}\ntarget {pf['target_fruit']} ({pf['used_arm']} arm)", fontsize=8, color=FS.INK)
    th = np.linspace(0, 2 * np.pi, 100)
    # layout (top view): both cuts
    ax = axes[1, 0]
    ext = float(ax_[-1])
    ax.add_patch(plt.Circle((c0[0], c0[1]), ball[3], fill=False, ls="--", color=FS.INK_2, lw=0.9, label="target ball"))
    xy_t = z["sl_xy_in_tgt"]; xy_d = z["sl_xy_in_dis"]
    X, Y = np.meshgrid(c0[0] + ax_, c0[1] + ax_, indexing="ij")
    ax.contourf(X, Y, xy_t.astype(float), levels=[0.5, 1.5], colors=[FS.INK], alpha=0.35)
    ax.contourf(X, Y, xy_d.astype(float), levels=[0.5, 1.5], colors=[FS.CATEGORICAL[3]], alpha=0.35)
    ax.axhline(c0[1], color=FS.CATEGORICAL[0], lw=1.0, label="cut A: x-z plane (y = ball centre) -> row 1")
    ax.set_title(f"top view at z = plane + 10 mm\nblack = target hull, orange = distractor hull;\nrow 2 is this horizontal cut", fontsize=8, color=FS.INK)
    ax.set_aspect("equal"); ax.set_xlim(c0[0] - ext, c0[0] + ext); ax.set_ylim(c0[1] - ext, c0[1] + ext)
    ax.set_xlabel("x (m, base)"); ax.set_ylabel("y (m)"); ax.legend(fontsize=6, loc="lower left")
    cmap = FS.sequential_cmap() if hasattr(FS, "sequential_cmap") else plt.get_cmap("Blues")
    neg_col = FS.CATEGORICAL[1]
    side = {"frame": r["key"], "selection": "demo gt_label frame with largest T-interior negative count guard minus plane (method ball)",
            "T_neg_ball_off_guard_plane_frame": [no, ng, np_], "support_plane_z_m": sz, "ball": [float(x) for x in ball], "panels": {}}
    vmax = 0.06
    for row, (pn, ylab) in enumerate((("xz", "z (m)"), ("xy", "y (m)"))):
        for j, v_ in enumerate(VS):
            ax = axes[row, 1 + j]
            val = z[f"sl_{pn}_ball|{v_}"].astype(np.float64)
            if pn == "xz":
                a1, a2 = c0[0] + ax_, c0[2] + ax_
            else:
                a1, a2 = c0[0] + ax_, c0[1] + ax_
            ex = [a1[0], a1[-1], a2[0], a2[-1]]
            ax.imshow(np.clip(val, 0, vmax).T, origin="lower", extent=ex, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
            negm = (val < 0).astype(float)
            ax.imshow(np.ma.masked_where(negm.T == 0, negm.T), origin="lower", extent=ex, cmap=matplotlib.colors.ListedColormap([neg_col]), alpha=0.85, aspect="equal")
            A1, A2 = np.meshgrid(a1, a2, indexing="ij")
            tm = z[f"sl_{pn}_in_tgt"].astype(float)
            if tm.max() > 0: ax.contour(A1, A2, tm, levels=[0.5], colors=[FS.INK], linewidths=1.2)
            if pn == "xz":
                ax.plot(c0[0] + ball[3] * np.cos(th), c0[2] + ball[3] * np.sin(th), color=FS.INK_2, ls="--", lw=0.8)
                if sz is not None:
                    ax.axhline(sz, color=FS.CATEGORICAL[3], lw=1.0, label="support plane")
                    ax.axhline(sz + 0.020, color=FS.CATEGORICAL[4], lw=1.0, ls="-.", label="plane + 20 mm (guard)")
                    ax.axhline(sz + 0.010, color=FS.CATEGORICAL[2], lw=0.8, ls=":", label="row-2 cut z")
                if j == 0: ax.legend(fontsize=6, loc="upper right")
            else:
                ax.plot(c0[0] + ball[3] * np.cos(th), c0[1] + ball[3] * np.sin(th), color=FS.INK_2, ls="--", lw=0.8)
            nneg = int((val < 0).sum()); nin = int(((val < 0) & (tm > 0)).sum()) if tm.max() > 0 else 0
            side["panels"][f"{pn}|{v_}"] = {"n_neg_cells": nneg, "n_neg_cells_in_target_hull": nin, "n_cells_in_target_hull": int((tm > 0).sum())}
            ax.set_title(f"{'x-z cut' if pn == 'xz' else 'x-y cut'}: {VLAB[v_]}\nred = negative; neg in target hull {nin}/{int((tm>0).sum())} cells", fontsize=8, color=FS.INK)
            ax.set_xlabel("x (m)"); ax.set_ylabel(ylab)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax * 1e3)); sm.set_array([])
    fig.colorbar(sm, ax=axes[:, 1:], fraction=0.015, pad=0.01, label="target-free distance, method ball (mm, clipped 0-60)")
    fig.suptitle("T44c-A6  coarse sign floor: guard (A5) vs plane (A6) at the bottom of the target, 2.5 mm resolution", fontsize=10, color=FS.INK)
    fig.savefig(FIG / "t44c-a6-scene.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    (FIG / "t44c-a6-scene.json").write_text(json.dumps(side, indent=1))


def per_frame_neg(R, m):
    out = []
    for r in R:
        z = np.load(r["_npz"]); k = f"{I}__T_d_{m}|off"
        if k not in z.files: continue
        out.append([int((z[f"{I}__T_d_{m}|{v}"] < 0).sum()) for v in VS])
    return np.array(out)


def fig_graph():
    S = A["sets"]; side = {}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9)); FS.style_axes(fig, axes)
    w = 0.26
    # (a) T neg totals
    ax = axes[0, 0]; xs = np.arange(4); labs = []
    for i, v in enumerate(VS):
        vals = [S[s][I]["methods"][m][v]["T_neg"] for s in ("demo", "lb") for m in MS]
        ax.bar(xs + (i - 1) * w, vals, w, color=VCOL[v], label=VLAB[v])
        for x, y in zip(xs + (i - 1) * w, vals): ax.text(x, y, str(y), ha="center", va="bottom", fontsize=6, color=FS.INK)
        side[f"a_T_neg|{v}"] = vals
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}\n{m}" for s in ("demo", "lb") for m in MS], fontsize=7)
    ax.set_ylabel("target-interior points with value < 0"); ax.legend(fontsize=7)
    ax.set_title("(a) cond 1: negative points inside the target hull (gt_label)", fontsize=8, color=FS.INK)
    # (b) per frame guard vs plane (ball, demo)
    ax = axes[0, 1]
    Rd = recs("raw_demo"); Rl = recs("raw_lb")
    for R, mk, nm in ((Rd, "o", "demo"), (Rl, "s", "lb")):
        P = per_frame_neg(R, "ball")
        ax.scatter(P[:, 0], P[:, 1], s=14, marker=mk, color=VCOL["guard"], label=f"{nm} guard vs off")
        ax.scatter(P[:, 0], P[:, 2], s=14, marker=mk, color=VCOL["plane"], label=f"{nm} plane vs off")
        side[f"b_per_frame_{nm}"] = P.tolist()
    mx = max(ax.get_xlim()[1], ax.get_ylim()[1]); ax.plot([0, mx], [0, mx], color=FS.INK_2, lw=0.8, ls=":")
    ax.set_xlabel("negatives, coarse off (per frame)"); ax.set_ylabel("negatives, coarse on (per frame)"); ax.legend(fontsize=6)
    ax.set_title("(b) per frame, method ball: points below the dotted line = on is better than off", fontsize=8, color=FS.INK)
    # (c) height split
    ax = axes[0, 2]
    xs = np.arange(2)
    for i, v in enumerate(VS):
        vals = [S["demo"][I]["methods"]["ball"][v]["T_neg_h_lt_20mm"], S["demo"][I]["methods"]["ball"][v]["T_neg_h_ge_20mm"]]
        ax.bar(xs + (i - 1) * w, vals, w, color=VCOL[v], label=VLAB[v])
        for x, y in zip(xs + (i - 1) * w, vals): ax.text(x, y, str(y), ha="center", va="bottom", fontsize=6, color=FS.INK)
    ax.set_xticks(xs); ax.set_xticklabels(["height above plane < 20 mm", ">= 20 mm"]); ax.set_ylabel("negative target-interior points (demo, ball)")
    ax.set_title("(c) where the negatives are", fontsize=8, color=FS.INK); ax.legend(fontsize=7)
    # (d) D clearance frames
    ax = axes[1, 0]; xs = np.arange(4)
    for i, v in enumerate(VS):
        vals = [S[s][I]["methods"][m][v]["D_clr_ge_margin_frames"] for s in ("demo", "lb") for m in MS]
        ax.bar(xs + (i - 1) * w, vals, w, color=VCOL[v], label=VLAB[v])
        for x, y in zip(xs + (i - 1) * w, vals): ax.text(x, y, str(y), ha="center", va="bottom", fontsize=6, color=FS.INK)
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}\n{m}" for s in ("demo", "lb") for m in MS], fontsize=7)
    ax.set_ylabel("frames with a distractor-interior point, clearance >= margin"); ax.legend(fontsize=7)
    ax.set_title("(d) cond 2: optimistic answers inside distractors (lower is safer)", fontsize=8, color=FS.INK)
    # (e) q cross at demo_last2
    ax = axes[1, 1]; xs = np.arange(4)
    labs = [("n queries", "q_cross"), ("n frames", "q_cross_frames")]
    cols = []
    for i, v in enumerate(VS):
        vals = [S["demo_last2"][I]["methods"][m][v][k] for m in MS for k in ("q_cross", "q_cross_frames")]
        ax.bar(xs + (i - 1) * w, vals, w, color=VCOL[v], label=VLAB[v])
        for x, y in zip(xs + (i - 1) * w, vals): ax.text(x, y, str(y), ha="center", va="bottom", fontsize=6, color=FS.INK)
        side[f"e_q_cross|{v}"] = vals
    ax.set_xticks(xs); ax.set_xticklabels([f"{m}\n{k}" for m in MS for k in ("queries", "frames")], fontsize=7)
    ax.set_ylabel("authorized spheres pushed below margin (demo last 2 frames)"); ax.legend(fontsize=7)
    ax.set_title("(e) cond 4: before grasp (32 frames), off - r >= m and on - r < m", fontsize=8, color=FS.INK)
    # (f) build ms
    ax = axes[1, 2]
    T = A.get("timing")
    if T:
        t = T["all"]; xs = np.arange(2); keys = (("fine", "off tier"), ("coarse_guard", "coarse guard"), ("coarse_plane", "coarse plane"))
        for i, (k, nm) in enumerate(keys):
            vals = [t[m][k]["p90"] for m in MS]
            ax.bar(xs + (i - 1) * w, vals, w, color=(VCOL["off"], VCOL["guard"], VCOL["plane"])[i], label=nm + " (p90)")
            for x, y in zip(xs + (i - 1) * w, vals): ax.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=6, color=FS.INK)
        ax.set_xticks(xs); ax.set_xticklabels(list(MS)); ax.set_ylabel("tier build time (ms, p90)"); ax.legend(fontsize=7)
        ax.set_title("(f) build ms, dedicated timing pass (112 frames x 10 reps)", fontsize=8, color=FS.INK)
        side["f_build_ms_p90"] = {m: {k: t[m][k]["p90"] for k in ("fine", "coarse_guard", "coarse_plane")} for m in MS}
    fig.suptitle("T44c-A6  coarse sign floor guard vs plane: demo 64 + lb 48 frames, identity gt_label", fontsize=10, color=FS.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "t44c-a6-graph.png", dpi=130); plt.close(fig)
    (FIG / "t44c-a6-graph.json").write_text(json.dumps(side, indent=1))


def fig_table():
    rows = []
    def g(s, ident, m, v, k): return A["sets"][s][ident]["methods"][m][v][k]
    def trip(f): return [f(v) for v in VS]
    spec = [
        ("1  target-interior negatives (demo)", lambda m, v: g("demo", I, m, v, "T_neg")),
        ("1  same, lb", lambda m, v: g("lb", I, m, v, "T_neg")),
        ("1  same, demo, identity gt", lambda m, v: g("demo", "gt", m, v, "T_neg")),
        ("2  distractor frames with clearance >= margin (demo, /64)", lambda m, v: g("demo", I, m, v, "D_clr_ge_margin_frames")),
        ("2  same, points (demo)", lambda m, v: g("demo", I, m, v, "D_clr_ge_margin_pts")),
        ("2  same, lb frames (/48)", lambda m, v: g("lb", I, m, v, "D_clr_ge_margin_frames")),
        ("3  table band (plane-30 mm..plane) optimistic flips (demo)", lambda m, v: g("demo", I, m, v, "TB_optimistic_flip")),
        ("3  table band: value > off (demo)", lambda m, v: g("demo", I, m, v, "TB_new_gt_old")),
        ("3  table top 0-10 mm: value > off (demo)", lambda m, v: g("demo", I, m, v, "TA_new_gt_old")),
        ("3  table top 0-10 mm: negatives (demo)", lambda m, v: g("demo", I, m, v, "TA_neg")),
        ("4  q below margin, queries (demo last 2, cross)", lambda m, v: g("demo_last2", I, m, v, "q_cross")),
        ("4  same, frames (/32)", lambda m, v: g("demo_last2", I, m, v, "q_cross_frames")),
        ("5  new > old (all sets, demo+lb)", lambda m, v: sum(g(s, I, m, v, f"{x}_new_gt_old") for s in ("demo", "lb") for x in ("D", "T", "TB", "TA", "q", "grid"))),
        ("5  new < min(old, d_main) (all sets, demo+lb)", lambda m, v: sum(g(s, I, m, v, f"{x}_new_lt_min_old_d") for s in ("demo", "lb") for x in ("D", "T", "TB", "TA", "q", "grid"))),
    ]
    hdr = ["condition (gt_label)"] + [f"{m}\n{v}" for m in MS for v in VS]
    cells = [[name] + [str(f(m, v)) for m in MS for v in VS] for name, f in spec]
    # gradient / repro rows
    cells.append(["value-gradient: cos median (same branch)"] + [str(A["sets"]["demo"][I]["methods"][m]["gradient"][v]["cos_median_same_branch"]) if v != "off" else "-" for m in MS for v in VS])
    cells.append(["value-gradient: composed grad != coarse-layer grad, D+T+grid pts"] + ["-" if v == "off" else str(json.load(open(D / "grad_mismatch.json"))[f"{m}|{v}"]["mismatch_by_set"]["D"] + json.load(open(D / "grad_mismatch.json"))[f"{m}|{v}"]["mismatch_by_set"]["T"] + json.load(open(D / "grad_mismatch.json"))[f"{m}|{v}"]["mismatch_by_set"]["grid"]) for m in MS for v in VS])
    T = A.get("timing")
    if T:
        cells.append(["coarse build ms p90 (dedicated pass)"] + ["-" if v == "off" else f"{T['all'][m]['coarse_' + v]['p90']:.2f}" for m in MS for v in VS])
    fig, ax = plt.subplots(figsize=(15, 0.42 * (len(cells) + 2) + 1)); ax.axis("off"); FS.style_axes(fig, [ax]) if False else None
    tb = ax.table(cellText=cells, colLabels=hdr, loc="center", cellLoc="center", colWidths=[0.40] + [0.10] * 6)
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.45)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(FS.GRID_INK)
        if i == 0: c.set_facecolor(FS.GRID_INK); c.set_text_props(color=FS.INK)
        elif j == 0: c.set_text_props(ha="left"); c._loc = "left"
        if i > 0 and j in (3, 6): c.set_facecolor("#e8f0fa")
    ax.set_title("T44c-A6 completion conditions 1-5, demo 64 + lb 48 frames (off = coarse layer off; guard = A5; plane = A6). Light blue = plane.", fontsize=9, color=FS.INK)
    fig.savefig(FIG / "t44c-a6-table.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    (FIG / "t44c-a6-table.json").write_text(json.dumps({"header": hdr, "rows": cells}, indent=1))


if __name__ == "__main__":
    which = sys.argv[1:] or ["scene", "graph", "table"]
    for w in which: {"scene": fig_scene, "graph": fig_graph, "table": fig_table}[w]()
