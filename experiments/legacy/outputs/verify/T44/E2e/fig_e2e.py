"""T44e V-E2e figures (verifier): scene / graph / table + .json sidecars -> AG3S/ag3s/docs/figures/t44/t44e-*."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, glob, pathlib, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle as FS
from AG3S.ag3s.experiments.live import exclusion_e2 as H

D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2e'))
E2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t44')); FIG.mkdir(parents=True, exist_ok=True)
FS.use_korean()
A = json.load(open(D / "agg_e2e.json"))
GM = json.load(open(D / "grad_mismatch.json"))
RQ = {p.stem: json.load(open(p)) for p in sorted((D / "requery").glob("*.json"))}
I = "gt_label"; MS = ("ball", "bo_d3"); VS = ("off", "guard", "keep", "exclude", "exclude_target")
C = FS.CATEGORICAL
VCOL = {"off": C[7], "guard": C[3], "keep": C[0], "exclude": C[1], "exclude_target": C[2]}
VLAB = {"off": "coarse off", "guard": "guard (A5)", "keep": "keep (A7, C1b)", "exclude": "exclude (A8)", "exclude_target": "exclude_target (A8b)"}
VSH = {"off": "off", "guard": "guard", "keep": "keep", "exclude": "exclude", "exclude_target": "excl_target"}


def g(s, ident, m, v, k, default=0):
    return A["sets"][s][ident]["methods"][m][v].get(k, default)


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
        if "sl_xy10_in_dis" not in z.files: continue
        ind = z["sl_xy10_in_dis"]
        if not ind.any(): continue
        ex = (z["sl_xy10_bo_d3|exclude"][ind] > 0).sum(); et = (z["sl_xy10_bo_d3|exclude_target"][ind] > 0).sum()
        kp = (z["sl_xy10_bo_d3|keep"][ind] > 0).sum()
        s = int(ex) - int(et)
        if best is None or s > best[0]: best = (s, r, (int(kp), int(ex), int(et), int(ind.sum())))
    return best


def fig_scene():
    R = recs("raw_demo")
    s, r, counts = pick_frame(R)
    z = np.load(r["_npz"]); ball = z["ball_gt_label"]; ax_ = z["sl_ax"].astype(np.float64); c0 = z["sl_centre"]
    sz = r["identities"][I]["ball"].get("support_plane_z_m")
    pfs = json.load(open(E2 / "e2_demo_posed_frames.json"))
    pf = next(p for p in pfs if p["episode_index"] == r["source"]["episode_index"] and p["frame"] == r["seq"])
    img, (u, v) = render_rgb(pf, "zed_left")
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.6)); FS.style_axes(fig, axes)
    ax = axes[0, 0]; cu, cv = int(np.clip(u, 120, 520)), int(np.clip(v, 90, 390))
    ax.imshow(img[cv - 90:cv + 90, cu - 120:cu + 120]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"MuJoCo zed_left, {r['key']}\ntarget {pf['target_fruit']} ({pf['used_arm']} arm), nearest other fruit {r['source']['nearest_other_fruit_xy_m']*1e3:.0f} mm (xy)", fontsize=8, color=FS.INK)
    th = np.linspace(0, 2 * np.pi, 100)
    X, Y = np.meshgrid(c0[0] + ax_, c0[1] + ax_, indexing="ij")
    ext = float(ax_[-1])
    # layout
    ax = axes[0, 1]
    ax.contourf(X, Y, z["sl_xy10_in_tgt"].astype(float), levels=[0.5, 1.5], colors=[FS.INK], alpha=0.40)
    ax.contourf(X, Y, z["sl_xy10_in_dis"].astype(float), levels=[0.5, 1.5], colors=[C[3]], alpha=0.45)
    ax.plot(c0[0] + ball[3] * np.cos(th), c0[1] + ball[3] * np.sin(th), color=FS.INK_2, ls="--", lw=0.9)
    ax.set_aspect("equal"); ax.set_xlim(c0[0] - ext, c0[0] + ext); ax.set_ylim(c0[1] - ext, c0[1] + ext)
    ax.set_title("layout: horizontal cut at support plane + 10 mm (top view)\nblack = target hull, orange = neighbouring fruit hull, dashed = target ball", fontsize=8, color=FS.INK)
    ax.set_xlabel("x (m, base)"); ax.set_ylabel("y (m)")
    cmap = FS.sequential_cmap() if hasattr(FS, "sequential_cmap") else plt.get_cmap("Blues")
    vmax = 0.06
    side = {"frame": r["key"], "selection": "demo gt_label frame maximising (neighbour-hull cells with value > 0 under exclude) - (under exclude_target) at plane+10 mm, method bo_d3",
            "cells_in_neighbour_hull_value_gt0_keep_exclude_exclude_target_total": list(counts), "support_plane_z_m": sz,
            "ball": [float(x) for x in ball], "cut_height_above_plane_mm": 10.0, "panels": {}}
    panels = [("off", axes[0, 2]), ("keep", axes[0, 3]), ("exclude", axes[1, 0]), ("exclude_target", axes[1, 1])]
    for v_, ax in panels:
        val = z[f"sl_xy10_bo_d3|{v_}"].astype(np.float64)
        ex = [c0[0] + ax_[0], c0[0] + ax_[-1], c0[1] + ax_[0], c0[1] + ax_[-1]]
        ax.imshow(np.clip(val, 0, vmax).T, origin="lower", extent=ex, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
        negm = (val < 0).astype(float)
        ax.imshow(np.ma.masked_where(negm.T == 0, negm.T), origin="lower", extent=ex, cmap=matplotlib.colors.ListedColormap([C[7]]), alpha=0.85, aspect="equal")
        ax.contour(X, Y, z["sl_xy10_in_tgt"].astype(float), levels=[0.5], colors=[FS.INK], linewidths=1.3)
        ax.contour(X, Y, z["sl_xy10_in_dis"].astype(float), levels=[0.5], colors=[C[3]], linewidths=1.5)
        ax.plot(c0[0] + ball[3] * np.cos(th), c0[1] + ball[3] * np.sin(th), color=FS.INK_2, ls="--", lw=0.8)
        ind = z["sl_xy10_in_dis"]; tgt = z["sl_xy10_in_tgt"]
        n_d = int((val[ind] > 0).sum()); n_t = int((val[tgt] > 0).sum())
        side["panels"][v_] = {"neighbour_hull_cells": int(ind.sum()), "neighbour_hull_cells_value_gt0": n_d, "target_hull_cells": int(tgt.sum()),
                              "target_hull_cells_value_gt0": n_t, "neg_cells_total": int((val < 0).sum())}
        ax.set_title(f"{VLAB[v_]}\nneighbour hull cells with value > 0: {n_d}/{int(ind.sum())}; target hull cells > 0: {n_t}/{int(tgt.sum())}", fontsize=8, color=FS.INK)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    # differences
    for ax, (a_, b_) in ((axes[1, 2], ("exclude", "exclude_target")), (axes[1, 3], ("exclude_target", "keep"))):
        dv = (z[f"sl_xy10_bo_d3|{a_}"].astype(np.float64) - z[f"sl_xy10_bo_d3|{b_}"].astype(np.float64)) * 1e3
        ex = [c0[0] + ax_[0], c0[0] + ax_[-1], c0[1] + ax_[0], c0[1] + ax_[-1]]
        im = ax.imshow(np.clip(dv, 0, 30).T, origin="lower", extent=ex, cmap=cmap, vmin=0, vmax=30, aspect="equal")
        ax.contour(X, Y, z["sl_xy10_in_tgt"].astype(float), levels=[0.5], colors=[FS.INK], linewidths=1.3)
        ax.contour(X, Y, z["sl_xy10_in_dis"].astype(float), levels=[0.5], colors=[C[3]], linewidths=1.5)
        ax.set_title(f"value difference {VSH[a_]} - {VSH[b_]} (mm, clipped 0-30)\n(where the support seed was removed {'in the neighbour footprint' if b_ == 'exclude_target' else 'under the target'})", fontsize=8, color=FS.INK)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        side["panels"][f"diff|{a_}-{b_}"] = {"cells_gt_1mm": int((dv > 1).sum()), "max_mm": float(dv.max())}
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax * 1e3)); sm.set_array([])
    fig.colorbar(sm, ax=axes[:, 2:], fraction=0.012, pad=0.01, label="target-free distance, bo_d3 (mm, clipped 0-60); red = negative")
    fig.suptitle("T44e V-E2e  bottom cross-section of target and neighbouring fruit at support plane + 10 mm: keep vs exclude vs exclude_target (2.5 mm resolution)", fontsize=10, color=FS.INK)
    fig.savefig(FIG / "t44e-scene.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    (FIG / "t44e-scene.json").write_text(json.dumps(side, indent=1))


def grouped(ax, groups, series, vals, ylabel, fmt="{:.0f}", w=0.16, rot=0, cols=None, labs=None):
    xs = np.arange(len(groups))
    for i, v in enumerate(series):
        off = (i - (len(series) - 1) / 2) * w
        y = [vals(gi, v) for gi in range(len(groups))]
        ax.bar(xs + off, y, w, color=(cols or VCOL)[v], label=(labs or VLAB)[v])
        for x, yy in zip(xs + off, y):
            ax.text(x, yy, fmt.format(yy), ha="center", va="bottom", fontsize=5.5, color=FS.INK, rotation=rot)
    ax.set_xticks(xs); ax.set_xticklabels(groups, fontsize=7); ax.set_ylabel(ylabel, fontsize=7)


def fig_graph():
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.4)); FS.style_axes(fig, axes)
    side = {}
    # (a) D frames
    ax = axes[0, 0]; gr = [f"{s}\n{m}" for s in ("demo", "lb") for m in MS]
    gl = [(s, m) for s in ("demo", "lb") for m in MS]
    grouped(ax, gr, VS, lambda i, v: g(gl[i][0], I, gl[i][1], v, "D_clr_ge_margin_frames"), "frames with a neighbour-interior point, clearance >= 10 mm")
    ax.legend(fontsize=6); ax.set_title("(a) neighbour-fruit interior optimistic frames (lower = safer)", fontsize=8, color=FS.INK)
    side["a_D_frames"] = {f"{s}|{m}|{v}": g(s, I, m, v, "D_clr_ge_margin_frames") for s, m in gl for v in VS}
    # (b) D points split
    ax = axes[0, 1]; xs = np.arange(2)
    w = 0.16
    for i, v in enumerate(VS):
        off = (i - 2) * w
        band = [g("demo", I, m, v, "D_clr_band0_15_pts") for m in MS]; ab = [g("demo", I, m, v, "D_clr_above15_pts") for m in MS]
        ax.bar(xs + off, band, w, color=VCOL[v], label=VLAB[v])
        ax.bar(xs + off, ab, w, bottom=band, color=VCOL[v], alpha=0.35, hatch="//", edgecolor=FS.INK_2, linewidth=0.3)
        for x, b_, a_ in zip(xs + off, band, ab): ax.text(x, b_ + a_, f"{b_}\n+{a_}", ha="center", va="bottom", fontsize=5.5, color=FS.INK)
        side[f"b_D_pts|{v}"] = {m: [band[j], ab[j]] for j, m in enumerate(MS)}
    ax.set_xticks(xs); ax.set_xticklabels(["demo ball", "demo bo_d3"], fontsize=7); ax.set_ylabel("neighbour-interior points, clearance >= 10 mm", fontsize=7)
    ax.set_title("(b) solid = 0-15 mm band above support, hatched = above 15 mm", fontsize=8, color=FS.INK)
    # (c) zero level
    ax = axes[0, 2]; gr = ["ball centre", "ring at fit r", "ring at r+10 mm"]; keys = ("centre", "ringfit", "ringout")
    grouped(ax, gr, VS, lambda i, v: A["sets"]["demo"][I]["methods"]["bo_d3"]["zero_level_mm"][v][keys[i]]["median"], "zero level above support (mm, median of 64 demo frames)", fmt="{:.1f}")
    ax.set_title("(c) support-surface zero level of the served target-free field, bo_d3", fontsize=8, color=FS.INK)
    side["c_zero_level_demo"] = {m: {v: {k: A["sets"]["demo"][I]["methods"][m]["zero_level_mm"][v][k]["median"] for k in keys} for v in VS} for m in MS}
    side["c_zero_level_lb"] = {m: {v: {k: A["sets"]["lb"][I]["methods"][m]["zero_level_mm"][v][k]["median"] for k in keys} for v in VS} for m in MS}
    # (d) pre-grasp cross
    ax = axes[0, 3]; gr = [f"{m}\n{k}" for m in MS for k in ("queries", "frames")]
    kk = [(m, k) for m in MS for k in ("q_cross", "q_cross_frames")]
    grouped(ax, gr, VS[1:], lambda i, v: g("demo_last2", I, kk[i][0], v, kk[i][1]), "authorized spheres pushed below margin (demo last 2 = 32 frames)")
    ax.set_title("(d) before grasp: off - r >= 10 mm and on - r < 10 mm", fontsize=8, color=FS.INK)
    side["d_last2_cross"] = {f"{m}|{v}": [g("demo_last2", I, m, v, "q_cross"), g("demo_last2", I, m, v, "q_cross_frames")] for m in MS for v in VS[1:]}
    # (e) T neg + TA neg
    ax = axes[1, 0]; gr = ["T interior\n(demo)", "T interior\n(lb)", "TA 0-10 mm\n(demo)", "TA 0-10 mm\n(lb)"]
    ek = [("demo", "T_neg"), ("lb", "T_neg"), ("demo", "TA_neg"), ("lb", "TA_neg")]
    grouped(ax, gr, VS, lambda i, v: g(ek[i][0], I, "bo_d3", v, ek[i][1]), "points with value < 0 (bo_d3)")
    ax.set_yscale("symlog", linthresh=1000)
    ax.set_title("(e) negatives: target interior (want kept) and table top 0-10 mm outside every fruit (want small)", fontsize=8, color=FS.INK)
    side["e_neg"] = {f"{s}|{k}|{v}": g(s, I, "bo_d3", v, k) for s, k in ek for v in VS}
    side["e_neg_ball"] = {f"{s}|{k}|{v}": g(s, I, "ball", v, k) for s, k in ek for v in VS}
    # (f) build ms
    ax = axes[1, 1]
    T = A.get("timing")
    if T:
        t = T["all"]; xs = np.arange(2)
        series = ("off", "keep", "exclude", "exclude_target")
        def tot(m, v):
            return t[m]["fine"]["p90"] if v == "off" else t[m][f"on_total_{v}"]["p90"]
        for i, v in enumerate(series):
            off = (i - 1.5) * 0.2
            y = [tot(m, v) for m in MS]
            ax.bar(xs + off, y, 0.2, color=VCOL[v], label=VLAB[v])
            for x, yy in zip(xs + off, y): ax.text(x, yy, f"{yy:.1f}", ha="center", va="bottom", fontsize=6, color=FS.INK)
        ax.set_xticks(xs); ax.set_xticklabels(list(MS)); ax.set_ylabel("target-free build per frame (ms, p90): fine + coarse + support evidence", fontsize=7); ax.legend(fontsize=6)
        ax.set_title("(f) build ms, dedicated timing pass (112 frames x 10 reps; pod load high)", fontsize=8, color=FS.INK)
        side["f_build_ms_p90_total"] = {m: {v: tot(m, v) for v in series} for m in MS}
        side["f_coarse_ms_median_p90"] = {m: {v: [t[m][f"coarse_{v}"]["median"], t[m][f"coarse_{v}"]["p90"]] for v in ("guard", "keep", "exclude", "exclude_target")} for m in MS}
        side["f_pod_load"] = t["pod_load"]
    # (g) re-query
    ax = axes[1, 2]
    pairs = list(RQ.keys())
    for k, p in enumerate(pairs):
        w_ = RQ[p]["query"]["refined_chunk"]["worst"]["finger_rows0_7"]
        for v, mk in (("off", "o"), ("keep", "s"), ("exclude", "^"), ("exclude_target", "D")):
            ax.scatter(w_[v]["clr_mm"], k, color=VCOL[v], marker=mk, s=38, label=VLAB[v] if k == 0 else None, zorder=3)
        ax.plot([w_["off"]["clr_mm"], w_["keep"]["clr_mm"]], [k, k], color=FS.GRID_INK, lw=1, zorder=1)
    ax.set_yticks(range(len(pairs))); ax.set_yticklabels(pairs, fontsize=7); ax.invert_yaxis()
    ax.axvline(10, color=FS.INK_2, ls=":", lw=0.8)
    ax.set_xlabel("finger worst row clearance d - r (mm), refined chunk rows 0-7, same q", fontsize=7); ax.legend(fontsize=6)
    ax.set_title("(g) offline re-query of the 7 C1b/C1a pairs at the first material chunk", fontsize=8, color=FS.INK)
    side["g_requery_worst_finger_clr_mm"] = {p: {v: RQ[p]["query"]["refined_chunk"]["worst"]["finger_rows0_7"][v]["clr_mm"] for v in ("recorded_off", "recorded_keep", "off", "keep", "exclude", "exclude_target")} for p in pairs}
    # (h) implied lift
    ax = axes[1, 3]
    series = ("keep", "exclude", "exclude_target")
    xs = np.arange(len(pairs))
    for i, v in enumerate(series):
        y = [RQ[p]["query"]["refined_chunk"]["worst"]["finger_rows0_7"][v]["implied_lift_vs_off_mm"] for p in pairs]
        ax.bar(xs + (i - 1) * 0.26, y, 0.26, color=VCOL[v], label=VLAB[v])
    ax.set_xticks(xs); ax.set_xticklabels(pairs, rotation=40, ha="right", fontsize=6.5); ax.set_ylabel("implied lift: clearance(off) - clearance(variant) at that worst row (mm)", fontsize=7)
    ax.legend(fontsize=6); ax.set_title("(h) how much more the TO must lift than with the coarse tier off", fontsize=8, color=FS.INK)
    side["h_implied_lift_mm"] = {p: {v: RQ[p]["query"]["refined_chunk"]["worst"]["finger_rows0_7"][v]["implied_lift_vs_off_mm"] for v in series} for p in pairs}
    fig.suptitle("T44e V-E2e  coarse tier support handling: off / guard / keep / exclude / exclude_target (demo 64 + lb 48 frames, identity gt_label)", fontsize=10, color=FS.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "t44e-graph.png", dpi=130); plt.close(fig)
    (FIG / "t44e-graph.json").write_text(json.dumps(side, indent=1))


def fig_table():
    def sums(m, v, k): return sum(g(s, I, m, v, f"{x}_{k}") for s in ("demo", "lb") for x in ("D", "T", "TB", "TA", "q", "grid"))
    spec = [
        ("target interior negatives, points (demo)", lambda m, v: g("demo", I, m, v, "T_neg")),
        ("neighbour interior clr >= 10 mm: frames (demo /64)", lambda m, v: g("demo", I, m, v, "D_clr_ge_margin_frames")),
        ("   points total (demo)", lambda m, v: g("demo", I, m, v, "D_clr_ge_margin_pts")),
        ("   points in 0-15 mm band above support (demo)", lambda m, v: g("demo", I, m, v, "D_clr_band0_15_pts")),
        ("   frames with band points (demo)", lambda m, v: g("demo", I, m, v, "D_clr_band0_15_frames")),
        ("   points above 15 mm (demo)", lambda m, v: g("demo", I, m, v, "D_clr_above15_pts")),
        ("neighbour interior clr >= 10 mm: frames (lb /48)", lambda m, v: g("lb", I, m, v, "D_clr_ge_margin_frames")),
        ("TB table band optimistic flips (demo+lb)", lambda m, v: g("demo", I, m, v, "TB_optimistic_flip") + g("lb", I, m, v, "TB_optimistic_flip")),
        ("TB band negatives (demo)", lambda m, v: g("demo", I, m, v, "TB_neg")),
        ("TA 0-10 mm negatives (demo)", lambda m, v: g("demo", I, m, v, "TA_neg")),
        ("TA 0-10 mm mean value - off (mm, demo)", lambda m, v: round(g("demo", I, m, v, "TA_mean_delta_vs_off_sum_mm") / g("demo", I, m, v, "TA_n", 1), 2)),
        ("grasp-pre cross, last 2: queries / frames (/32)", lambda m, v: f"{g('demo_last2', I, m, v, 'q_cross')} / {g('demo_last2', I, m, v, 'q_cross_frames')}"),
        ("grasp-pre cross, all 64: queries / frames", lambda m, v: f"{g('demo', I, m, v, 'q_cross')} / {g('demo', I, m, v, 'q_cross_frames')}"),
        ("grasp-pre cross, lb 48: queries / frames", lambda m, v: f"{g('lb', I, m, v, 'q_cross')} / {g('lb', I, m, v, 'q_cross_frames')}"),
        ("new > old (D,T,TB,TA,q,grid; demo+lb)", lambda m, v: sums(m, v, "new_gt_old")),
        ("new < min(old, d) (same)", lambda m, v: sums(m, v, "new_lt_min_old_d")),
        ("zero level mm: centre / ring fit / ring r+10 (demo median)", lambda m, v: " / ".join(f"{A['sets']['demo'][I]['methods'][m]['zero_level_mm'][v][k]['median']:g}" for k in ("centre", "ringfit", "ringout"))),
    ]
    hdr = ["metric (gt_label)"] + [f"{m}\n{VSH[v]}" for m in MS for v in VS]
    cells = [[name] + [str(f(m, v)) for m in MS for v in VS] for name, f in spec]
    cells.append(["value-gradient cos median, same branch"] + ["-" if v == "off" else str(A["sets"]["demo"][I]["methods"][m]["gradient"][v]["cos_median_same_branch"]) for m in MS for v in VS])
    cells.append(["gradient != coarse-layer grad: points (nonzero value)"] + ["-" if v == "off" else f"{GM[f'{m}|{v}']['mismatch_total']} ({GM[f'{m}|{v}']['mismatch_with_nonzero_value']})" for m in MS for v in VS])
    cells.append(["answer value != target_free_distance (max abs, m)"] + ["-" if v == "off" else str(A["sets"]["demo"][I]["methods"][m]["gradient"][v]["value_answer_vs_target_free_distance_max_absdiff_m"]) for m in MS for v in VS])
    T = A.get("timing")
    if T:
        cells.append(["coarse build ms median / p90 (dedicated pass)"] + ["-" if v == "off" else f"{T['all'][m]['coarse_' + v]['median']:.1f} / {T['all'][m]['coarse_' + v]['p90']:.1f}" for m in MS for v in VS])
        cells.append(["fine + coarse (+evidence) p90 total, ms"] + [f"{T['all'][m]['fine']['p90']:.1f}" if v == "off" else f"{T['all'][m]['on_total_' + v]['p90']:.1f}" for m in MS for v in VS])
    # re-query table
    ab = {"target_free": "TF", "target_free_coarse": "TFC", "coarse": "C", "fine": "F"}
    rq_hdr = ["pair (C1b outcome)", "chunk seq", "off: clr mm (tier)", "keep: clr (tier) lift", "exclude: clr (tier) lift", "exclude_target: clr (tier) lift", "recorded keep: clr (tier)"]
    rq = []
    for p, r in RQ.items():
        w_ = r["query"]["refined_chunk"]["worst"]["finger_rows0_7"]
        f = lambda v: f"{w_[v]['clr_mm']:.1f} ({ab.get(w_[v]['tier'], w_[v]['tier'])}) {w_[v]['implied_lift_vs_off_mm']:+.1f}"
        rq.append([f"{p} ({r['failure_mode']['C1b']})", str(r["first_mat_seq"]), f"{w_['off']['clr_mm']:.1f} ({ab.get(w_['off']['tier'])})", f("keep"), f("exclude"), f("exclude_target"),
                   f"{w_['recorded_keep']['clr_mm']:.1f} ({ab.get(w_['recorded_keep']['tier'])})"])
    fig, axs = plt.subplots(2, 1, figsize=(16, 0.36 * (len(cells) + len(rq) + 6) + 1.5), gridspec_kw={"height_ratios": [len(cells) + 2, len(rq) + 2]})
    for ax in axs: ax.axis("off")
    tb = axs[0].table(cellText=cells, colLabels=hdr, loc="center", cellLoc="center", colWidths=[0.30] + [0.07] * 10)
    tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.4)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(FS.GRID_INK)
        if i == 0: c.set_facecolor(FS.GRID_INK); c.set_text_props(color=FS.INK)
        elif j == 0: c.set_text_props(ha="left"); c._loc = "left"
        if i > 0 and j in (4, 5, 9, 10): c.set_facecolor("#e8f0fa" if j in (5, 10) else "#fbeee6")
    axs[0].set_title("T44e V-E2e  demo 64 + lb 48 frames. Shaded: orange = exclude (A8), light blue = exclude_target (A8b).", fontsize=9, color=FS.INK)
    tb2 = axs[1].table(cellText=rq, colLabels=rq_hdr, loc="center", cellLoc="center", colWidths=[0.22, 0.07, 0.13, 0.15, 0.15, 0.17, 0.13])
    tb2.auto_set_font_size(False); tb2.set_fontsize(7.5); tb2.scale(1, 1.5)
    for (i, j), c in tb2.get_celld().items():
        c.set_edgecolor(FS.GRID_INK)
        if i == 0: c.set_facecolor(FS.GRID_INK); c.set_text_props(color=FS.INK)
    axs[1].set_title("offline re-query, first material chunk of each C1b/C1a pair: finger worst row (refined chunk rows 0-7), clearance d - r in mm, tier (TF target_free, TFC target_free_coarse), lift = clearance(off) - clearance(variant)", fontsize=8, color=FS.INK)
    fig.savefig(FIG / "t44e-table.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    (FIG / "t44e-table.json").write_text(json.dumps({"header": hdr, "rows": cells, "requery_header": rq_hdr, "requery_rows": rq}, indent=1))


if __name__ == "__main__":
    which = sys.argv[1:] or ["scene", "graph", "table"]
    for w in which: {"scene": fig_scene, "graph": fig_graph, "table": fig_table}[w]()
