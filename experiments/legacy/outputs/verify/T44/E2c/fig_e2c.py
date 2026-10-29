"""T44c D-sign2 figures (verifier): (a) scene, (b) graph, (c) table + .json sidecars.
Writes AG3S/ag3s/docs/figures/t44/t44c-*. Mesh samples for drawing come from the T44E2b npz of the same frame key
(same posed scene; read only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, glob, pathlib, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from AG3S.ag3s.experiments.common import figstyle as FS
from AG3S.ag3s.experiments.live import exclusion_e2 as H

D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2c'))
E2B = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2b'))
E2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2'))
FIG = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t44'))
FIG.mkdir(parents=True, exist_ok=True)
FS.use_korean()
A = json.load(open(D / "agg_e2c.json"))
I = "gt_label"
MARGIN = 0.010
TC = ("outside", "coarse", "fine", "target_free", "window_edge", "static", "target_free_coarse")
NAMES = ("t44c-scene", "t44c-graph", "t44c-table")
for n in NAMES:
    assert not (FIG / f"{n}.png").exists() or "--force" in sys.argv, f"{n}.png exists"


def render_rgb(pf, cam):
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    import mujoco
    sc = TransportScene(_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml'), settle_steps=0)
    H.pose_posed(sc, pf)
    r = mujoco.Renderer(sc.model, 480, 640)
    r.update_scene(sc.data, camera=cam)
    img = r.render().copy()
    f = sc.capture(cam)
    K, T = np.asarray(f.camera_intrinsics), np.asarray(f.T_base_cam)
    c = H.body_in_base(sc, pf["target_fruit"])
    pc = np.linalg.inv(T) @ np.r_[c, 1]
    h, w = f.depth.shape
    u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
    r.close(); sc.close()
    return img, (u * 640 / w, v * 480 / h)


def demo_recs():
    out = []
    for f in sorted((D / "raw_demo").glob("*.json")):
        r = json.load(open(f)); r["_npz"] = str(f)[:-5] + ".npz"; out.append(r)
    return out


def sign_cls(x):
    return np.where(np.abs(x) <= 1e-6, "zero", np.where(x < 0, "neg", "pos"))


COLS = {"neg": FS.CATEGORICAL[0], "zero": FS.CATEGORICAL[6], "pos": FS.CATEGORICAL[1]}


def slab_panel(ax, P, val, mesh, ml, ball, axis_y, yc, vs, title, extra_line=None):
    slab = np.abs(P[:, 1] - yc) <= 0.5 * vs + 1e-9
    p, x = P[slab], val[slab]
    st = sign_cls(x)
    cnt = {}
    for c in ("neg", "zero", "pos"):
        k = st == c
        cnt[c] = int(k.sum())
        if k.any():
            ax.scatter(p[k, 0], p[k, 2], s=30, marker="s", c=COLS[c], lw=0, label=f"{c} ({k.sum()})")
    ms = np.abs(mesh[:, 1] - yc) <= 0.003
    for li, col in ((0, FS.INK), (1, FS.INK_2)):
        k = ms & (ml == li)
        ax.scatter(mesh[k, 0], mesh[k, 2], s=1, c=col)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(ball[0] + ball[3] * np.cos(th), ball[2] + ball[3] * np.sin(th), color=FS.INK_2, ls="--", lw=0.8)
    if extra_line is not None:
        ax.axhline(extra_line[0], color=FS.CATEGORICAL[3], lw=1.0, ls="-.", label=extra_line[1])
    ax.set_aspect("equal"); ax.set_xlabel("x (m, base)"); ax.set_ylabel("z (m)")
    ax.set_title(title, fontsize=8, color=FS.INK); ax.legend(fontsize=6, loc="upper right")
    return cnt, slab


def fig_scene():
    recs = demo_recs()
    # (row 1) frame with most distractor-interior points that are positive under off at depth >= 10 mm
    best = None
    for r in recs:
        z = np.load(r["_npz"])
        if f"{I}__D_d_ball|off" not in z.files:
            continue
        v = int(((z[f"{I}__D_d_ball|off"] > 0) & (z[f"{I}__D_depth"] >= 0.010)).sum())
        if best is None or v > best[0]:
            best = (v, r)
    r1 = best[1]
    # (row 2) grasp_close_start frame (last demo frame) with the largest authorized-sphere decrease (ball)
    best2 = None
    by_ep = {}
    for r in recs:
        by_ep.setdefault(r["source"]["episode_index"], []).append(r["seq"])
    for r in recs:
        if r["seq"] != max(by_ep[r["source"]["episode_index"]]):
            continue
        z = np.load(r["_npz"])
        d = (z[f"{I}__q_d_ball|off"] - z[f"{I}__q_d_ball|on"]).max()
        if best2 is None or d > best2[0]:
            best2 = (float(d), r)
    r2 = best2[1]
    pfs = json.load(open(E2 / "e2_demo_posed_frames.json"))
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.6))
    FS.style_axes(fig, axes)
    side = {"row1_frame": r1["key"], "row1_selection": "demo gt_label frame with most distractor-interior points with ball|off > 0 at depth >= 10 mm",
            "row2_frame": r2["key"], "row2_selection": "demo grasp_close_start frame with the largest authorized-sphere decrease off-on (ball)"}
    th = np.linspace(0, 2 * np.pi, 100)
    # ---- row 1
    z = np.load(r1["_npz"]); zb = np.load(E2B / "raw_demo" / f"{r1['key']}.npz")
    mesh, ml = zb["mesh_pts"], zb["mesh_lab"]
    ball = z[f"{I}__ball"]
    P = z[f"{I}__D_pts"].astype(np.float64)
    vs = 0.005
    near_d = P
    yc = float(np.median(P[:, 1]))
    ep, fr = r1["source"]["episode_index"], r1["seq"]
    pf = next(p for p in pfs if p["episode_index"] == ep and p["frame"] == fr)
    img, (u, v) = render_rgb(pf, "zed_left")
    ax = axes[0, 0]
    cu, cv = int(np.clip(u, 120, 520)), int(np.clip(v, 90, 390))
    ax.imshow(img[cv - 90:cv + 90, cu - 120:cu + 120]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"(1) {r1['key']} zed_left\ntarget {pf['target_fruit']} ({pf['used_arm']} arm)", fontsize=8, color=FS.INK)
    ax = axes[0, 1]
    nr = np.linalg.norm(mesh[:, :2] - ball[:2], axis=1) < 0.15
    for li, col, nm in ((0, FS.INK, "target mesh"), (1, FS.INK_2, "distractor mesh")):
        k = nr & (ml == li)
        ax.scatter(mesh[k, 0], mesh[k, 1], s=1, c=col, label=nm)
    ax.plot(ball[0] + ball[3] * np.cos(th), ball[1] + ball[3] * np.sin(th), color=FS.INK_2, ls="--", lw=0.8, label="target ball")
    ax.axhline(yc, color=FS.CATEGORICAL[0], lw=1.0, label=f"cut y = {yc:.3f} m (x-z slab, 5 mm)")
    ax.set_aspect("equal"); ax.set_xlabel("x (m, base)"); ax.set_ylabel("y (m)")
    ax.set_title("(1) layout, top view: where the slab is cut", fontsize=8, color=FS.INK); ax.legend(fontsize=6, loc="lower left")
    rf = float(z["r_finger"])
    for j, (key, ttl) in enumerate((("ball|off", "distractor interior, target-free query OFF\nmin(f, max(d, b))"),
                                    ("ball|on", "distractor interior, target-free query ON\nmin(f, max(c, d), max(d, b))"))):
        val = z[f"{I}__D_d_{key}"].astype(np.float64)
        cnt, slab = slab_panel(axes[0, 2 + j], P, val, mesh, ml, ball, 1, yc, vs, ttl)
        side[f"row1_{key}_slab_counts"] = cnt
        side[f"row1_{key}_frame_counts"] = {"n": int(len(val)), "positive": int((val > 0).sum()),
                                            "clearance_ge_margin": int((val - rf >= MARGIN).sum())}
        lx, hx = P[:, 0].min() - 0.03, P[:, 0].max() + 0.03
        axes[0, 2 + j].set_xlim(min(lx, ball[0] - ball[3] - 0.01), max(hx, ball[0] + ball[3] + 0.01))
        axes[0, 2 + j].set_ylim(P[:, 2].min() - 0.02, P[:, 2].max() + 0.02)
    # ---- row 2
    z = np.load(r2["_npz"]); zb = np.load(E2B / "raw_demo" / f"{r2['key']}.npz")
    mesh, ml = zb["mesh_pts"], zb["mesh_lab"]
    ball = z[f"{I}__ball"]
    ca, ra = z["auth_centres"].astype(np.float64), z["auth_radii"].astype(np.float64)
    links = z["auth_links"].astype(str)
    off, on = z[f"{I}__q_d_ball|off"].astype(np.float64), z[f"{I}__q_d_ball|on"].astype(np.float64)
    oon, ooff = z[f"{I}__q_d_oracle|on"].astype(np.float64), z[f"{I}__q_d_oracle|off"].astype(np.float64)
    gtt = z["q_gt_target_dist"].astype(np.float64)
    delta = (off - on) * 1e3
    ax = axes[1, 0]
    nr = np.linalg.norm(mesh[:, :2] - ball[:2], axis=1) < 0.15
    for li, col, nm in ((0, FS.INK, "target mesh"), (1, FS.INK_2, "distractor mesh")):
        k = nr & (ml == li)
        ax.scatter(mesh[k, 0], mesh[k, 2], s=1, c=col, label=nm)
    cmap = FS.sequential_cmap() if hasattr(FS, "sequential_cmap") else plt.get_cmap("Blues")
    vmax = max(1.0, float(delta.max()))
    for c_, r_, d_ in zip(ca, ra, delta):
        ax.add_patch(Circle((c_[0], c_[2]), r_, facecolor=cmap(d_ / vmax), edgecolor=FS.INK_2, lw=0.4, alpha=0.9))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax)); sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.04, label="off - on (mm)")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_xlim(ca[:, 0].min() - 0.06, ca[:, 0].max() + 0.06); ax.set_ylim(ca[:, 2].min() - 0.06, ca[:, 2].max() + 0.06)
    ax.set_title(f"(2) {r2['key']} (grasp_close_start)\nauthorized spheres, x-z projection, colour = decrease off-on, ball", fontsize=8, color=FS.INK)
    ax.legend(fontsize=6, loc="lower left")
    # target interior slab
    T = z[f"{I}__T_pts"].astype(np.float64)
    yc2 = float(ball[1])
    sz = json.load(open(str(r2["_npz"])[:-4] + ".json"))["identities"][I]["ball"].get("support_plane_z_m")
    for j, (key, ttl) in enumerate((("ball|off", "target interior, query OFF"), ("ball|on", "target interior, query ON"))):
        val = z[f"{I}__T_d_{key}"].astype(np.float64)
        Pn = T[np.abs(T[:, 1] - yc2) <= 0.0025 + 1e-9]
        if not len(Pn):
            yc2 = float(np.median(T[:, 1]))
        cnt, _ = slab_panel(axes[1, 1 + j], T, val, mesh, ml, ball, 1, yc2, vs, ttl + f" (slab y = {yc2:.3f})",
                            extra_line=(sz + 0.020, "support z + 20 mm (coarse z_min)") if sz is not None else None)
        side[f"row2_target_{key}_slab_counts"] = cnt
        side[f"row2_target_{key}_frame_counts"] = {"n": int(len(val)), "negative": int((val < 0).sum())}
        axes[1, 1 + j].set_xlim(ball[0] - ball[3] - 0.01, ball[0] + ball[3] + 0.01)
        axes[1, 1 + j].set_ylim(ball[2] - ball[3] - 0.01, ball[2] + ball[3] + 0.01)
    ax = axes[1, 3]
    order = np.argsort(gtt - ra)[:16]
    x = np.arange(len(order))
    w = 0.2
    for i_, (arr, lab) in enumerate(((gtt - ra, "GT target surface - r"), (off - ra, "target-free OFF - r"),
                                     (on - ra, "target-free ON - r"), (oon - ra, "oracle ON - r"))):
        ax.bar(x + (i_ - 1.5) * w, arr[order] * 1e3, w, color=FS.CATEGORICAL[i_], label=lab)
    ax.axhline(MARGIN * 1e3, color=FS.INK_2, ls=":", lw=0.8, label="margin 10 mm")
    ax.set_xticks(x); ax.set_xticklabels([f"{i}:{l.replace('link_', '').replace('ee_', '')}" for i, l in zip(order, links[order])], rotation=70, fontsize=6)
    ax.set_ylabel("clearance (mm)"); ax.legend(fontsize=6)
    ax.set_title("(2) 16 authorized spheres nearest the target:\nGT distance vs target-free clearance", fontsize=8, color=FS.INK)
    side["row2_spheres"] = [{"link": str(links[i]), "r_mm": round(float(ra[i]) * 1e3, 2), "gt_target_minus_r_mm": round(float(gtt[i] - ra[i]) * 1e3, 2),
                             "off_minus_r_mm": round(float(off[i] - ra[i]) * 1e3, 2), "on_minus_r_mm": round(float(on[i] - ra[i]) * 1e3, 2),
                             "oracle_on_minus_r_mm": round(float(oon[i] - ra[i]) * 1e3, 2),
                             "oracle_off_minus_r_mm": round(float(ooff[i] - ra[i]) * 1e3, 2)} for i in order]
    side["row2_max_decrease_mm"] = round(float(delta.max()), 3)
    fig.suptitle("T44c D-sign2 (gt_label, ball). Row 1: distractor-interior fine-lattice points in a 5 mm x-z slab, sign of the target-free answer "
                 "off vs on. Row 2: last demo frame before grasp close: authorized spheres and target interior. dashed = target ball",
                 fontsize=9, color=FS.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "t44c-scene.png"; fig.savefig(out, dpi=130); plt.close(fig)
    json.dump(side, open(out.with_suffix(".json"), "w"), indent=1)
    return out


def pooled(set_, key_fn):
    vals = []
    for f in sorted(glob.glob(str(D / f"raw_{set_}" / "*.npz"))):
        z = np.load(f)
        v = key_fn(z, f)
        if v is not None:
            vals.append(v)
    return np.concatenate(vals) if vals else np.zeros(0)


def fig_graph():
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.4))
    FS.style_axes(fig, axes)
    side = {}
    recs = demo_recs()
    last = {}
    for r in recs:
        last.setdefault(r["source"]["episode_index"], []).append(r["seq"])
    lastkeys = {r["key"] for r in recs if r["seq"] in sorted(last[r["source"]["episode_index"]])[-2:]}
    # (1) ECDF of decrease at authorized spheres (q), demo last2 + all demo, per method
    ax = axes[0]
    for i, m in enumerate(("ball", "bo_d3", "oracle")):
        for ls, sub, nm in (("-", lastkeys, "demo last2"), (":", None, "demo all")):
            def fn(z, f, m=m, sub=sub):
                if sub is not None and pathlib.Path(f).stem not in sub:
                    return None
                return (z[f"{I}__q_d_{m}|off"] - z[f"{I}__q_d_{m}|on"]).astype(np.float64) * 1e3
            d = np.sort(pooled("demo", fn))
            if len(d):
                ax.plot(d, np.arange(1, len(d) + 1) / len(d), ls, color=FS.CATEGORICAL[i], label=f"{m} {nm} (n={len(d)})")
                side[f"q_decrease_mm|{m}|{nm}"] = {"n": int(len(d)), "frac_gt_0.001mm": round(float((d > 1e-3).mean()), 4),
                                                   "median": round(float(np.median(d)), 3), "p95": round(float(np.percentile(d, 95)), 3),
                                                   "max": round(float(d.max()), 3)}
    ax.set_xlabel("target-free distance decrease off - on (mm)"); ax.set_ylabel("ECDF (authorized spheres)")
    ax.set_title("(a) authorized-sphere decrease, demo (gt_label)", fontsize=9, color=FS.INK); ax.legend(fontsize=6); ax.grid(color=FS.GRID_INK, lw=0.4)
    # (2) off vs on clearance scatter, demo last2, ball and oracle
    ax = axes[1]
    for i, m in enumerate(("ball", "oracle")):
        def fo(z, f, m=m):
            if pathlib.Path(f).stem not in lastkeys:
                return None
            return np.stack([z[f"{I}__q_d_{m}|off"] - z["auth_radii"], z[f"{I}__q_d_{m}|on"] - z["auth_radii"]], 1).astype(np.float64) * 1e3
        xy = pooled("demo", fo)
        if len(xy):
            xy = xy.reshape(-1, 2)
            ax.scatter(xy[:, 0], xy[:, 1], s=6, color=FS.CATEGORICAL[0 if m == "ball" else 2], alpha=0.6, label=f"{m} (n={len(xy)})")
            side[f"clearance_off_on_last2|{m}"] = {"n": int(len(xy)), "n_cross_margin": int(((xy[:, 0] >= 10) & (xy[:, 1] < 10)).sum())}
    lim = (-5, 80)
    ax.plot(lim, lim, color=FS.INK_2, lw=0.7); ax.axhline(10, color=FS.INK_2, ls=":", lw=0.7); ax.axvline(10, color=FS.INK_2, ls=":", lw=0.7)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("clearance OFF (mm) = d_free - r"); ax.set_ylabel("clearance ON (mm)")
    ax.set_title("(b) demo last2 frames: authorized-sphere clearance off vs on", fontsize=9, color=FS.INK); ax.legend(fontsize=6); ax.grid(color=FS.GRID_INK, lw=0.4)
    # (3) safety counts off/on (distractor interior) demo & lb
    ax = axes[2]
    labels, pos, clr = [], [], []
    for s in ("demo", "lb"):
        sf = A["sets"][s][I]["safety"]
        if "ball|off" not in sf:
            continue
        for m in ("ball", "bo_d3", "oracle"):
            for c in ("off", "on"):
                labels.append(f"{s}\n{m}|{c}")
                pos.append(sf[f"{m}|{c}"]["n_frames_with_positive"]); clr.append(sf[f"{m}|{c}"]["n_frames_clearance_ge_margin"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, pos, 0.4, color=FS.CATEGORICAL[0], label="frames with any point > 0")
    ax.bar(x + 0.2, clr, 0.4, color=FS.CATEGORICAL[1], label="frames with clearance >= 10 mm")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=70, fontsize=6); ax.set_ylabel("frames")
    ax.set_title("(c) finger sphere inside a distractor: frames (64 demo / 48 lb)", fontsize=9, color=FS.INK); ax.legend(fontsize=6)
    side["safety_frames"] = {"labels": labels, "frames_positive": pos, "frames_clearance_ge_margin": clr}
    # (4) target interior: on-negative by height above support
    ax = axes[3]
    def fh(z, f, m):
        r = json.load(open(f[:-4] + ".json"))
        if f"{I}__T_pts" not in z.files:
            return None
        sz = r["identities"][I]["ball"].get("support_plane_z_m")
        P = z[f"{I}__T_pts"].astype(np.float64)
        on = z[f"{I}__T_d_{m}|on"].astype(np.float64); off = z[f"{I}__T_d_{m}|off"].astype(np.float64)
        return np.stack([(P[:, 2] - sz) * 1e3, on < 0, off < 0], 1)
    bins = np.arange(0, 80, 5)
    for i, m in enumerate(("ball", "bo_d3", "oracle")):
        arr = pooled("demo", lambda z, f, m=m: fh(z, f, m)).reshape(-1, 3)
        if not len(arr):
            continue
        h = arr[:, 0]
        idx = np.digitize(h, bins)
        fr_on = [float(arr[idx == k, 1].mean()) if (idx == k).any() else np.nan for k in range(1, len(bins))]
        ax.plot(bins[:-1] + 2.5, fr_on, "-o", ms=3, color=FS.CATEGORICAL[i], label=f"{m} ON")
        if m == "ball":
            fr_off = [float(arr[idx == k, 2].mean()) if (idx == k).any() else np.nan for k in range(1, len(bins))]
            ax.plot(bins[:-1] + 2.5, fr_off, "--", color=FS.INK_2, label="ball OFF")
            side["target_interior_off_frac_neg_by_height|ball"] = [None if np.isnan(v) else round(v, 4) for v in fr_off]
        side[f"target_interior_on_frac_neg_by_height|{m}"] = [None if np.isnan(v) else round(v, 4) for v in fr_on]
        side[f"target_interior_n_by_height|{m}"] = [int((idx == k).sum()) for k in range(1, len(bins))]
    side["height_bins_mm"] = bins.tolist()
    ax.axvline(20, color=FS.CATEGORICAL[3], ls="-.", lw=0.9, label="coarse z_min (support + 20 mm)")
    ax.set_xlabel("height above support plane (mm)"); ax.set_ylabel("fraction of target-interior points < 0")
    ax.set_title("(d) target interior (demo, gt_label): answer < 0 by height", fontsize=9, color=FS.INK); ax.legend(fontsize=6)
    ax.grid(color=FS.GRID_INK, lw=0.4)
    fig.suptitle("T44c D-sign2: target-free coarse tier off vs on (gt_label)", fontsize=10, color=FS.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG / "t44c-graph.png"; fig.savefig(out, dpi=130); plt.close(fig)
    json.dump(side, open(out.with_suffix(".json"), "w"), indent=1)
    return out


def fig_table():
    rows = []
    for s in ("demo", "lb", "demo_last2", "demo_last1"):
        for ident in ("gt", "gt_label"):
            g = A["sets"][s][ident]
            for m in ("ball", "bo_d3", "oracle"):
                bq = g["blocking_q"][m]; ti = g["target_interior"][m]; cs = g["coarse_seeds"][m]
                sf = g.get("safety", {})
                if "n_frames" not in bq:
                    continue
                if f"{m}|off" not in sf:
                    sf = {}
                saf = lambda c: (f"{sf[f'{m}|{c}']['n_positive']}/{sf[f'{m}|{c}']['n_frames_with_positive']}  "  # noqa: E731
                                 f"{sf[f'{m}|{c}']['n_clearance_ge_margin']}/{sf[f'{m}|{c}']['n_frames_clearance_ge_margin']}") if sf else "-"
                dd = bq["decrease_mm_among_decreased"]
                rows.append([s, ident, m, bq["n_frames"], saf("off"), saf("on"),
                             f"{bq['n_decreased']}/{bq['n_queries']}",
                             "-" if not dd.get("n") else f"{dd['median']}/{dd['p95']}/{dd['max']}",
                             f"{bq['n_cross_margin']} ({bq['n_frames_cross_margin']} fr.)",
                             "-" if m == "oracle" else f"{bq.get('n_decreased_shared_caused', 0)} / {bq.get('n_decreased_near_shared', 0)}",
                             f"{ti['n_on_negative']}/{ti['n_points']} ({ti['n_frames_on_negative']} fr.)",
                             f"{ti['n_off_negative']}",
                             "-" if m == "oracle" else f"{cs['n_shared_kept']['sum']} (med {cs['n_shared_kept']['median']}, max {cs['n_shared_kept']['max']})"])
    hdr = ["set", "identity", "method", "frames", "D: >0 pts/fr  clr>=10 pts/fr  OFF", "same ON",
           "auth. spheres decreased", "decrease mm med/p95/max", "cross margin", "decr. shared-caused / near-shared",
           "target interior ON<0 (fr.)", "OFF<0", "n_shared_kept sum"]
    fig = plt.figure(figsize=(22, 0.26 * len(rows) + 2.6))
    ax = fig.add_axes([0.005, 0.02, 0.99, 0.88]); ax.axis("off")
    t = ax.table(cellText=[[str(v) for v in r] for r in rows], colLabels=hdr, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(6.5); t.scale(1, 1.12)
    for (rr, cc), cell in t.get_celld().items():
        cell.set_edgecolor(FS.GRID_INK)
        if rr == 0:
            cell.set_facecolor(FS.SEQ_BLUE[0])
    fig.patch.set_facecolor(FS.SURFACE)
    fig.suptitle("T44c D-sign2 summary. D = finger-radius sphere centre at GT distractor-interior fine-lattice points (T44E2b definition); "
                 "authorized spheres = DEFAULT_CONTACT_LINKS[arm]; decrease = d_free OFF - ON; cross = OFF clearance >= 10 mm and ON < 10 mm; "
                 "shared-caused = ON < ON without shared-voxel preservation", fontsize=8)
    out = FIG / "t44c-table.png"; fig.savefig(out, dpi=130); plt.close(fig)
    json.dump({"rows": [dict(zip(hdr, r)) for r in rows]}, open(out.with_suffix(".json"), "w"), indent=1, default=str)
    return out


if __name__ == "__main__":
    for f in (fig_table, fig_graph, fig_scene):
        print(f())
