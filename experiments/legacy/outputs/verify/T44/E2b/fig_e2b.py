"""T44b D-sign figures (verifier): (a) scene: distractor interior sign per layer (slab), (b) graph: criteria vs d,
unowned distance CDFs, interior sign composition, (c) table. + .json sidecars. Writes AG3S/ag3s/docs/figures/t44/t44e2b-*."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, pathlib, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle as FS
from AG3S.ag3s.experiments.live import exclusion_e2 as H

D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2b'))
E2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t44')); FIG.mkdir(parents=True, exist_ok=True)
KO = FS.use_korean()
A = json.load(open(D / "agg_e2b.json"))
IDENT = "gt_label"
ZERO = 1e-6
for n in ("t44e2b-scene", "t44e2b-graph", "t44e2b-table"):
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


def pick_frame():
    """demo frame (gt_label) with the most target-free-positive distractor-interior points at depth >= 10 mm."""
    best = None
    for f in sorted((D / "raw_demo").glob("*.json")):
        r = json.load(open(f))
        s = r["identities"][IDENT].get("sign", {}).get("safety")
        if not s:
            continue
        v = s["ball"]["n_positive_depth_ge_10mm"]
        if best is None or v > best[0]:
            best = (v, r)
    return best[1]


def fig_scene():
    r = pick_frame()
    key = r["key"]
    z = np.load(D / "raw_demo" / f"{key}.npz", allow_pickle=True)
    I = IDENT
    ijk = z[f"{I}__sg_ijk"].astype(float); org = z[f"{I}__sg_origin"]; vs = float(z[f"{I}__sg_vs"])
    P = org + ijk * vs
    hid = z[f"{I}__sg_hull"]
    # largest hull in this frame
    h = np.bincount(hid - hid.min()).argmax() + hid.min()
    sel = hid == h
    yc = np.median(P[sel, 1])
    slab = sel & (np.abs(P[:, 1] - yc) <= 0.5 * vs + 1e-9)
    mesh, ml = z["mesh_pts"], z["mesh_lab"]
    ball = z[f"{I}__sg_ball"]
    fs, fsdf = z[f"{I}__sg_fine_state"], z[f"{I}__sg_fine_sdf"]
    panels = [("fine TSDF state (5 mm, trunc 30 mm)", None),
              ("main fine layer (5 mm)", z[f"{I}__sg_val_main_fine_layer"]),
              ("coarse layer (20 mm, trunc 60 mm)", z[f"{I}__sg_val_coarse_layer"]),
              ("main composite = min(coarse, fine)", z[f"{I}__sg_val_main_composite"]),
              ("target-free tier, ball (raw)", z[f"{I}__sg_val_ball"]),
              ("target-free query, ball\nmin(f, max(d, b))", z[f"{I}__sg_comp_ball"])]
    pfs = json.load(open(E2 / "e2_demo_posed_frames.json"))
    ep, fr = int(key.split("_ep")[1][:4]), int(key.split("_f")[1])
    pf = next(p for p in pfs if p["episode_index"] == ep and p["frame"] == fr)
    cam = "zed_left" if "zed_left" in r["attention_cameras"] else r["attention_cameras"][0]
    img, (u, v) = render_rgb(pf, cam)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    FS.style_axes(fig, axes)
    ax = axes[0, 0]
    cu, cv = int(np.clip(u, 120, 520)), int(np.clip(v, 90, 390))
    ax.imshow(img[cv - 90:cv + 90, cu - 120:cu + 120]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{key} ({cam})\ntarget {pf['target_fruit']}, nearest {r['nearest_distractor_3d_mm'][0]} {r['nearest_distractor_3d_mm'][1]} mm",
                 fontsize=8, color=FS.INK)
    # layout: top view x-y
    ax = axes[0, 1]
    near = np.linalg.norm(mesh[:, :2] - ball[:2], axis=1) < 0.15
    for li, col, nm in ((0, FS.CATEGORICAL[0], "target mesh"), (1, FS.CATEGORICAL[1], "distractor mesh")):
        k = near & (ml == li)
        ax.scatter(mesh[k, 0], mesh[k, 1], s=1, c=col, label=nm)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(ball[0] + ball[3] * np.cos(th), ball[1] + ball[3] * np.sin(th), color=FS.INK_2, ls="--", lw=0.8, label="ball")
    ax.axhline(yc, color=FS.INK, lw=1.0, label=f"slab y = {yc:.3f} m (x-z cut)")
    ax.set_aspect("equal"); ax.set_xlabel("x (m, base)"); ax.set_ylabel("y (m)")
    ax.set_title("layout (top view): cut plane", fontsize=8, color=FS.INK); ax.legend(fontsize=6, loc="lower left")
    side = {"key": key, "identity": I, "hull": int(h), "slab_y_m": float(yc), "n_interior_points_hull": int(sel.sum()),
            "n_slab_points": int(slab.sum()), "panels": {}}
    mslab = near & (np.abs(mesh[:, 1] - yc) <= 0.003)
    cols = {"neg": FS.CATEGORICAL[0], "zero": FS.CATEGORICAL[6], "pos": FS.CATEGORICAL[1],
            "unobserved": FS.CATEGORICAL[1], "observed_sdf<0": FS.CATEGORICAL[0], "observed_sdf>=0": FS.CATEGORICAL[2]}
    for j, (title, val) in enumerate(panels):
        ax = axes.flat[2 + j]
        p = P[slab]
        if val is None:
            st = np.where(fs[slab] <= 1, "unobserved", np.where(fsdf[slab] < 0, "observed_sdf<0", "observed_sdf>=0"))
        else:
            x = val[slab]
            st = np.where(np.abs(x) <= ZERO, "zero", np.where(x < 0, "neg", "pos"))
        cnt = {}
        for c in np.unique(st):
            k = st == c
            cnt[str(c)] = int(k.sum())
            ax.scatter(p[k, 0], p[k, 2], s=34, marker="s", c=cols[c], lw=0, label=f"{c} ({k.sum()})")
        for li, col in ((0, FS.CATEGORICAL[0]), (1, FS.INK_2)):
            k = mslab & (ml == li)
            ax.scatter(mesh[k, 0], mesh[k, 2], s=1, c=col)
        ax.plot(ball[0] + ball[3] * np.cos(th), ball[2] + ball[3] * np.sin(th), color=FS.INK_2, ls="--", lw=0.8)
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        lx, hx = P[sel, 0].min() - 0.03, P[sel, 0].max() + 0.03
        lz, hz = P[sel, 2].min() - 0.02, P[sel, 2].max() + 0.02
        ax.set_xlim(min(lx, ball[0] - ball[3] - 0.01), max(hx, ball[0] + ball[3] + 0.01)); ax.set_ylim(lz, hz)
        ax.set_title(title, fontsize=8, color=FS.INK); ax.legend(fontsize=6, loc="upper right")
        side["panels"][title] = cnt
    fig.suptitle("T44b D-sign: distractor interior voxels (fine lattice, inside GT convex hull) in one x-z slab, sign per layer. "
                 "grey line = distractor mesh, dashed = target ball", fontsize=10, color=FS.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "t44e2b-scene.png"; fig.savefig(out, dpi=130); plt.close(fig)
    side["selection"] = "demo frame (gt_label) with the most target-free (ball) positive interior points at depth >= 10 mm"
    json.dump(side, open(out.with_suffix(".json"), "w"), indent=1)
    return out


def fig_graph():
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    FS.style_axes(fig, axes)
    side = {}
    # (1) criteria vs d
    ax = axes[0]
    k = 0
    for s in ("demo", "lb"):
        for ident in ("gt", "gt_label"):
            b = A["sets"][s].get(ident)
            if not b:
                continue
            red = [b["criteria"][m]["(1)_loss_reduction_pooled"] for m in ("bo_d2", "bo_d3", "bo_d4")]
            rec = [b["criteria"][m]["(2)_recall_median"] - b["criteria"][m]["(2)_ball_recall_median"] for m in ("bo_d2", "bo_d3", "bo_d4")]
            col = FS.CATEGORICAL[k]; k += 1
            if all(x is not None for x in red):
                ax.plot([2, 3, 4], red, "-o", color=col, label=f"{s}/{ident} (1) loss reduction")
            ax.plot([2, 3, 4], rec, "--s", color=col, label=f"{s}/{ident} (2) recall med - ball")
            side[f"{s}|{ident}"] = {"loss_reduction": red, "recall_minus_ball": rec}
    ax.axhline(0.5, color=FS.INK_2, lw=0.8, ls=":"); ax.axhline(-0.05, color=FS.INK_2, lw=0.8, ls=":")
    ax.set_xticks([2, 3, 4]); ax.set_xlabel("owned_max_dist_voxels (x 5 mm)"); ax.set_ylabel("fraction")
    ax.set_title("criteria (1) >= 0.5 and (2) >= -0.05 vs d", fontsize=9, color=FS.INK); ax.legend(fontsize=6)
    ax.grid(color=FS.GRID_INK, lw=0.4)
    # (2) unowned distractor seed distances (demo, gt_label, d=2) — CDF
    ax = axes[1]
    import glob
    arr = {}
    for f in sorted(glob.glob(str(D / "raw_demo" / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        for kk in ("raw_depth", "per_camera_min", "fused", "nontarget_evidence", "distractor_mesh"):
            nm = f"{IDENT}__unbo_d2__{kk}"
            if nm in z:
                arr.setdefault(kk, []).append(z[nm].astype(float) * 1000)
    side["unowned_cdf_n"] = {}
    for i, (kk, v) in enumerate(arr.items()):
        v = np.sort(np.concatenate(v)); side["unowned_cdf_n"][kk] = int(len(v))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=FS.CATEGORICAL[i], label=f"{kk} (n={len(v)})")
        side[f"unowned_{kk}_quantiles_mm"] = [round(float(np.percentile(v, p)), 2) for p in (10, 25, 50, 75, 90)]
    for t in (10, 15, 20):
        ax.axvline(t, color=FS.INK_2, lw=0.7, ls=":")
    ax.set_xlim(0, 40); ax.set_xlabel("distance from unowned distractor seed (mm)"); ax.set_ylabel("CDF")
    ax.set_title("demo/gt_label, d=2: unowned GT-distractor seeds\nnearest point per evidence level (dotted: d=2,3,4 thresholds)",
                 fontsize=9, color=FS.INK); ax.legend(fontsize=6); ax.grid(color=FS.GRID_INK, lw=0.4)
    # (3) interior composition per source (pooled demo+lb, gt_label)
    ax = axes[2]
    srcs = ("none", "ball", "main_fine_layer", "coarse_layer", "main_composite")
    comp = {}
    for s in ("demo", "lb"):
        sg = A["sets"][s][IDENT]["sign"]
        comp[s] = sg
    labels, neg, zero, pos = [], [], [], []
    for s in ("demo", "lb"):
        tab = json.load(open(D / f"sign_comp_{s}.json"))
        for src in srcs + ("composed_ball",):
            labels.append(f"{s}\n{src}")
            neg.append(tab[src]["neg"]); zero.append(tab[src]["zero"]); pos.append(tab[src]["pos"])
    x = np.arange(len(labels))
    ax.bar(x, neg, color=FS.CATEGORICAL[0], label="< 0 (inside)")
    ax.bar(x, zero, bottom=neg, color=FS.CATEGORICAL[6], label="= 0 (seed voxel)")
    ax.bar(x, pos, bottom=np.add(neg, zero), color=FS.CATEGORICAL[1], label="> 0 (reads free)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=70, fontsize=6)
    ax.set_ylabel("fraction of distractor-interior voxels (pooled)")
    ax.set_title("sign composition per layer (gt_label)", fontsize=9, color=FS.INK); ax.legend(fontsize=7)
    side["sign_composition"] = {"labels": labels, "neg": neg, "zero": zero, "pos": pos}
    fig.suptitle("T44b D-sign: owned_max_dist sweep, unowned cause distances, distractor-interior sign", fontsize=10, color=FS.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG / "t44e2b-graph.png"; fig.savefig(out, dpi=130); plt.close(fig)
    json.dump(side, open(out.with_suffix(".json"), "w"), indent=1)
    return out


def fig_table():
    hdr = ["set", "identity", "method", "n", "hit fr.", "recall med", "dist. lost", "(1) reduct.", "(2) recall vs ball",
           "(3) FF q/grid (ball)", "(4) p90 +ms", "unowned rm", "unowned distractor"]
    rows = []
    for s in ("demo_lt_0.08", "demo_0.08_0.10", "demo", "lb"):
        for ident in ("gt", "gt_label", "gt_cluster_only"):
            b = A["sets"][s].get(ident)
            if not b:
                continue
            for m in ("ball", "bo_d2", "bo_d3", "bo_d4", "oracle"):
                c = b["methods"][m]; cr = b["criteria"].get(m, {})
                pf = lambda v: "-" if v is None else ("P" if v else "F")  # noqa: E731
                un = (b.get("unowned") or {}).get(m, {})
                rows.append([s, ident, m, b["n_frames"], b["n_frames_ball_lost_distractor"], c["recall_median"], c["distractor_lost_sum"],
                             "-" if not cr else f"{cr['(1)_loss_reduction_pooled']} {pf(cr['(1)_pass'])}",
                             "-" if not cr else f"{cr['(2)_recall_median']}/{cr['(2)_ball_recall_median']} {pf(cr['(2)_pass'])}",
                             "-" if m == "oracle" else (f"{c['q_vs_oracle_roi']['n_falsely_feasible']}/{c['grid_vs_oracle_roi']['n_falsely_feasible']}"
                                                        + ("" if not cr else f" {pf(cr['(3)_pass'])}")),
                             "-" if not cr else f"{cr.get('(4)_p90_increase_ms_dedicated')} {pf(cr.get('(4)_pass'))}",
                             c["unowned_removed_sum"], un.get("n_unowned_distractor", "-")])
    sg_rows = []
    for s in ("demo", "lb"):
        sg = A["sets"][s][IDENT]["sign"]
        hf = sg["hull_frac_negative_median"]
        saf = sg["safety"]
        sg_rows.append([s, sg["n_frames"], sg["n_hulls"], sg["n_points"], hf.get("none"), hf.get("ball"), hf.get("main_fine_layer"),
                        hf.get("coarse_layer"), hf.get("main_composite"), hf.get("composed_ball"),
                        f"{saf['main_composite']['n_positive']} / {saf['main_composite']['n_frames_with_positive']}",
                        f"{saf['target_free_ball']['n_positive']} / {saf['target_free_ball']['n_frames_with_positive']}",
                        f"{saf['main_composite']['n_clearance_ge_margin']} / {saf['target_free_ball']['n_clearance_ge_margin']} "
                        f"({saf['target_free_ball']['n_frames_clearance_ge_margin']} fr.)"])
    sh = ["set", "frames", "hulls", "points", "none", "ball", "main fine", "coarse", "main comp.", "TF query ball",
          "main >0 pts/fr", "TF ball >0 pts/fr", "clear>=10mm main / TF (fr.)"]
    fig = plt.figure(figsize=(20, 0.27 * (len(rows) + len(sg_rows)) + 3))
    ax1 = fig.add_axes([0.01, 0.22, 0.98, 0.74]); ax2 = fig.add_axes([0.01, 0.01, 0.98, 0.18])
    for ax in (ax1, ax2):
        ax.axis("off")
    t1 = ax1.table(cellText=[[str(v) for v in r] for r in rows], colLabels=hdr, loc="upper center", cellLoc="center")
    t1.auto_set_font_size(False); t1.set_fontsize(6.5); t1.scale(1, 1.1)
    t2 = ax2.table(cellText=[[str(v) for v in r] for r in sg_rows], colLabels=sh, loc="upper center", cellLoc="center")
    t2.auto_set_font_size(False); t2.set_fontsize(7)
    for tb in (t1, t2):
        for (rr, cc), cell in tb.get_celld().items():
            cell.set_edgecolor(FS.GRID_INK)
            if rr == 0:
                cell.set_facecolor(FS.SEQ_BLUE[0])
    fig.patch.set_facecolor(FS.SURFACE)
    fig.suptitle("T44b D-sign. top: ball_owned_remove with owned_max_dist 2/3/4 voxels (tie 1.0), criteria (1)-(4) fixed in T44.task.md; "
                 "FF = falsely feasible vs oracle (ROI, margin 10 mm). bottom: distractor-interior median per-hull fraction < 0 "
                 "and authorized-sphere centre at interior points (gt_label)", fontsize=8)
    out = FIG / "t44e2b-table.png"; fig.savefig(out, dpi=130); plt.close(fig)
    json.dump({"rows": [dict(zip(hdr, r)) for r in rows], "sign_rows": [dict(zip(sh, r)) for r in sg_rows]},
              open(out.with_suffix(".json"), "w"), indent=1, default=str)
    return out


def sign_comp():
    """pooled neg/zero/pos fractions per source (gt_label) -> sign_comp_<set>.json (used by fig_graph)."""
    import glob
    for s, sub in (("demo", "raw_demo"), ("lb", "raw_lb")):
        acc = {}
        for f in sorted(glob.glob(str(D / sub / "*.npz"))):
            z = np.load(f, allow_pickle=True)
            if f"{IDENT}__sg_val_none" not in z:
                continue
            for src in ("none", "ball", "main_fine_layer", "coarse_layer", "main_composite"):
                acc.setdefault(src, []).append(z[f"{IDENT}__sg_val_{src}"].astype(float))
            acc.setdefault("composed_ball", []).append(z[f"{IDENT}__sg_comp_ball"].astype(float))
        tab = {}
        for src, v in acc.items():
            x = np.concatenate(v)
            tab[src] = {"n": int(len(x)), "neg": round(float((x < -ZERO).mean()), 4), "zero": round(float((np.abs(x) <= ZERO).mean()), 4),
                        "pos": round(float((x > ZERO).mean()), 4)}
        json.dump(tab, open(D / f"sign_comp_{s}.json", "w"), indent=1)


if __name__ == "__main__":
    sign_comp()
    for f in (fig_graph, fig_table, fig_scene):
        print(f())
