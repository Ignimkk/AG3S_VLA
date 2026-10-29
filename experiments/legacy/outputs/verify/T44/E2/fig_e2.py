"""T44 V-E2 figures (verifier): (a) scene, (b) recall vs distractor-loss scatter, (c) summary table. + .json sidecars."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, pathlib, numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle as FS
from AG3S.ag3s.experiments.live import exclusion_e2 as H

E2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t44')); FIG.mkdir(parents=True, exist_ok=True)
KO = FS.use_korean()
A = json.load(open(E2 / "agg_e2.json"))
METH = ("ball", "ball_owned_remove", "ball_owned_keep", "oracle")
MCOL = {m: FS.CATEGORICAL[i] for i, m in enumerate(METH)}
IDENT = "gt_label"


def pick_frames():
    fr = [f for f in A["sets"]["demo"][IDENT]["frames"] if f["ball__dlost"] >= 1]
    v = np.array([f["ball_owned_remove__dlost"] for f in fr])
    med = fr[int(np.argmin(np.abs(v - np.median(v))))]
    worst = max(fr, key=lambda f: (f["ball_owned_remove__dlost"], -f["ball_owned_remove__recall"]))
    return [("median", med), ("worst", worst)]


def render_rgb(key, cam):
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    import mujoco
    pfs = json.load(open(E2 / "e2_demo_posed_frames.json"))
    ep, fr = int(key.split("_ep")[1][:4]), int(key.split("_f")[1])
    pf = next(p for p in pfs if p["episode_index"] == ep and p["frame"] == fr)
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
    u, v = u * 640 / w, v * 480 / h
    r.close(); sc.close()
    return img, (u, v), pf


def fig_scene():
    picks = pick_frames()
    meths = ("ball", "ball_owned_remove", "ball_owned_keep", "oracle")
    fig, axes = plt.subplots(len(picks), 1 + len(meths), figsize=(19, 4.3 * len(picks)))
    FS.style_axes(fig, axes)
    side = {"used": []}
    for row, (tag, f) in enumerate(picks):
        rec = json.load(open(E2 / "raw_demo" / f"{f['key']}.json"))
        z = np.load(E2 / "raw_demo" / f"{f['key']}.npz")
        cam = "zed_left" if "zed_left" in rec["attention_cameras"] else rec["attention_cameras"][0]
        img, (u, v), pf = render_rgb(f["key"], cam)
        ax = axes[row, 0]
        cu, cv = int(np.clip(u, 120, 520)), int(np.clip(v, 90, 390))
        ax.imshow(img[cv - 90:cv + 90, cu - 120:cu + 120]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{tag}: {f['key']} ({cam})\n{pf['target_fruit']} target, nearest {rec['nearest_distractor_3d_mm'][0]} "
                     f"{rec['nearest_distractor_3d_mm'][1]} mm", fontsize=8, color=FS.INK)
        ijk, lab = z[f"{IDENT}__seed_ijk"].astype(float), z[f"{IDENT}__seed_label"]
        org, vs = z[f"{IDENT}__origin"], float(z[f"{IDENT}__voxel_size"])
        p = org + ijk * vs
        bc, br = z[f"{IDENT}__ball"][:3], z[f"{IDENT}__ball"][3]
        LT = {"target": 0, "distractor": 1}
        for j, m in enumerate(meths):
            ax = axes[row, j + 1]
            rm = z[f"{IDENT}__processed_{m}"] == 1
            # x-z side view (base frame), seeds within |y - yc| <= 10 mm slab
            sl = np.abs(p[:, 1] - bc[1]) <= 0.010
            for name, col, mk in (("target", FS.CATEGORICAL[0], "o"), ("distractor", FS.CATEGORICAL[1], "o"),
                                  ("unlabelled", FS.CATEGORICAL[6], "o"), ("support", FS.CATEGORICAL[2], "o")):
                li = H.LABELS.index(name)
                kk = sl & (lab == li) & ~rm
                rr = sl & (lab == li) & rm
                ax.scatter(p[kk, 0], p[kk, 2], s=9, c=col, marker="o", lw=0, label=f"{name} kept" if row == 0 and j == 0 else None)
                ax.scatter(p[rr, 0], p[rr, 2], s=14, facecolors="none", edgecolors=col, marker="o", lw=0.8,
                           label=f"{name} removed" if row == 0 and j == 0 else None)
            th = np.linspace(0, 2 * np.pi, 100)
            ax.plot(bc[0] + br * np.cos(th), bc[2] + br * np.sin(th), color=FS.INK_2, lw=0.8, ls="--")
            ax.set_aspect("equal"); ax.set_xlabel("x (m, base)"); ax.set_ylabel("z (m)")
            mt = rec["identities"][IDENT]["metrics"][m]
            ax.set_title(f"{m}\nrecall {mt['target_recall']}  distractor lost {mt['distractor_seeds_lost']}  "
                         f"unlab. rm {mt['unlabelled_removed']}", fontsize=8, color=FS.INK)
            side["used"].append({"row": tag, "key": f["key"], "method": m, "target_recall": mt["target_recall"],
                                 "distractor_seeds_lost": mt["distractor_seeds_lost"], "unlabelled_removed": mt["unlabelled_removed"],
                                 "n_slab_seeds": int(sl.sum()), "ball_centre": bc.tolist(), "ball_radius_m": float(br)})
    fig.legend(loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.suptitle(f"T44 E2-demo, identity {IDENT}: ROI seeds (ball + 20 mm shell), side view x-z, slab |y - y_ball| <= 10 mm. "
                 "filled = kept (obstacle), hollow = removed; dashed = ball", fontsize=10, color=FS.INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = FIG / "t44e2-scene.png"; fig.savefig(out, dpi=130); plt.close(fig)
    side.update({"identity": IDENT, "selection": "frames with ball distractor loss >= 1 (demo, gt_label): ball_owned_remove "
                 "loss closest to median, and max (tie: lower recall)", "source": str(E2 / "raw_demo")})
    json.dump(side, open(out.with_suffix(".json"), "w"), indent=1)
    return out


def fig_scatter():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    FS.style_axes(fig, axes)
    side = {}
    for i, ident in enumerate(("gt", "gt_label")):
        for j, s in enumerate(("demo", "lb")):
            ax = axes[i, j]
            fr = A["sets"][s][ident]["frames"]
            for k, m in enumerate(METH):
                x = [f[f"{m}__recall"] for f in fr]; y = [f[f"{m}__dlost"] for f in fr]
                jit = (k - 1.5) * 0.15
                ax.scatter(x, np.asarray(y) + jit, s=18, color=MCOL[m], alpha=0.8, lw=0, label=m if (i, j) == (0, 0) else None)
                side[f"{s}|{ident}|{m}"] = {"recall": x, "distractor_lost": y, "keys": [f["key"] for f in fr]}
            ax.set_title(f"{s} · identity {ident} (n={len(fr)})", fontsize=9, color=FS.INK)
            ax.set_xlabel("target removal recall (ROI)"); ax.set_ylabel("distractor seeds removed (ROI)")
            ax.grid(color=FS.GRID_INK, lw=0.4)
    fig.legend(loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("T44 V-E2-1: per frame recall vs distractor seed loss (y jittered by method +-0.2)", fontsize=10, color=FS.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = FIG / "t44e2-scatter.png"; fig.savefig(out, dpi=130); plt.close(fig)
    json.dump(side, open(out.with_suffix(".json"), "w"))
    return out


def fig_table():
    T = json.load(open(E2 / "timing" / "agg_timing.json")) if (E2 / "timing" / "agg_timing.json").exists() else None
    rows, side = [], []
    hdr = ["set", "identity", "method", "n", "recall med", "dist. lost\n(ball-hit fr.)", "reduct.", "amb.", "unowned\nrm/kept",
           "unlab.\nrm/ROI", "over q med/p95\nvs oracle mm", "FF q / grid\nvs oracle", "FF q / grid\nvs oracle'", "build p90\nms"]
    for s in ("demo_lt_0.08", "demo_0.08_0.10", "demo", "lb"):
        for ident in ("gt", "gt_label", "gt_cluster_only"):
            a = A["sets"][s][ident]
            if not a.get("n_frames"):
                continue
            for m in ("ball", "ball_owned_remove", "ball_owned_keep", "oracle"):
                c = a["methods"][m]
                r = [s, ident, m, a["n_frames"], c["recall_median"], c["distractor_lost_sum_on_ball_hit_frames"],
                     c.get("loss_reduction_pooled_on_hit_frames", "-"), c["ambiguous_sum"], f"{c['unowned_removed_sum']}/{c['unowned_kept_sum']}",
                     f"{c['unlabelled_removed_sum']}/{c['unlabelled_seeds_roi_sum']}",
                     "-" if m == "oracle" else f"{c['q_vs_oracle']['over_median_mm']}/{c['q_vs_oracle']['over_p95_mm']}",
                     "-" if m == "oracle" else f"{c['q_vs_oracle']['n_falsely_feasible']}/{c['grid_vs_oracle']['n_falsely_feasible']}",
                     f"{c['q_vs_oracle_prime']['n_falsely_feasible']}/{c['grid_vs_oracle_prime']['n_falsely_feasible']}",
                     c.get("build_ms_p90", "-")]
                rows.append([str(v) for v in r]); side.append(dict(zip([h.replace("\n", " ") for h in hdr], r)))
    crit = []
    for s in ("demo_lt_0.08", "demo_0.08_0.10", "demo", "lb"):
        for ident in ("gt", "gt_label", "gt_cluster_only"):
            a = A["sets"][s][ident]
            if not a.get("n_frames"):
                continue
            for m, c in a["criteria"].items():
                crit.append([s, ident, m, str(c["(1)_n_hit_frames"]),
                             f"{c['(1)_loss_reduction_pooled']} {'P' if c['(1)_pass'] else ('-' if c['(1)_pass'] is None else 'F')}",
                             f"{c['(2)_recall_median']} vs {c['(2)_ball_recall_median']} {'P' if c['(2)_pass'] else 'F'}",
                             f"q {c['(3)_falsely_feasible_q']}/{c['(3)_ball_q']} grid {c['(3)_falsely_feasible_grid']}/{c['(3)_ball_grid']} {'P' if c['(3)_pass'] else 'F'}",
                             f"{c['(4)_p90_increase_ms']} {'P' if c['(4)_pass'] else 'F'}"])
    if T:
        for m in ("ball_owned_remove", "ball_owned_keep"):
            t = T["methods"][m]
            crit.append(["timing pass", "gt_label", m, "-", "-", "-", "-",
                         f"{t['p90_increase_vs_ball_ms']} {'P' if t['p90_increase_vs_ball_ms'] <= 10 else 'F'}"])
    fig = plt.figure(figsize=(20, 0.26 * (len(rows) + len(crit)) + 2.5))
    ax1 = fig.add_axes([0.01, 0.47, 0.98, 0.49]); ax2 = fig.add_axes([0.01, 0.01, 0.98, 0.44])
    for ax in (ax1, ax2):
        ax.axis("off")
    t1 = ax1.table(cellText=rows, colLabels=hdr, loc="upper center", cellLoc="center"); t1.auto_set_font_size(False); t1.set_fontsize(6.5)
    t1.scale(1, 1.15)
    ch = ["set", "identity", "method", "ball-hit frames", "(1) loss reduction >= 0.5", "(2) recall med >= ball - 0.05",
          "(3) falsely feasible <= ball (method/ball)", "(4) p90 increase <= 10 ms"]
    t2 = ax2.table(cellText=crit, colLabels=ch, loc="upper center", cellLoc="center"); t2.auto_set_font_size(False); t2.set_fontsize(6.5)
    for tb in (t1, t2):
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor(FS.GRID_INK)
            if r == 0:
                cell.set_facecolor(FS.SEQ_BLUE[0])
    fig.patch.set_facecolor(FS.SURFACE)
    fig.suptitle("T44 V-E2 summary. FF = falsely feasible (oracle clearance < 10 mm, method >= 10 mm), queries = authorized "
                 "spheres in ROI; oracle' = oracle that also removes unlabelled seeds. P/F = fixed criteria", fontsize=9)
    out = FIG / "t44e2-table.png"; fig.savefig(out, dpi=130); plt.close(fig)
    json.dump({"summary_rows": side, "criteria_rows": crit}, open(out.with_suffix(".json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    for f in (fig_scatter, fig_table, fig_scene):
        print(f())
