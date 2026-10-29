"""Per-experiment figures on real scenes (rule A): the key mechanism of each IFG step, drawn on the policy image.

Each figure picks its example frame by a fixed rule (recorded in the sidecar), rebuilds that frame's graph,
recomputes the arms involved and projects the 5 mm surface nodes on the camera that sees the most target.

  MUJOCO_GL=osmesa PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_experiment_figures \
      --run $RUN --out AG3S/ag3s/docs/figures/ifg/exp
"""
import argparse, collections, glob, json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_graph as G  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_memory as M  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_reader as R  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_round as RD  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_round_mass as RM  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_round_ns as NS  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_run as RN  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_s1 as S1  # noqa: E402
from AG3S.ag3s.experiments.studies.ifg_figures import _project  # noqa: E402

CAT = figstyle.CATEGORICAL
TP_C, FP_C, FN_C = CAT[2], CAT[1], CAT[0]
CAMS = ("head", "left wrist", "right wrist")
REFS = None


def refs():
    global REFS
    if REFS is None:
        REFS = {r.frame_id: r for r in R.list_frames()}
    return REFS


def rows_of(run, sub, arm, thr=0.5):
    out = {}
    for f in glob.glob(str(run / sub / "rows" / "*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r["arm_id"] == arm and r.get("threshold", thr) == thr:
                out[r["frame_id"]] = r
    return out


class Frame:
    def __init__(self, run, fid, mask="fk", feats=()):
        self.ref = refs()[fid]
        self.fid = fid
        self.fin = R.load_frame(self.ref, features=feats)
        with np.load(run / "s0/replay" / f"{fid}.npz") as z:
            self.depth, rm, self.K, self.T = z["depth"], z["robot_mask"], z["K"], z["T_base_cam"]
        if mask == "render":
            with np.load(run / "rounds/R6/robot_mask" / f"{fid}.npz") as z:
                rm = z["robot_mask"]
        self.rmask = rm
        with np.load(run / "s0/labels" / f"{fid}.npz") as z:
            self.class_map, self.classes, self.cov = z["class_map"], [str(c) for c in z["classes"]], z["class_cov"]
        self.cfg = G.GraphConfig()
        self.geo = G.build_geometry(self.depth, self.K, self.T, rm, self.class_map, self.classes, self.cfg)
        with np.load(self.ref.path / "inputs.npz") as z:
            self.imgs = [np.clip((z[f"image_{c}"] + 1) / 2, 0, 1) for c in R.DUMP_CAMERAS]
        self.gt = self.geo.eval_target[self.geo.node_eval] >= 0.5

    def best_cam(self, mask=None):
        m = self.gt if mask is None else mask
        cnt = [int(_project(self.geo.pos[m], self.K[c], self.T[c])[2].sum()) for c in range(3)]
        return int(np.argmax(cnt))

    def draw(self, ax, c, layers, title, pos=None):
        pos = self.geo.pos if pos is None else pos
        ax.imshow(self.imgs[c])
        u, v, ok = _project(pos, self.K[c], self.T[c])
        for mask, col, s in layers:
            sel = ok & mask
            ax.scatter(u[sel], v[sel], s=s, c=[col] if isinstance(col, str) else col, lw=0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    def eval_layers(self, pred, pos_mask=None):
        """TP / FP / FN layers for a node mask over geo nodes."""
        return [(pred & self.gt, TP_C, 2), (pred & ~self.gt, FP_C, 2), (~pred & self.gt, FN_C, 2)]

    def iou(self, pred):
        return G.evaluate(self.geo, np.where(pred, 1.0, -1.0), 0.5, self.cfg).get("iou") or 0.0


def seed_and_g(fr, eligible_nonsupport):
    vp = NS.eligible(fr.geo, fr.cfg, eligible_nonsupport)
    six = S1.top_k(fr.fin.maps["N"], S1.SEED_K, vp)
    sw = fr.fin.maps["N"][six] / fr.fin.maps["N"][six].sum()
    roi = G.roi_nodes(fr.geo, G.seed_unary(fr.geo, six, sw), fr.cfg.roi_radius)
    sub = G.subset(fr.geo, roi)
    n = len(sub.pos)
    sl = G.seed_unary(sub, six, sw)
    lam, s = RN._combine([(sl, np.ones(n)), (fr.cfg.kappa_support * sub.support.astype(float), np.zeros(n))])
    m, _ = G.solve(sub, np.ones(n, bool), G.edge_weights(sub, fr.cfg, None), lam, s, fr.cfg.lam)
    idx = np.flatnonzero(roi)
    full = np.full(len(fr.geo.pos), -1.0); full[idx] = m
    sl_full = np.zeros(len(fr.geo.pos)); sl_full[idx] = sl
    return dict(six=six, sw=sw, vp=vp, roi=roi, sub=sub, m_sub=m, sl_sub=sl, m=full, sl=sl_full, idx=idx)


def to_full(fr, idx, x, fill=-1.0):
    out = np.full(len(fr.geo.pos), fill, float); out[idx] = x
    return out


def seed_patch_boxes(ax, six, c, color):
    for p in six:
        if p // 256 != c:
            continue
        r, col = divmod(p % 256, 16)
        ax.add_patch(plt.Rectangle((col * 14 - 0.5, r * 14 - 0.5), 14, 14, fill=False, ec=color, lw=1.4))


def save(fig, out, name, side):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=150); plt.close(fig)
    (out / f"{name}.json").write_text(json.dumps(side, indent=1, default=float))
    print(name)


def legend(fig, items):
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], ls="", marker="o", color=c, label=l) for l, c in items], loc="lower center",
               ncol=len(items), frameon=False, fontsize=8)


# ----------------------------------------------------------------------------------------- figures

def fig_s0(run, out):
    fid = "T53_ep1839_f023"
    fr = Frame(run, fid)
    fig, axes = plt.subplots(3, 4, figsize=(15, 9)); figstyle.style_axes(fig, axes.ravel())
    side = dict(frame=fid, rule="frame family of the protocol's debug frame022", cameras={})
    for c in range(3):
        axes[c, 0].imshow(fr.imgs[c]); axes[c, 0].set_title(f"{CAMS[c]}: policy RGB (dump)", fontsize=8)
        axes[c, 1].imshow(fr.depth[c], cmap=figstyle.sequential_cmap()); axes[c, 1].set_title("replay depth (planar z, m)", fontsize=8)
        axes[c, 2].imshow(fr.class_map[c] == 0, cmap="gray"); axes[c, 2].set_title("GT target (evaluation only)", fontsize=8)
        axes[c, 3].imshow(fr.rmask[c], cmap="gray"); axes[c, 3].set_title("serving FK robot mask (sphere volume)", fontsize=8)
        side["cameras"][CAMS[c]] = dict(valid_depth=float(((fr.depth[c] > 0.05) & (fr.depth[c] < 3)).mean()), robot_px=int(fr.rmask[c].sum()))
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"S0 alignment: the dumped policy image, MuJoCo replay depth, GT labels and robot mask of one frame ({fid}) line up", fontsize=10)
    save(fig, out, "exp-S0-alignment", side)


def fig_r1_r3(run, out):
    """R1 component gate vs R2 region gate vs R3 top-1 region, on the frame where R1 lost most recall."""
    g = rows_of(run, "rounds/R1", "G:N"); c1 = rows_of(run, "rounds/R1", "FGc:N:prefix_L8")
    fid = max(c1, key=lambda f: (g[f].get("recall") or 0) - (c1[f].get("recall") or 0))
    fr = Frame(run, fid, feats=("prefix_L8",))
    s = seed_and_g(fr, False)
    sub, n = s["sub"], len(s["sub"].pos)
    W = sp.coo_matrix((np.ones(len(sub.edges)), (sub.edges[:, 0], sub.edges[:, 1])), shape=(n, n))
    _, comp = connected_components(W, directed=False)
    uf = R.normalize(fr.fin.features["prefix_L8"]); mu = R.normalize((s["sw"] @ uf[s["six"]])[None])[0]
    pu, _ = G.proto_unary(uf @ mu, s["six"], s["vp"]); node_p = G.node_patch_mean(sub, pu)
    px = np.asarray(sub.patch_px.sum(1)).ravel()
    m1, _ = RD.component_gate(s["m_sub"], comp, s["sl_sub"], px, node_p, 0.5)
    m2, reg, pred, _ = RD.region_gate(s["m_sub"], sub, px, node_p, 0.5)
    m3 = RD.region_gate(s["m_sub"], sub, px, np.zeros(n), 0.5, s["sl_sub"], top1_only=True)[0]
    c = fr.best_cam()
    fig, axes = plt.subplots(1, 4, figsize=(18, 5)); figstyle.style_axes(fig, axes)
    # panel 1: graph components (colour per component) of the ROI
    cols = np.array([CAT[i % len(CAT)] for i in range(comp.max() + 1)])
    u, v, okp = _project(sub.pos, fr.K[c], fr.T[c])
    axes[0].imshow(fr.imgs[c]); axes[0].scatter(u[okp], v[okp], s=2, c=cols[comp][okp].tolist(), lw=0)
    axes[0].set_title("ROI graph components (one colour each)", fontsize=8); axes[0].set_xticks([]); axes[0].set_yticks([])
    arms = [("R1 component gate (FGc)", m1), ("R2 region gate (FGr)", m2), ("R3 top-1 seed-mass region", m3)]
    side = dict(frame=fid, rule="frame with the largest recall loss of R1 FGc:N:prefix_L8 vs G:N", panels={})
    for ax, (t, m) in zip(axes[1:], arms):
        pm = to_full(fr, s["idx"], m) >= 0.5
        fr.draw(ax, c, fr.eval_layers(pm), f"{t}  IoU={fr.iou(pm):.2f}")
        side["panels"][t] = fr.iou(pm)
    legend(fig, [("TP", TP_C), ("FP", FP_C), ("FN / missed target", FN_C)])
    ious = {t: side["panels"][t] for t, _ in arms}
    fig.suptitle(f"R1-R3 ({fid}, {fr.ref.moment}; frame with the largest R1 recall loss). Feature gates decide whole components (R1) or "
                 f"predicted regions (R2) and can drop the target; the top seed-mass rule (R3) needs no feature. IoU: "
                 + ", ".join(f"{k.split(' ')[0]} {v:.2f}" for k, v in ious.items()), fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(out / "exp-R1-R3-gate-unit.png", dpi=150); plt.close(fig)
    (out / "exp-R1-R3-gate-unit.json").write_text(json.dumps(side, indent=1)); print("exp-R1-R3-gate-unit")


def fig_r4(run, out):
    t1 = rows_of(run, "rounds/R4", "top1-only:N"); ms = rows_of(run, "rounds/R4", "MassTop1:N")
    fid = max(ms, key=lambda f: (t1[f].get("iou") or 0) - (ms[f].get("iou") or 0))
    fr = Frame(run, fid)
    comp = RM.regions(fr.geo, ~fr.geo.support)
    node_m = fr.geo.patch_px @ (fr.fin.maps["N"] / np.maximum(fr.geo.patch_alg_px, 1.0))
    ok = comp >= 0
    rm = np.bincount(comp[ok], weights=node_m[ok], minlength=comp.max() + 1)
    best = int(np.argmax(rm))
    s = seed_and_g(fr, False)
    top1 = to_full(fr, s["idx"], RD.region_gate(s["m_sub"], s["sub"], np.ones(len(s["sub"].pos)), np.zeros(len(s["sub"].pos)), 0.5,
                                                    s["sl_sub"], top1_only=True)[0]) >= 0.5
    c = fr.best_cam()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)); figstyle.style_axes(fig, axes)
    vals = np.where(ok, rm[np.maximum(comp, 0)] / max(rm.max(), 1e-12), np.nan)
    u, v, okp = _project(fr.geo.pos, fr.K[c], fr.T[c])
    axes[0].imshow(fr.imgs[c]); sel = okp & ok
    sc = axes[0].scatter(u[sel], v[sel], s=2, c=vals[sel], cmap=figstyle.sequential_cmap(), lw=0)
    fig.colorbar(sc, ax=axes[0], fraction=0.04); axes[0].set_title("region noun-attention mass (relative)", fontsize=8); axes[0].set_xticks([]); axes[0].set_yticks([])
    pm = comp == best
    fr.draw(axes[1], c, fr.eval_layers(pm), f"R4 largest-mass region  IoU={fr.iou(pm):.2f}")
    fr.draw(axes[2], c, fr.eval_layers(top1), f"R3 top-1 seed-mass region  IoU={fr.iou(top1):.2f}")
    legend(fig, [("TP", TP_C), ("FP", FP_C), ("FN", FN_C)])
    fig.suptitle(f"R4 ({fid}, {fr.ref.moment}): summing attention over a region favours large surfaces (crate) over the small target", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93)); fig.savefig(out / "exp-R4-mass-bias.png", dpi=150); plt.close(fig)
    (out / "exp-R4-mass-bias.json").write_text(json.dumps(dict(frame=fid, rule="largest IoU loss of MassTop1:N vs top1-only:N",
                                                               iou_mass=fr.iou(pm), iou_top1=fr.iou(top1)), indent=1)); print("exp-R4-mass-bias")


def fig_r5(run, out):
    a = rows_of(run, "rounds/R5", "top1-only:N"); b = rows_of(run, "rounds/R5", "top1-only-ns:N")
    fid = max(b, key=lambda f: (b[f].get("iou") or 0) - (a[f].get("iou") or 0))
    fr = Frame(run, fid)
    sa, sb = seed_and_g(fr, False), seed_and_g(fr, True)
    res = []
    for s in (sa, sb):
        n = len(s["sub"].pos)
        res.append(to_full(fr, s["idx"], RD.region_gate(s["m_sub"], s["sub"], np.ones(n), np.zeros(n), 0.5, s["sl_sub"], top1_only=True)[0]) >= 0.5)
    c = fr.best_cam()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10)); figstyle.style_axes(fig, axes.ravel())
    for ci in range(3):
        axes[0, ci].imshow(fr.imgs[ci]); seed_patch_boxes(axes[0, ci], sa["six"], ci, CAT[1]); seed_patch_boxes(axes[0, ci], sb["six"], ci, CAT[0])
        na, nb = int((sa["six"] // 256 == ci).sum()), int((sb["six"] // 256 == ci).sum())
        axes[0, ci].set_title(f"{CAMS[ci]}: R3 seeds {na} (orange), R5 seeds {nb} (blue)", fontsize=8)
        axes[0, ci].set_xticks([]); axes[0, ci].set_yticks([])
    fr.draw(axes[1, 0], c, fr.eval_layers(res[0]), f"R3 seeds -> region  IoU={fr.iou(res[0]):.2f}")
    fr.draw(axes[1, 1], c, fr.eval_layers(res[1]), f"R5 non-support seeds -> region  IoU={fr.iou(res[1]):.2f}")
    axes[1, 2].axis("off")
    legend(fig, [("TP", TP_C), ("FP", FP_C), ("FN", FN_C)])
    fig.suptitle(f"R5 ({fid}, {fr.ref.moment}; frame with the largest R5 gain): seed patches of the two eligibility rules in all cameras (top) "
                 "and the resulting target region (bottom)", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(out / "exp-R5-nonsupport-seed.png", dpi=150); plt.close(fig)
    (out / "exp-R5-nonsupport-seed.json").write_text(json.dumps(dict(frame=fid, rule="largest IoU gain of top1-only-ns vs top1-only",
                                                                    seeds_r3=sa["six"].tolist(), seeds_r5=sb["six"].tolist(),
                                                                    iou_r3=fr.iou(res[0]), iou_r5=fr.iou(res[1])), indent=1)); print("exp-R5-nonsupport-seed")


def fig_r7(run, out):
    g = rows_of(run, "rounds/R7", "G-Q+top1"); f = rows_of(run, "rounds/R7", "FGu-Q:prefix_L8+top1")
    fid = max(f, key=lambda k: (g[k].get("iou") or 0) - (f[k].get("iou") or 0))
    fr = Frame(run, fid, mask="render", feats=("prefix_L8",))
    s = seed_and_g(fr, True); sub, n = s["sub"], len(s["sub"].pos)
    uf = R.normalize(fr.fin.features["prefix_L8"]); mu = R.normalize((s["sw"] @ uf[s["six"]])[None])[0]
    pu, _ = G.proto_unary(uf @ mu, s["six"], s["vp"]); node_p = G.node_patch_mean(sub, pu)
    lam, sv = RN._combine([(s["sl_sub"], np.ones(n)), (fr.cfg.kappa_support * sub.support.astype(float), np.zeros(n)), (np.full(n, fr.cfg.kappa_proto), node_p)])
    mu_sol, _ = G.solve(sub, np.ones(n, bool), G.edge_weights(sub, fr.cfg, None), lam, sv, fr.cfg.lam)
    gq = to_full(fr, s["idx"], RD.region_gate(s["m_sub"], sub, np.ones(n), np.zeros(n), 0.5, s["sl_sub"], top1_only=True)[0]) >= 0.5
    fu = to_full(fr, s["idx"], RD.region_gate(mu_sol, sub, np.ones(n), np.zeros(n), 0.5, s["sl_sub"], top1_only=True)[0]) >= 0.5
    c = fr.best_cam()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)); figstyle.style_axes(fig, axes)
    u, v, ok = _project(sub.pos, fr.K[c], fr.T[c])
    axes[0].imshow(fr.imgs[c]); sc = axes[0].scatter(u[ok], v[ok], s=2, c=node_p[ok], cmap=figstyle.sequential_cmap(), vmin=0, vmax=1, lw=0)
    fig.colorbar(sc, ax=axes[0], fraction=0.04); axes[0].set_title("feature prototype score per surface node (16x16 patch copied)", fontsize=8)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fr.draw(axes[1], c, fr.eval_layers(gq), f"Attention + 3D (no feature)  IoU={fr.iou(gq):.2f}")
    fr.draw(axes[2], c, fr.eval_layers(fu), f"+ feature score as node evidence  IoU={fr.iou(fu):.2f}")
    legend(fig, [("TP", TP_C), ("FP", FP_C), ("FN (target lost)", FN_C)])
    fig.suptitle(f"R7 ({fid}, {fr.ref.moment}): target surface in patches shared with hand / table gets a low feature score and drops out", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93)); fig.savefig(out / "exp-R7-feature-unary-loss.png", dpi=150); plt.close(fig)
    (out / "exp-R7-feature-unary-loss.json").write_text(json.dumps(dict(frame=fid, rule="largest IoU loss of FGu-Q:prefix_L8+top1 vs G-Q+top1",
                                                                       iou_g=fr.iou(gq), iou_fgu=fr.iou(fu)), indent=1)); print("exp-R7-feature-unary-loss")


def fig_r9(run, out):
    cand = rows_of(run, "rounds/R6", "top1-only-ns+rm:N"); b0 = rows_of(run, "baseline_rm", "T54region+rm:N")
    gains = sorted(cand, key=lambda f: (cand[f].get("iou") or 0) - (b0.get(f, {}).get("iou") or 0))
    picks = [("candidate ahead", gains[-1]), ("B0 ahead", gains[0])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5)); figstyle.style_axes(fig, axes.ravel())
    side = dict(rule="largest and smallest IoU(candidate) - IoU(B0+render mask) frame", rows={})
    for row, (lab, fid) in enumerate(picks):
        fr = Frame(run, fid, mask="render")
        s = seed_and_g(fr, True); n = len(s["sub"].pos)
        cm = to_full(fr, s["idx"], RD.region_gate(s["m_sub"], s["sub"], np.ones(n), np.zeros(n), 0.5, s["sl_sub"], top1_only=True)[0]) >= 0.5
        with np.load(run / "baseline_rm/raw" / f"{fid}.npz") as z:
            pts = z["N:points"] if "N:points" in z.files else np.zeros((0, 3))
        meta = json.loads((run / "baseline_rm/raw" / f"{fid}.json").read_text())
        vx = meta["fusion_voxel"]
        cells = {tuple(x) for x in np.floor(pts / vx).astype(np.int64)}
        bm = np.fromiter((tuple(x) in cells for x in np.floor(fr.geo.pos / vx).astype(np.int64)), bool, len(fr.geo.pos))
        c = fr.best_cam()
        axes[row, 0].imshow(fr.imgs[c]); axes[row, 0].set_title(f"{lab}: {fid} ({fr.ref.moment}), {CAMS[c]}", fontsize=8)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        fr.draw(axes[row, 1], c, fr.eval_layers(bm), f"B0 serving cluster (render mask)  IoU={b0[fid].get('iou') or 0:.2f}")
        fr.draw(axes[row, 2], c, fr.eval_layers(cm), f"Attention + 3D candidate  IoU={cand[fid].get('iou') or 0:.2f}")
        side["rows"][lab] = dict(frame=fid, iou_b0=b0[fid].get("iou"), iou_candidate=cand[fid].get("iou"))
    legend(fig, [("TP", TP_C), ("FP", FP_C), ("FN", FN_C)])
    fig.suptitle("R9, same render robot mask for both: frame where the Attention + 3D candidate is most ahead of the serving cluster (top) "
                 "and frame where serving is most ahead (bottom)", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94)); fig.savefig(out / "exp-R9-vs-serving.png", dpi=150); plt.close(fig)
    (out / "exp-R9-vs-serving.json").write_text(json.dumps(side, indent=1)); print("exp-R9-vs-serving")


def fig_r10(run, out):
    fid_m, fid = "T53_ep1822_f005", "T53_ep1822_f019"
    fm, fc = Frame(run, fid_m, mask="render", feats=("prefix_L8",)), Frame(run, fid, mask="render", feats=("prefix_L8",))
    cm, cc = M.r6_choice(fm.geo, fm.cfg, fm.fin.maps["N"]), M.r6_choice(fc.geo, fc.cfg, fc.fin.maps["N"])
    comp = RM.regions(fc.geo, ~fc.geo.support); ok = comp >= 0; px = fc.geo.patch_px.tocsr()
    c = fc.best_cam()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2)); figstyle.style_axes(fig, axes)
    fm.draw(axes[0], fm.best_cam(cm["chosen"]), [(cm["chosen"], CAT[2], 2)], f"memory frame {fid_m} ({fm.ref.moment}): R6 choice stored")
    side = dict(memory_frame=fid_m, frame=fid, rule="an R6-failed frame (IoU 0) whose target region exists", panels={})
    for ax, name in zip(axes[1:], ("raw cosine", "centered cosine")):
        Um, Uc = R.normalize(fm.fin.features["prefix_L8"]), R.normalize(fc.fin.features["prefix_L8"])
        if name.startswith("centered"):
            Um, Uc = R.normalize(Um - Um.mean(0)), R.normalize(Uc - Uc.mean(0))
        mem = M.unit_rows(np.asarray(fm.geo.patch_px.tocsr()[cm["chosen"]].sum(0)).ravel(), Um)
        val = np.full(len(fc.geo.pos), np.nan); rows = []
        for r in np.unique(comp[ok]):
            k = comp == r
            if k.sum() < 20 or (fc.geo.pos[k].max(0) - fc.geo.pos[k].min(0)).max() > 0.4:
                continue
            cs = float(M.unit_rows(np.asarray(px[k].sum(0)).ravel(), Uc) @ mem); val[k] = cs
            rows.append((cs, fc.classes[int(fc.geo.eval_class_frac[fc.geo.node_eval][k].sum(0).argmax())]))
        u, v, okp = _project(fc.geo.pos, fc.K[c], fc.T[c]); sel = okp & np.isfinite(val)
        ax.imshow(fc.imgs[c]); sc = ax.scatter(u[sel], v[sel], s=2, c=val[sel], cmap=figstyle.sequential_cmap(), lw=0)
        fig.colorbar(sc, ax=ax, fraction=0.04)
        rows.sort(reverse=True)
        ax.set_title(f"{name} to memory: " + ", ".join(f"{cl} {v_:.3f}" for v_, cl in rows[:3]), fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        side["panels"][name] = rows[:5]
    fig.suptitle(f"R10/R10b ({fid}, {fc.ref.moment}): raw VLA features make every region ~0.97 similar; removing the shared component "
                 "(centering) separates the target", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out / "exp-R10-centering.png", dpi=150); plt.close(fig)
    (out / "exp-R10-centering.json").write_text(json.dumps(side, indent=1, default=str)); print("exp-R10-centering")


def fig_r11(run, out):
    C = [json.loads(l) for l in open(run / "rounds/R11b/conf.jsonl")]
    C = [c for c in C if c.get("q_mem_chain:prefix_L8") is not None and c["target_voxels"] > 0]
    thr = json.loads((run / "LOCK_heldout.json").read_text())["arms"]["confidence_gate"]["abstain_if_below"]
    C.sort(key=lambda c: c["q_mem_chain:prefix_L8"])
    show = C[:6] + C[-2:]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5)); figstyle.style_axes(fig, axes.ravel())
    side = dict(threshold=thr, rule="6 lowest + 2 highest chain-memory scores (centered)", frames=[])
    for ax, c in zip(axes.ravel(), show):
        fr = Frame(run, c["frame_id"], mask="render")
        ch = M.r6_choice(fr.geo, fr.cfg, fr.fin.maps["N"])["chosen"]
        cam = fr.best_cam(ch if ch.any() else None)
        flag = c["q_mem_chain:prefix_L8"] < thr; wrong = (c["r6_iou"] or 0) < 0.1
        fr.draw(ax, cam, [(fr.gt & ~ch, FN_C, 1.5), (ch, CAT[1] if wrong else CAT[2], 2)],
                f"{c['frame_id']} {c['moment'][:2]}\nscore {c['q_mem_chain:prefix_L8']:.2f} -> {'ALERT' if flag else 'ok'} | chosen IoU {c['r6_iou'] or 0:.2f}")
        side["frames"].append(dict(frame=c["frame_id"], score=c["q_mem_chain:prefix_L8"], alert=bool(flag), wrong=bool(wrong)))
    legend(fig, [("chosen region, correct", CAT[2]), ("chosen region, wrong", CAT[1]), ("true target", FN_C)])
    alerts = [f for f in side["frames"] if f["alert"]]
    n_wrong_all = sum((c["r6_iou"] or 0) < 0.1 for c in C)
    n_alert_all = sum(c["q_mem_chain:prefix_L8"] < thr for c in C)
    n_alert_wrong = sum((c["q_mem_chain:prefix_L8"] < thr) and ((c["r6_iou"] or 0) < 0.1) for c in C)
    side.update(all_frames=len(C), wrong_frames=n_wrong_all, alerts=n_alert_all, alerts_that_are_wrong=n_alert_wrong)
    fig.suptitle(f"R11 confidence gate (locked threshold {thr:.3f}): over {len(C)} frames, {n_alert_all} alerts cover {n_alert_wrong} of the "
                 f"{n_wrong_all} wrong choices; the rest are false alarms. Shown: 6 lowest and 2 highest scores", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94)); fig.savefig(out / "exp-R11-confidence-gate.png", dpi=150); plt.close(fig)
    (out / "exp-R11-confidence-gate.json").write_text(json.dumps(side, indent=1)); print("exp-R11-confidence-gate")


def fig_r12(run, out):
    rows = [json.loads(l) for l in open(run / "rounds/R12/features.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["episode_id"]].append(r)
    eps = [e for e, v in by.items() if len(v) == 7][:3]
    fig, axes = plt.subplots(len(eps), 1, figsize=(12, 3.2 * len(eps))); figstyle.style_axes(fig, axes)
    side = dict(rule="first three 7-frame (successful) episodes", episodes={})
    for ax, e in zip(axes, eps):
        v = sorted(by[e], key=lambda r: r["moment"])
        x = np.arange(len(v))
        ax.plot(x, [r["hand_d"] for r in v], "o-", color=CAT[0], label="target-hand distance (m)")
        ax.plot(x, [r["height"] for r in v], "s-", color=CAT[2], label="target height in base frame (m)")
        ax.plot(x, [1 - r["grip"] for r in v], "^-", color=CAT[3], label="gripper closing (1 - opening)")
        ax.plot(x, [r["hsr"][1] for r in v], "d--", color=CAT[4], label="HSR p(place)")
        for xi, r in zip(x, v):
            ax.axvspan(xi - 0.5, xi + 0.5, color={"pre": CAT[0], "held": CAT[1], "placed": CAT[5]}.get(r["grasp_state"], "white"), alpha=0.06)
        ax.set_xticks(x); ax.set_xticklabels([r["moment"] for r in v], fontsize=7); ax.set_title(e, fontsize=8)
        side["episodes"][e] = [{k: r[k] for k in ("moment", "grasp_state", "hand_d", "height", "grip")} for r in v]
    axes[0].legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.45))
    fig.suptitle("R12: 3D relations of the target region separate approach / held / placed where HSR p(place) is ambiguous (background = GT grasp state)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / "exp-R12-phase-3d.png", dpi=150); plt.close(fig)
    (out / "exp-R12-phase-3d.json").write_text(json.dumps(side, indent=1)); print("exp-R12-phase-3d")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    figs = dict(S0=fig_s0, R1R3=fig_r1_r3, R4=fig_r4, R5=fig_r5, R7=fig_r7, R9=fig_r9, R10=fig_r10, R11=fig_r11, R12=fig_r12)
    for k, f in figs.items():
        if a.only and k not in a.only:
            continue
        f(a.run, a.out)


if __name__ == "__main__":
    main()
