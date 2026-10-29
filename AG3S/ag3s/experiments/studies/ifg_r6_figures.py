"""IFG R4-R6 figures (new names; R0-R3 figures are kept as evidence):
  ifg-rounds-r4r6.png   paired IoU / recall vs G:N for R3-R6 arms + per-moment IoU (R5, R6, B0)
  ifg-scene-r6-<frame>.png  held frame: serving FK sphere mask vs rendered-robot mask, and the
                        top1-only-ns prediction with each (5 mm nodes on the policy image)
  PYTHONPATH=. MUJOCO_GL=osmesa .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_r6_figures --run $RUN --out AG3S/ag3s/docs/figures/ifg
"""
import argparse, collections, json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_graph as G, ifg_reader as R, ifg_round_ns as NS, ifg_summary as S  # noqa: E402
from AG3S.ag3s.experiments.studies.ifg_figures import _project  # noqa: E402

ARMS = [("R3", "top1-only:N"), ("R4", "MassTop1:N"), ("R4", "O-region-best"), ("R5", "G-ns:N"), ("R5", "top1-only-ns:N"),
        ("R6", "G+rm:N"), ("R6", "top1-only+rm:N"), ("R6", "top1-only-ns+rm:N"), ("R6", "T54region:N")]
MOMENTS = ["F1", "F2", "F3", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]


def rounds(run, out):
    data = []
    for rnd, arm in ARMS:
        comps = json.loads((run / f"rounds/{rnd}/paired_summary.json").read_text())["comparisons"]
        got = {}
        for c in comps:
            if c["metric"] not in ("iou", "recall"):
                continue
            if c["a"][0] == arm and c["b"][0] == "G:N":
                got[c["metric"]] = dict(diff=c["diff"], ci95=c["ci95"])
            if arm == "T54region:N" and c["a"][0] == "G:N" and c["b"][0] == arm:
                got[c["metric"]] = dict(diff=-c["diff"], ci95=[-c["ci95"][1], -c["ci95"][0]])
        data.append(dict(round=rnd, arm=arm, **got))
    rows, _ = S.load_rows(run / "rounds/R6", (run / "baseline",), S.s0_valid(run / "s0"))
    F = collections.defaultdict(dict)
    for r in rows:
        F[r["frame_id"]][r["arm_id"]] = r
    mom = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in F.values():
        for a in ("top1-only-ns:N", "top1-only-ns+rm:N", "T54region:N"):
            mom[a][d["G:N"]["moment"][:2]].append(d[a].get("iou") or 0)
    momm = {a: {m: float(np.mean(v[m])) for m in MOMENTS} for a, v in mom.items()}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    figstyle.style_axes(fig, axes)
    labels = [f"{d['round']}  {d['arm']}" for d in data]
    for ax, metric, line in ((axes[0], "iou", 0.02), (axes[1], "recall", -0.02)):
        for i, d in enumerate(data):
            m = d[metric]
            k = 2 if d["arm"].startswith("O-") else (7 if d["arm"].startswith("T54") else (0 if "+rm" in d["arm"] else 1))
            ax.errorbar(m["diff"], i, xerr=[[m["diff"] - m["ci95"][0]], [m["ci95"][1] - m["diff"]]], fmt="o", color=figstyle.CATEGORICAL[k], capsize=2)
        ax.axvline(0, color=figstyle.INK, lw=0.8); ax.axvline(line, color=figstyle.INK_2, lw=0.8, ls="--")
        ax.set_xlabel(f"paired episode {metric} diff vs G:N (95% CI, 32 ep)")
    axes[0].set_yticks(range(len(labels))); axes[0].set_yticklabels(labels, fontsize=8); axes[0].invert_yaxis()
    axes[1].set_yticks(range(len(labels))); axes[1].set_yticklabels([]); axes[1].invert_yaxis()
    x = np.arange(len(MOMENTS)); w = 0.27
    for j, (a, lab, k) in enumerate((("top1-only-ns:N", "R5 (serving FK mask)", 1), ("top1-only-ns+rm:N", "R6 (render mask)", 0), ("T54region:N", "B0 serving region", 7))):
        axes[2].bar(x + (j - 1) * w, [momm[a][m] for m in MOMENTS], w, color=figstyle.CATEGORICAL[k], label=lab)
    axes[2].set_xticks(x); axes[2].set_xticklabels(MOMENTS); axes[2].set_ylim(0, 1.05); axes[2].set_ylabel("frame-mean IoU per moment")
    axes[2].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("IFG rounds R4-R6 (dev, 180 frames). orange = serving FK mask, blue = render mask, teal = GT oracle, red = B0 (serving filter)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "ifg-rounds-r4r6.png", dpi=150); plt.close(fig)
    (out / "ifg-rounds-r4r6.json").write_text(json.dumps(dict(reference="G:N", rows=data, per_moment_iou=momm), indent=1))
    return F


def scene(run, out, F):
    # pre-fixed pick: the M3/M4 frame with the largest R6-R5 IoU gain
    cand = [(F[f]["top1-only-ns+rm:N"].get("iou", 0) - (F[f]["top1-only-ns:N"].get("iou") or 0), f) for f in F
            if F[f]["G:N"]["moment"][:2] in ("M3", "M4")]
    fid = max(cand)[1]
    ref = {r.frame_id: r for r in R.list_frames()}[fid]
    cfg = G.GraphConfig()
    fin = R.load_frame(ref, features=())
    with np.load(run / "s0/replay" / f"{fid}.npz") as z:
        depth, fk, K, T = z["depth"], z["robot_mask"], z["K"], z["T_base_cam"]
    with np.load(run / "rounds/R6/robot_mask" / f"{fid}.npz") as z:
        rm = z["robot_mask"]
    with np.load(run / "s0/labels" / f"{fid}.npz") as z:
        cm, classes = z["class_map"], [str(c) for c in z["classes"]]
    with np.load(ref.path / "inputs.npz") as z:
        imgs = [np.clip((z[f"image_{c}"] + 1) / 2, 0, 1) for c in R.DUMP_CAMERAS]
    c = 2  # right wrist camera (held fruit)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4)); figstyle.style_axes(fig, axes)
    for ax, mk, title in ((axes[0], fk, "serving FK sphere mask"), (axes[1], rm, "rendered-robot mask")):
        ax.imshow(imgs[c]); small = mk[c][::480 // 224 + 1, ::640 // 224 + 1]
        ax.imshow(np.ma.masked_where(~mk[c], mk[c]), extent=(-.5, 223.5, 223.5, -.5), cmap="autumn", alpha=0.45)
        tg = (cm[c] == 0) & mk[c]
        ax.set_title(f"{title}: target px removed {int(tg.sum())}", fontsize=9)
    side = dict(frame_id=fid, camera="wrist_cam_r", panels={})
    for ax, mk, arm in ((axes[2], fk, "top1-only-ns:N"), (axes[3], rm, "top1-only-ns+rm:N")):
        geo = G.build_geometry(depth, K, T, mk, cm, classes, cfg)
        res = NS.g_and_top1(geo, cfg, fin.maps["N"], NS.eligible(geo, cfg, True))
        sub, _, _, mt, _ = res
        pred = mt >= 0.5; gt = geo.eval_target[sub.node_eval] >= 0.5
        u, v, ok = _project(sub.pos, K[c], T[c])
        ax.imshow(imgs[c])
        for sel, k in ((ok & pred & gt, 2), (ok & pred & ~gt, 1), (ok & ~pred & gt, 0)):
            ax.scatter(u[sel], v[sel], s=2, c=figstyle.CATEGORICAL[k], lw=0)
        r = F[fid][arm]
        ax.set_title(f"{arm}  IoU={r['iou']:.2f} (TP green, FP orange, FN blue)", fontsize=8)
        side["panels"][arm] = dict(iou=r["iou"], recall=r["recall"], precision=r.get("precision"))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{fid} ({ref.moment}) right wrist camera: held fruit removed by the sphere self-filter vs kept by the render mask", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"ifg-scene-r6-{fid}.png", dpi=150); plt.close(fig)
    (out / f"ifg-scene-r6-{fid}.json").write_text(json.dumps(side, indent=1, default=float))
    return fid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    F = rounds(a.run, a.out)
    print(scene(a.run, a.out, F))


if __name__ == "__main__":
    main()
