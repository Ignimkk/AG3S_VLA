"""Figure 3 v2 — ATS identity panel: the remembered target descriptor and today's candidate regions. CPU, read only.

  cd /mnt/dev/AG3S_VLA-development; MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python \
      paper/tools/plotting/fig3_v2_identity.py --outdir paper/figures/paperbanana/figure3/assets_v2

Same recorded run and descriptor code as `ifg_overview_render.region_scores` (T53 C1a ep1842, frozen-VLA prefix layer-8 patch
features, centered by the frame mean, pooled over a region by pixel count; Eq. descriptor of the method). Memory = the target
region chosen at the approach frame f010; candidates = size-filtered non-support components of frame f016.
Outputs (no text in any image):
  f3v2_id_memory.png   memory frame, remembered region in orange
  f3v2_id_now.png      current frame, each candidate region coloured by its centered cosine to the memory (grey -> orange ramp)
  f3v2_id_codes.png    feature codes: one row per descriptor (memory, then candidates by score), the K dimensions where the
                       memory descriptor has the largest magnitude, diverging colours (same scale for all rows)
"""
import argparse, json, pathlib, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ifg_overview_render as V  # noqa: E402

FM, FC = "T53_ep1842_f010", "T53_ep1842_f016"
GREY, ORANGE = np.array([0.80, 0.80, 0.80]), np.array([0.91, 0.44, 0.10])


def descriptors(fm, fc):
    M, R, RM = V.M, V.R, V.RM
    cm = M.r6_choice(fm.geo, fm.cfg, fm.fin.maps["N"])["chosen"]
    Um, Uc = R.normalize(fm.fin.features["prefix_L8"]), R.normalize(fc.fin.features["prefix_L8"])
    Um, Uc = R.normalize(Um - Um.mean(0)), R.normalize(Uc - Uc.mean(0))
    mem = M.unit_rows(np.asarray(fm.geo.patch_px.tocsr()[cm].sum(0)).ravel(), Um)
    comp = RM.regions(fc.geo, ~fc.geo.support); ok = comp >= 0; px = fc.geo.patch_px.tocsr()
    cands = []
    for r in np.unique(comp[ok]):
        k = comp == r
        if k.sum() < 20 or (fc.geo.pos[k].max(0) - fc.geo.pos[k].min(0)).max() > 0.4:
            continue
        g = M.unit_rows(np.asarray(px[k].sum(0)).ravel(), Uc)
        cl = fc.classes[int(fc.geo.eval_class_frac[fc.geo.node_eval][k].sum(0).argmax())]
        cands.append(dict(g=np.asarray(g).ravel(), k=k, cls=cl, cos=float(np.asarray(g).ravel() @ np.asarray(mem).ravel())))
    return cm, np.asarray(mem).ravel(), sorted(cands, key=lambda c: -c["cos"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True); ap.add_argument("--dims", type=int, default=32); ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--dist", type=float, default=0.42); ap.add_argument("--az", type=float, default=150); ap.add_argument("--el", type=float, default=-34)
    a = ap.parse_args()
    from PIL import Image
    out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    fm = V.XF.Frame(V.RUN, FM, mask="render", feats=("prefix_L8",)); fc = V.XF.Frame(V.RUN, FC, mask="render", feats=("prefix_L8",))
    cm, mem, cands = descriptors(fm, fc)
    # memory frame
    ctr_m = fm.geo.pos[cm].mean(0)
    rp = V.Replay(FM); cam_m = V.cam_at(ctr_m, a.dist, a.az, a.el)
    Image.fromarray(rp.render(cam_m, [(fm.geo.pos[cm], (*ORANGE, 1.0), V.HALF)])).save(out / "f3v2_id_memory.png"); rp.close()
    # current frame: candidates coloured by similarity (cos <= 0 -> grey, max -> orange)
    top = max(c["cos"] for c in cands)
    layers = []
    for c in cands:
        t = float(np.clip(c["cos"] / top, 0, 1)) ** 2
        layers.append((fc.geo.pos[c["k"]], (*((1 - t) * GREY + t * ORANGE), 1.0), V.HALF))
    ctr = fc.geo.pos[cands[0]["k"]].mean(0)
    rp = V.Replay(FC); cam = V.cam_at(ctr, a.dist + 0.06, a.az, a.el)
    Image.fromarray(rp.render(cam, layers)).save(out / "f3v2_id_now.png"); rp.close()
    # feature codes
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dims = np.argsort(-np.abs(mem))[:a.dims]
    rows = [mem] + [c["g"] for c in cands[:a.rows]]
    X = np.array([r[dims] for r in rows]); s = np.abs(X).max()
    fig = plt.figure(figsize=(a.dims * 0.12, len(rows) * 0.30), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    gap = np.full((1, a.dims), np.nan)   # a blank line between memory and candidates
    ax.imshow(np.vstack([X[:1], gap, X[1:]]), cmap="PuOr_r", vmin=-s, vmax=s, aspect="auto", interpolation="nearest")
    for y in np.arange(-0.5, len(rows) + 0.5, 1):
        ax.axhline(y, color="white", lw=1.2)
    for x in np.arange(-0.5, a.dims, 1):
        ax.axvline(x, color="white", lw=0.6)
    fig.savefig(out / "f3v2_id_codes.png", dpi=400, facecolor="white"); plt.close(fig)
    side = {"script": "paper/tools/plotting/fig3_v2_identity.py", "run": str(V.RUN), "memory_frame": FM, "current_frame": FC,
            "feature": "prefix_L8 patch features of the frozen VLA, centered by the frame mean, pooled by region pixel count, unit norm",
            "camera_memory": cam_m, "camera_now": cam,
            "candidates": [{"class": c["cls"], "nodes": int(c["k"].sum()), "centered_cosine_to_memory": round(c["cos"], 4)} for c in cands],
            "codes": {"rows": ["memory"] + [c["cls"] for c in cands[:a.rows]], "dims": dims.tolist(),
                      "rule": "dimensions with the largest |memory descriptor|; colour PuOr_r symmetric at the largest |value| shown"},
            "colour_now": "grey (cos <= 0) -> orange (highest cos), t = (cos / max cos)^2",
            "is_measured_result": False, "note": "Illustration from one recorded frame pair; no number printed in any image."}
    (out / "f3v2_identity.json").write_text(json.dumps(side, indent=1, default=float))
    print(json.dumps(side["candidates"][:5]))


if __name__ == "__main__":
    main()
