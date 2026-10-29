"""Real-scene renders for the IFG-revised overview figure (slide / Figure-2 style montage).

Every image is a MuJoCo replay of a recorded dev frame (qpos from `$RUN/s0/replay/<frame>.npz`), rendered at
high resolution from a free camera. Overlays are the 5 mm surface nodes of that frame, recomputed with the
IFG study code (read-only import): noun-attention seeds, the graph-energy top-1 region, the serving
radius + DBSCAN cluster (B0), robot masks, and centered / raw feature agreement with a remembered region.
Frames are chosen by the same fixed rules as the IFG experiment figures (recorded in each sidecar).
The robot base frame equals the MuJoCo world frame in these scenes (checked: camera-derived transform = I).

Run (CPU renderer, no GPU):
    MUJOCO_GL=osmesa PYTHONPATH=. .venv-ag3s/bin/python paper/tools/plotting/ifg_overview_render.py [--probe FRAME]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from AG3S.ag3s.experiments.sources.policy_record import load_run, replay_scene  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_experiment_figures as XF  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_memory as M  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_reader as R  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_round as RD  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_round_mass as RM  # noqa: E402

RUN = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba")
OUT = REPO / "paper/figures/ifg_overview/assets"

ORANGE = (0.91, 0.44, 0.10, 1.0)     # target region (correct)
RED = (0.80, 0.08, 0.08, 1.0)        # wrongly included / wrong choice
BLUE = (0.18, 0.46, 0.71, 1.0)       # protected neighbour surface
GOLD = (1.00, 0.80, 0.05, 1.0)       # seeds
GHOST = (0.91, 0.44, 0.10, 0.25)     # target surface missed / removed
HALF = 0.0026                        # drawn half-size of a 5 mm node


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Replay:
    """MuJoCo scene of one record, posed at a dev frame."""

    def __init__(self, fid, W=1500, H=1125):
        self.ref = XF.refs()[fid]
        self.scene = replay_scene(load_run(R.record_dir(self.ref)))
        self.m, self.d = self.scene.model, self.scene.data
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = max(W, 1600), max(H, 1200)
        with np.load(RUN / "s0/replay" / f"{fid}.npz") as z:
            self.d.qpos[:] = z["qpos"]
            self.cams = [str(c) for c in z["cameras"]]
        mujoco.mj_forward(self.m, self.d)
        self.r = mujoco.Renderer(self.m, H, W, max_geom=40000)

    def render(self, cam, layers=(), mjcam=None):
        if mjcam is not None:
            self.r.update_scene(self.d, mjcam)
        else:
            c = mujoco.MjvCamera()
            c.lookat[:], c.distance, c.azimuth, c.elevation = cam["lookat"], cam["dist"], cam["az"], cam["el"]
            self.r.update_scene(self.d, c)
        sc = self.r.scene
        for pts, rgba, half in layers:
            for p in np.asarray(pts, float):
                if sc.ngeom >= sc.maxgeom:
                    break
                mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([half] * 3), p,
                                    np.eye(3).ravel(), np.asarray(rgba, np.float32))
                sc.ngeom += 1
        return self.r.render().copy()

    def close(self):
        self.r.close(); self.scene.close()


def top1(fr, s):
    n = len(s["sub"].pos)
    return XF.to_full(fr, s["idx"], RD.region_gate(s["m_sub"], s["sub"], np.ones(n), np.zeros(n), 0.5, s["sl_sub"],
                                                   top1_only=True)[0]) >= 0.5


def seed_nodes(fr, six):
    px = fr.geo.patch_px.tocsr()
    return np.asarray(px[:, six].sum(1)).ravel() > 0


def neighbours(fr, centre, r=0.16):
    """Observed non-target, non-support nodes near the target (protected neighbour surfaces)."""
    near = np.linalg.norm(fr.geo.pos - centre[None], axis=1) <= r
    return near & ~fr.gt & ~fr.geo.support


def b0_nodes(fr, fid):
    with np.load(RUN / "baseline_rm/raw" / f"{fid}.npz") as z:
        pts = z["N:points"] if "N:points" in z.files else np.zeros((0, 3))
    vx = json.loads((RUN / "baseline_rm/raw" / f"{fid}.json").read_text())["fusion_voxel"]
    cells = {tuple(x) for x in np.floor(pts / vx).astype(np.int64)}
    return np.fromiter((tuple(x) in cells for x in np.floor(fr.geo.pos / vx).astype(np.int64)), bool, len(fr.geo.pos))


def cam_at(p, dist, az, el):
    return dict(lookat=np.asarray(p, float).tolist(), dist=dist, az=az, el=el)


SIDE = {}


def save(name, img, fid, note, extra):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    Image.fromarray(img).save(path)
    meta = dict(asset=str(path.relative_to(REPO)), frame=fid, run=str(RUN),
                renderer=f"MuJoCo {mujoco.__version__} replay of recorded qpos (s0/replay)",
                script="paper/tools/plotting/ifg_overview_render.py",
                sources={str(p): sha(p) for p in [RUN / "s0/replay" / f"{fid}.npz"]},
                note=note, is_measured_result=False, **extra)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    SIDE[name] = meta
    print("wrote", path.relative_to(REPO))


def eval_layers(fr, pred, extra=()):
    return [*extra, (fr.geo.pos[pred & fr.gt], ORANGE, HALF), (fr.geo.pos[pred & ~fr.gt], RED, HALF)]


# ------------------------------------------------------------------------------------- panels
def c1_seeds(view):
    """R5 rule frame: seeds of the support-inclusive rule land on the table; non-support eligibility fixes it."""
    a = XF.rows_of(RUN, "rounds/R5", "top1-only:N"); b = XF.rows_of(RUN, "rounds/R5", "top1-only-ns:N")
    fid = max(b, key=lambda f: (b[f].get("iou") or 0) - (a[f].get("iou") or 0))
    fr = XF.Frame(RUN, fid, mask="render")
    sa, sb = XF.seed_and_g(fr, False), XF.seed_and_g(fr, True)
    ra, rb = top1(fr, sa), top1(fr, sb)
    ctr = fr.geo.pos[fr.gt].mean(0)
    seeds_a = sa["sl"] > 0
    rp = Replay(fid)
    cam = cam_at(0.5 * ctr + 0.5 * fr.geo.pos[seeds_a].mean(0), *view)
    neigh = neighbours(fr, ctr)
    img = rp.render(cam, [(fr.geo.pos[fr.gt & ~ra], GHOST, HALF), (fr.geo.pos[seeds_a], GOLD, HALF * 1.1)] + eval_layers(fr, ra))
    save("c1_prior_seeds_on_support", img, fid, "Seed mass of the top-5 noun-attention patches when support-dominated patches are "
         "eligible (R3 rule): gold = nodes with seed mass; faint orange = target surface; resulting region empty.",
         dict(rule="largest IoU gain of top1-only-ns:N vs top1-only:N (R5)", iou=fr.iou(ra), seeds=sa["six"].tolist(),
              seed_nodes=int(seeds_a.sum()), seed_nodes_on_support=int((seeds_a & fr.geo.support).sum()),
              seed_nodes_on_target=int((seeds_a & fr.gt).sum()), camera=cam))
    img = rp.render(cam, [(fr.geo.pos[neigh & ~rb], BLUE, HALF)] + eval_layers(fr, rb))
    seeds_b = sb["sl"] > 0
    save("c1_ours_seed_region", img, fid, "Non-support eligible seeds -> graph-energy top-1 region (orange); nearby non-target surfaces blue.",
         dict(iou=fr.iou(rb), seeds=sb["six"].tolist(), seed_nodes=int(seeds_b.sum()),
              seed_nodes_on_target=int((seeds_b & fr.gt).sum()), camera=cam))
    # policy-camera view with the noun attention heatmap and seed patch boxes: the camera holding most seeds
    # camera whose seed patches contain the most target pixels; crop around the target there
    cm = fr.class_map; tgt_cls = 0
    def tpx(p):
        cc, r, col = p // 256, (p % 256) // 16, p % 16
        h, w = cm.shape[1] // 16, cm.shape[2] // 16
        return int((cm[cc, r * h:(r + 1) * h, col * w:(col + 1) * w] == tgt_cls).sum())
    per = np.zeros(3, int)
    for p in sb["six"]:
        per[int(p) // 256] += tpx(int(p))
    c = int(per.argmax())
    img = rp.render(None, mjcam=rp.cams[c])
    att = fr.fin.maps["N"][c * 256:(c + 1) * 256].reshape(16, 16)
    ys, xs = np.nonzero(cm[c] == tgt_cls)
    centre = (xs.mean() / cm.shape[2], ys.mean() / cm.shape[1]) if len(xs) else (0.5, 0.5)
    save_attention("c1_ours_noun_attention", img, att, sb["six"], c, fid,
                   dict(camera=rp.cams[c], map="N (prefix L10 H7 noun attention)", target_px_in_seed_patches=per.tolist(),
                        crop_centre_uv_frac=centre), crop=centre)
    rp.close()


def save_attention(name, img, att, six, c, fid, extra, crop=None, half=0.22):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    H, W = img.shape[:2]
    a = np.asarray(Image.fromarray((att / att.max() * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), float) / 255
    heat = cm.get_cmap("inferno")(a)[..., :3]
    alpha = np.clip(a * 1.3, 0, 0.75)[..., None]
    out = (img / 255 * (1 - alpha) + heat * alpha)
    out = (out * 255).astype(np.uint8)
    im = Image.fromarray(out)
    from PIL import ImageDraw
    dr = ImageDraw.Draw(im)
    sx, sy = W / 16, H / 16
    for p in six:
        if p // 256 != c:
            continue
        r, col = divmod(p % 256, 16)
        dr.rectangle([col * sx, r * sy, (col + 1) * sx, (r + 1) * sy], outline=(255, 205, 13), width=5)
    if crop is not None:
        cx, cy = crop[0] * W, crop[1] * H; hw = half * W; hh = hw * 0.75
        cx, cy = min(max(cx, hw), W - hw), min(max(cy, hh), H - hh)
        im = im.crop((int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh))).resize((W, H), Image.LANCZOS)
    save(name, np.asarray(im), fid, "Policy camera re-rendered at high resolution; 16x16 noun-attention map upsampled; gold = seed patches.", extra)


def c2_extent(view, view_rm):
    cand = XF.rows_of(RUN, "rounds/R6", "top1-only-ns+rm:N"); b0 = XF.rows_of(RUN, "baseline_rm", "T54region+rm:N")
    # largest B0 neighbour false positive among frames with B0 IoU > 0.2 where the candidate has none
    fid = max((f for f in b0 if f in cand and (b0[f].get("iou") or 0) > 0.2 and not cand[f].get("neighbor_fp")),
              key=lambda f: b0[f].get("neighbor_fp") or 0)
    fr = XF.Frame(RUN, fid, mask="render")
    s = XF.seed_and_g(fr, True); cm_ = top1(fr, s); bm = b0_nodes(fr, fid)
    ctr = fr.geo.pos[fr.gt].mean(0)
    look = 0.5 * ctr + 0.5 * fr.geo.pos[bm].mean(0) if bm.any() else ctr
    rp = Replay(fid); cam = cam_at(look, *view)
    neigh = neighbours(fr, ctr)
    img = rp.render(cam, [(fr.geo.pos[fr.gt & ~bm], GHOST, HALF)] + eval_layers(fr, bm))
    save("c2_prior_radius_cluster", img, fid, "Serving radius growth + DBSCAN cluster (B0, same render mask): orange = target, red = non-target.",
         dict(rule="max B0 neighbor_fp | B0 IoU > 0.2, candidate neighbor_fp = 0", iou=b0[fid].get("iou"), neighbor_fp=b0[fid].get("neighbor_fp"), camera=cam))
    img = rp.render(cam, [(fr.geo.pos[neigh & ~cm_], BLUE, HALF)] + eval_layers(fr, cm_))
    save("c2_ours_surface_region", img, fid, "Depth-continuous surface graph, top-1 seed-mass region; neighbour surfaces stay blue (protected).",
         dict(iou=cand[fid].get("iou"), camera=cam))
    rp.close()
    # held fruit: sphere self-filter vs rendered robot mask
    fid = "T54c_ep1817_f020"
    fs, frm = XF.Frame(RUN, fid, mask="fk"), XF.Frame(RUN, fid, mask="render")
    ctr = frm.geo.pos[frm.gt].mean(0)
    rp = Replay(fid); cam = cam_at(ctr, *view_rm)
    from scipy.spatial import cKDTree
    gone = cKDTree(fs.geo.pos[fs.gt]).query(frm.geo.pos[frm.gt])[0] > 0.004
    img_s = rp.render(cam, [(fs.geo.pos[fs.gt], ORANGE, HALF), (frm.geo.pos[frm.gt][gone], RED, HALF)])
    img_r = rp.render(cam, [(frm.geo.pos[frm.gt], ORANGE, HALF)])
    extra = dict(rule="R6 example frame of the paper figure", target_nodes_sphere=int(fs.gt.sum()), target_nodes_render=int(frm.gt.sum()),
                 render_nodes_absent_under_sphere=int(gone.sum()), camera=cam)
    save("c2_prior_sphere_mask", img_s, fid, "Target nodes kept by the serving sphere/capsule self-filter (orange); red = target nodes the rendered mask keeps but the sphere filter removes.", extra)
    save("c2_ours_render_mask", img_r, fid, "Observed target nodes left after the rendered robot-surface mask.", extra)
    rp.close()


def region_scores(fm, fc, centered):
    cm = M.r6_choice(fm.geo, fm.cfg, fm.fin.maps["N"])
    Um, Uc = R.normalize(fm.fin.features["prefix_L8"]), R.normalize(fc.fin.features["prefix_L8"])
    if centered:
        Um, Uc = R.normalize(Um - Um.mean(0)), R.normalize(Uc - Uc.mean(0))
    mem = M.unit_rows(np.asarray(fm.geo.patch_px.tocsr()[cm["chosen"]].sum(0)).ravel(), Um)
    comp = RM.regions(fc.geo, ~fc.geo.support); ok = comp >= 0; px = fc.geo.patch_px.tocsr()
    out = []
    for r in np.unique(comp[ok]):
        k = comp == r
        if k.sum() < 20 or (fc.geo.pos[k].max(0) - fc.geo.pos[k].min(0)).max() > 0.4:
            continue
        cs = float(M.unit_rows(np.asarray(px[k].sum(0)).ravel(), Uc) @ mem)
        out.append((cs, k, fc.classes[int(fc.geo.eval_class_frac[fc.geo.node_eval][k].sum(0).argmax())]))
    return sorted(out, key=lambda t: -t[0])


NAMES = {"target": "target", "destination": "basket", "other_object": "other fruit"}


def project(rp, cam, p):
    """Pixel of world point p in a MuJoCo free camera (azimuth/elevation in degrees, fovy from the model)."""
    az, el = np.radians(cam["az"]), np.radians(cam["el"])
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = np.asarray(cam["lookat"]) - cam["dist"] * fwd
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    v = np.asarray(p) - eye
    x, y, z = v @ right, v @ up, v @ fwd
    H, W = rp.r.height, rp.r.width
    f = 0.5 * H / np.tan(np.radians(rp.m.vis.global_.fovy) / 2)
    return W / 2 + f * x / z, H / 2 - f * y / z


def label(img, rp, cam, items):
    from PIL import ImageDraw, ImageFont
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
    for p, t in items:
        u, v = project(rp, cam, p)
        x0, y0, x1, y1 = dr.textbbox((u, v), t, font=font, anchor="mm")
        dr.rectangle([x0 - 8, y0 - 6, x1 + 8, y1 + 6], fill=(255, 255, 255), outline=(40, 40, 40), width=2)
        dr.text((u, v), t, font=font, fill=(20, 20, 20), anchor="mm")
    return np.asarray(im)


def c3_identity(view, view_alert):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as mcm
    fid_m, fid = "T53_ep1822_f005", "T53_ep1822_f019"
    fm, fc = XF.Frame(RUN, fid_m, mask="render", feats=("prefix_L8",)), XF.Frame(RUN, fid, mask="render", feats=("prefix_L8",))
    ctr = fc.geo.pos[fc.gt].mean(0)
    rp = Replay(fid); cam = cam_at(ctr, *view)
    cmap = mcm.get_cmap("viridis")
    for centered, name in ((False, "c3_prior_raw_cosine"), (True, "c3_ours_centered_cosine")):
        sc = region_scores(fm, fc, centered)
        vals = np.array([v for v, _, _ in sc]); lo, hi = vals.min(), vals.max()
        layers, labels = [], []
        lo, hi = (-0.45, 0.65) if centered else (0.90, 1.00)
        for v, k, cl in sc:
            rgba = (*cmap(float(np.clip((v - lo) / (hi - lo), 0, 1)))[:3], 1.0)
            layers.append((fc.geo.pos[k], rgba, HALF))
            labels.append((fc.geo.pos[k].mean(0), f"{NAMES.get(cl, cl)} {v:.3f}"))
        img = label(rp.render(cam, layers), rp, cam, labels[:3])
        save(name, img, fid, ("Centered" if centered else "Raw") + " prefix-L8 cosine of each candidate region to the remembered region "
             f"({fid_m}); darker = more similar.", dict(memory_frame=fid_m, scores=[(round(v, 3), cl) for v, _, cl in sc[:5]], camera=cam))
    rp.close()
    # confidence alert: lowest chain score that is a wrong choice
    C = [json.loads(l) for l in open(RUN / "rounds/R11b/conf.jsonl")]
    C = [c for c in C if c.get("q_mem_chain:prefix_L8") is not None and c["target_voxels"] > 0 and (c["r6_iou"] or 0) < 0.1]
    c0 = min(C, key=lambda c: c["q_mem_chain:prefix_L8"])
    fr = XF.Frame(RUN, c0["frame_id"], mask="render")
    ch = M.r6_choice(fr.geo, fr.cfg, fr.fin.maps["N"])["chosen"]
    ctr = fr.geo.pos[ch].mean(0) * 0.5 + fr.geo.pos[fr.gt].mean(0) * 0.5
    rp = Replay(c0["frame_id"]); cam = cam_at(ctr, *view_alert)
    img = rp.render(cam, [(fr.geo.pos[fr.gt & ~ch], GHOST, HALF), (fr.geo.pos[ch], RED, HALF)])
    thr = json.loads((RUN / "LOCK_heldout.json").read_text())["arms"]["confidence_gate"]["abstain_if_below"]
    save("c3_ours_alert", img, c0["frame_id"], "Wrong region chosen (red); centered chain score below the locked threshold -> exception withheld.",
         dict(rule="lowest centered chain score among wrong choices (R11b)", score=c0["q_mem_chain:prefix_L8"], threshold=thr,
              chosen_iou=c0["r6_iou"], camera=cam))
    rp.close()


def probe(fid, out):
    fr = XF.Frame(RUN, fid, mask="render")
    ctr = fr.geo.pos[fr.gt].mean(0)
    rp = Replay(fid, 600, 450)
    tiles = []
    for el in (-25, -45):
        row = []
        for az in (90, 135, 180, 225, 270):
            row.append(rp.render(cam_at(ctr, 0.45, az, el), [(fr.geo.pos[fr.gt], ORANGE, HALF)]))
        tiles.append(np.concatenate(row, 1))
    Image.fromarray(np.concatenate(tiles, 0)).save(out)
    print("probe", out, "target centre", ctr)
    rp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", nargs=2, metavar=("FRAME", "PNG"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.probe:
        return probe(*a.probe)
    V = json.loads((pathlib.Path(__file__).with_name("ifg_overview_views.json")).read_text())
    jobs = dict(c1=lambda: c1_seeds(V["c1"]), c2=lambda: c2_extent(V["c2"], V["c2_rm"]), c3=lambda: c3_identity(V["c3"], V["c3_alert"]))
    for k, f in jobs.items():
        if not a.only or k in a.only:
            f()


if __name__ == "__main__":
    main()
