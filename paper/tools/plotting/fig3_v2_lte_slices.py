"""Figure 3 v2 — LTE panels: the scene field d_t and the target-free field d_t^{\\X} on one vertical slice. CPU only, read only.

  cd /mnt/dev/AG3S_VLA-development; MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python \
      paper/tools/plotting/fig3_v2_lte_slices.py extract --run <RUN> --ep 119 --state closing --out <npz>
  ... fig3_v2_lte_slices.py plot --npz <npz> --outdir paper/figures/paperbanana/figure3/assets_v2

`extract` reuses the IFG-AUDIT-LTE harness (`ifg_audit_lte.process`, `fixed_input_ablation.build_field`) on one recorded server
chunk: field values are the recorded cuRobo main layer (coarse -> fine min) and target-free layer, GT class labels come from the
run's MuJoCo model at the planning-row qpos. The slice is the vertical plane through the target and nearest-neighbour centres
(same plane as AG3S/ag3s/docs/figures/ifg-audit/ifg-audit-lte-section.png). `plot` writes text-free panels for the figure tool.
"""
import argparse, glob, json, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def extract(a):
    from AG3S.ag3s.experiments.studies import ifg_audit_lte as L
    from AG3S.trajopt.experiments.fixed_input_ablation import build_field
    import mujoco
    K = a.run / "verify" / a.kit
    scene = L.Scene(a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml")
    jf = next(f for f in sorted(glob.glob(str(K / "replay" / f"runs_{a.arm}_{a.arm}_ep{a.ep}_*.json"))) if not f.endswith("npz"))
    js = json.load(open(jf)); b = js["server_chunk_alignment"]["block"]
    rd = pathlib.Path(js["run_dir"]); man = json.load(open(rd / "manifest.json"))["scene"]
    pl = [r for r in (json.loads(x) for x in open(rd / "frames.jsonl")) if r.get("kind") == "planning"]
    ks = []
    for k in range(b["block_len"]):
        z = np.load(pathlib.Path(b["chunk_dir"]) / f"chunk_{k + b['offset']:05d}.npz", allow_pickle=True)
        if str((json.loads(str(z["summary_json"])).get("grasp") or {}).get("state")) == a.state:
            ks.append(k)
    k = a.request if a.request is not None else ks[min(a.nth, len(ks) - 1)]
    cp = pathlib.Path(b["chunk_dir"]) / f"chunk_{k + b['offset']:05d}.npz"
    z = np.load(cp, allow_pickle=True)
    sl = {}
    L.process(scene, z, json.loads(str(z["summary_json"])), pl[k], a.ep, k, js, man, L.fib_dirs(L.N_DIR), {}, build_field, want_slice=sl)
    ct, cn = sl["ct"], sl["cn"]
    e1 = cn - ct; e1[2] = 0; dist = float(np.linalg.norm(e1)); e1 /= dist
    e2 = np.array([0, 0, 1.0]); nrm = np.cross(e1, e2)
    us = np.arange(-a.pad, dist + a.pad, a.px); zs = np.arange(-a.below, a.above, a.px)
    U, Z = np.meshgrid(us, zs)
    G = ct[None] + U.ravel()[:, None] * e1 + Z.ravel()[:, None] * e2
    D, _ = scene.query(G, force=sl["target_g"] + sl["neigh_g"])
    cls = sl["cls"]
    gt_cls = np.full(len(G), -1)
    best = D.min(1)
    for ci in sorted(set(cls.tolist())):
        Dc = np.where((cls == ci)[None], D, np.inf).min(1)
        gt_cls[(Dc <= best + 1e-9)] = ci
    fm = sl["field"].distance(G).reshape(U.shape)
    ff = sl["field"].target_free_distance(G).reshape(U.shape)
    Dt = np.where((cls == 0)[None], D, np.inf).min(1).reshape(U.shape)
    Dn = np.where((cls == 1)[None], D, np.inf).min(1).reshape(U.shape)
    sph_h = (sl["C"] - ct) @ nrm
    cut = np.abs(sph_h) < sl["radii"]
    circ = [[float((sl["C"][i] - ct) @ e1), float((sl["C"][i] - ct) @ e2), float(np.sqrt(sl["radii"][i] ** 2 - sph_h[i] ** 2)), bool(sl["auth"][i])]
            for i in np.flatnonzero(cut)]
    # scene render with the slice plane, from the head camera
    rnd = mujoco.Renderer(scene.m, 600, 800)
    rnd.update_scene(scene.d, camera="zed_left"); img = rnd.render(); rnd.close()
    np.savez_compressed(a.out, us=us, zs=zs, main=fm, free=ff, gt_cls=gt_cls.reshape(U.shape), Dt=Dt, Dn=Dn, circles=np.array(circ),
                        img=img, ct=ct, cn=cn, e1=e1,
                        meta=json.dumps({"run": str(a.run), "kit": a.kit, "arm": a.arm, "episode": a.ep, "request": k, "state": str((json.loads(str(z["summary_json"])).get("grasp") or {}).get("state")),
                                         "chunk": str(cp), "classes": list(L.CLASSES), "target_neighbour_centre_dist_m": dist,
                                         "pixel_m": a.px}))
    print("wrote", a.out, "request", k, "of", ks)


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    q = np.load(a.npz, allow_pickle=True); meta = json.loads(str(q["meta"]))
    us, zs = q["us"] * 1000, q["zs"] * 1000
    ext = [us[0], us[-1], max(zs[0], a.zmin) if a.zmin is not None else zs[0], zs[-1]]
    ORANGE, BLUE, GOLD, GREY, TABLE = "#E8711A", "#2E75B6", "#C99A06", "#7F7F7F", "#B9A27A"
    glow = LinearSegmentedColormap.from_list("glow", ["#5B8FCB", "#A9C8EA", "#DCE9F7", "#FFFFFF"])
    out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    side = {}
    zs_support = float(np.repeat(zs[:, None], q["Dt"].shape[1], 1)[q["Dt"] < 0].min())   # mm, lowest GT target point = support top
    for name, F, kept_target in (("main", q["main"], True), ("free", q["free"], False)):
        F = F * 1000
        fig = plt.figure(figsize=(4, 4 * (ext[3] - ext[2]) / (ext[1] - ext[0])), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
        ax.imshow(np.clip(F, 0, a.vmax), origin="lower", extent=ext, cmap=glow, vmin=0, vmax=a.vmax, interpolation="bilinear")
        occ = F <= 0
        rgba = np.zeros(F.shape + (4,))
        col = {0: ORANGE, 1: BLUE, 3: TABLE}
        g = q["gt_cls"].copy()
        if a.fill == "support":   # nearest class, but the band just above the support top (target bottom) is the support plane
            zt = np.repeat(zs[:, None], g.shape[1], 1)
            g[(zt <= zs_support + a.band) & np.isin(g, (0, 1))] = 3
        if a.fill == "inside":   # a fruit colour only inside that fruit's GT surface; elsewhere the nearest non-fruit class
            g[(g == 0) & ~(q["Dt"] < 0)] = 3
            g[(g == 1) & ~(q["Dn"] < 0)] = 3
        for ci, c in col.items():
            m = occ & (g == ci)
            rgba[m] = matplotlib.colors.to_rgba(c)
        other = occ & ~np.isin(g, list(col))
        rgba[other] = matplotlib.colors.to_rgba("#4D4D4D")
        ax.imshow(rgba, origin="lower", extent=ext, interpolation="nearest")
        U, Z = np.meshgrid(us, zs)
        ax.contour(U, Z, F, levels=[0], colors=["#1F1F1F"], linewidths=0.8)
        if not kept_target:
            ax.contour(U, Z, q["Dt"] * 1000, levels=[0], colors=[ORANGE], linewidths=1.6, linestyles=[(0, (3, 2))])
        for u, zz, r, auth in q["circles"]:
            ax.add_patch(plt.Circle((u * 1000, zz * 1000), r * 1000, fill=False, lw=1.4 if auth else 1.0, color=GOLD if auth else GREY))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        p = out / f"f3v2_lte_field_{name}.png"
        fig.savefig(p, dpi=300); plt.close(fig)
        side[name] = str(p)
    # head-camera render with the slice location
    img = q["img"]
    fig = plt.figure(figsize=(4, 3), dpi=300); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.imshow(img)
    p = out / "f3v2_lte_slice_location.png"; fig.savefig(p, dpi=300); plt.close(fig)
    m = dict(meta)
    m.update(assets=side | {"location": str(p)}, script="paper/tools/plotting/fig3_v2_lte_slices.py", vmax_mm=a.vmax, fill=a.fill, zmin_mm=a.zmin, support_band_mm=a.band, support_top_mm=zs_support,
             encoding={"background": "field distance 0..vmax mm, blue = near a surface", "fill": "field <= 0, coloured by the nearest GT geom (fill=nearest) or, with fill=inside, orange / blue only "
                       "inside the target / neighbour GT surface and the nearest non-fruit class elsewhere; with fill=support, nearest GT geom except that "
                       "fruit-coloured pixels within support_band_mm above the support top (lowest GT target point) are drawn as the support (tan table, dark grey other)", "black line": "field = 0",
                       "dashed orange (free only)": "GT target outline; the target-free field no longer treats it as an obstacle",
                       "gold circles": "query spheres of contact links (target_field_exclude = True) cut by the plane",
                       "grey circles": "other query spheres cut by the plane"},
             is_measured_result=False, note="Recorded field values from one closed-loop request; illustration, no metric printed.")
    (out / "f3v2_lte_field.json").write_text(json.dumps(m, indent=1))
    print("wrote", side, p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--run", type=pathlib.Path, required=True); e.add_argument("--kit", default="IFG-APPLE-SPLIT"); e.add_argument("--arm", default="SPL")
    e.add_argument("--ep", type=int, default=119); e.add_argument("--state", default="closing"); e.add_argument("--nth", type=int, default=0); e.add_argument("--request", type=int, default=None)
    e.add_argument("--pad", type=float, default=0.07); e.add_argument("--below", type=float, default=0.04); e.add_argument("--above", type=float, default=0.12)
    e.add_argument("--px", type=float, default=0.001); e.add_argument("--out", type=pathlib.Path, required=True)
    p = sp.add_parser("plot")
    p.add_argument("--npz", type=pathlib.Path, required=True); p.add_argument("--outdir", required=True); p.add_argument("--vmax", type=float, default=40.0)
    p.add_argument("--fill", choices=("nearest", "inside", "support"), default="nearest"); p.add_argument("--band", type=float, default=6.0, help="mm"); p.add_argument("--zmin", type=float, default=None, help="mm")
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else plot(a)
