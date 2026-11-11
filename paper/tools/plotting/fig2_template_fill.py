"""Fig. 2 (method pipeline): fill the user's template (paper/figures/figure02_method_pipeline.png, AI-drawn mock scenes)
with recorded content of one SafeLIBERO episode. The template file itself is not modified.

Episode: SafeLIBERO spatial Level I, task 2, episode 9 (init 9, seed 7, wine bottle), AG3S final run pe123458.
Slots (template pixel boxes, 1920 x 816):
  rgb  input thumbnail        head RGB at obs 20 + left-wrist RGB inset (same instant)
  p1   attention-guided seg.  3D: recorded filtered cloud of call 2 (obs 30), target points (orange), seed points (gold)
  p2   object memory          3D: the target over time (call 2, call 6 target points; attached-sphere centres of held
                              calls 10, 14, 18, 22), dashed line through the recorded centroids
  p3   collision constraints  horizontal slice (z = 0.062 m) of the two recorded distance fields of call 6 (obs 50):
                              left = esdf_distance (all links, C1), right = esdf_free_distance (contact links, C2);
                              red = occupied (keep clear), orange = occupied in C1 but free in C2 (target surface)
  p5   action refinement      head RGB at obs 110 with hand collision spheres of call 18 (gold) and the executed
                              end-effector path obs 85 -> 139 (green), projected with the recorded camera
  s1-s5 episode strip         head RGB crops centred on the end effector at obs 35, 60, 95, 125, 139
Text overpainted because it does not match the recorded setup: "p40.5" -> pi0.5, "RGB + 2 wrists" -> "RGB-D +
wrist" (one wrist camera on the Panda), S4 caption "release" -> "lower onto plate", S5 "Retreat / move away" ->
"Success / episode ends" (the gripper never opens: SafeLIBERO ends the episode on success while the bowl is held).
Legend spheres for target / protected neighbour are recoloured to the paper's code (orange / blue).

    .venv-ag3s/bin/python paper/tools/plotting/fig2_template_fill.py [abstract] [scene]
Variant 'scene' draws slots p1-p3 on the head camera instead (fig02_method_pipeline_scene.*).
Suffix '-mem' (abstract-mem, scene-mem) replaces module 2 by the observed-vs-remembered panel (*_mem.*, 2026-11-09);
the original module-2 panel (target and attached spheres drawn alike) stays in the files without the suffix.
"""
import datetime as dt, glob, hashlib, json, pathlib, subprocess, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/segmentation"))
import gtto_feature_scene as fs  # noqa: E402  (read-only: load_calls, map_chunks, HAND_LINKS)

TEMPLATE = ROOT / "paper/figures/figure02_method_pipeline.png"
OUT = ROOT / "paper/figures/fig02_method"
RUN = pathlib.Path(glob.glob("/mnt/dev/AG3S_VLA-storage/results/benchmark/*qe-pe123458-e1a-spatial-LI-t2-*")[0])
STEM = "safelibero_spatial_t2_e9"
S = 2                                                     # canvas scale over the 1920 px template
WIDTH_IN = 6.875
ORANGE, BLUE, GOLD, RED, GREEN = "#E8711A", "#2E75B6", "#D9A406", "#D64545", "#2E9E4F"
FONT_R = "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Regular.otf"
FONT_B = "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Bold.otf"
SLOTS = {"rgb": (29, 244, 137, 327), "p1": (392, 144, 704, 294), "p2": (750, 144, 1025, 294), "p3": (1070, 144, 1350, 294),
         "p5": (1648, 144, 1892, 283), "s1": (43, 588, 385, 703), "s2": (413, 588, 748, 703), "s3": (770, 588, 1129, 703),
         "s4": (1152, 588, 1499, 703), "s5": (1522, 588, 1875, 703)}
STRIP_OBS = {"s1": 35, "s2": 60, "s3": 95, "s4": 125, "s5": 139}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------- recorded data
EP = np.load(RUN / "episodes" / f"{STEM}.npz")
GEOM = np.load(RUN / "episodes" / f"{STEM}_geom.npz", allow_pickle=True)
K = np.asarray(GEOM["K/head"][0]); T_BC = np.asarray(GEOM["T_base_cam/head"][0]); T_WB = np.asarray(GEOM["T_world_base"][0])
CALLS = fs.load_calls(RUN, STEM)
SEG, HOW = fs.map_chunks(RUN, STEM, CALLS)
OBS = [int(c["obs_index"]) for c in CALLS]


def chunk(obs):
    return np.load(SEG[OBS.index(obs)], allow_pickle=True)


def head(obs):
    return np.ascontiguousarray(EP["obs_agentview_image"][obs][::-1])


def wrist(obs):
    return np.ascontiguousarray(EP["obs_robot0_eye_in_hand_image"][obs][::-1])


def project(P):
    P = np.asarray(P, np.float64).reshape(-1, 3)
    Tc = np.linalg.inv(T_BC)
    pc = P @ Tc[:3, :3].T + Tc[:3, 3]
    return pc[:, :2] / np.maximum(pc[:, 2:3], 1e-6) * K[0, 0] + K[:2, 2], pc[:, 2]


def eef_base(obs):
    p = np.asarray(EP["obs_robot0_eef_pos"][obs], np.float64)
    Ti = np.linalg.inv(T_WB)
    return Ti[:3, :3] @ p + Ti[:3, 3]


def crop_fit(img, w, h, centre=None):
    """Crop ``img`` to aspect w/h (largest window, centred at ``centre`` in px if given) and resize to w x h."""
    H, W = img.shape[:2]; asp = w / h
    cw, ch = (W, int(W / asp)) if W / H < asp else (int(H * asp), H)
    cx, cy = centre if centre is not None else (W / 2, H / 2)
    x0 = int(np.clip(cx - cw / 2, 0, W - cw)); y0 = int(np.clip(cy - ch / 2, 0, H - ch))
    return Image.fromarray(img[y0:y0 + ch, x0:x0 + cw]).resize((w, h), Image.LANCZOS), [x0, y0, cw, ch]


def zoom(img, factor, centre, w, h):
    """Zoom crop: window of (W/factor) wide at ``centre``, aspect w/h."""
    H, W = img.shape[:2]; asp = w / h
    cw = int(W / factor); ch = int(cw / asp)
    if ch > H:
        ch = H; cw = int(H * asp)
    x0 = int(np.clip(centre[0] - cw / 2, 0, W - cw)); y0 = int(np.clip(centre[1] - ch / 2, 0, H - ch))
    return Image.fromarray(img[y0:y0 + ch, x0:x0 + cw]).resize((w, h), Image.LANCZOS), [x0, y0, cw, ch]


def mpl_canvas(w, h, bg="#F3F5F8"):
    fig = plt.figure(figsize=(w / 200, h / 200), dpi=200, facecolor=bg)
    return fig


def mpl_to_img(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(a)


# ------------------------------------------------------------------------------------------- slot renderers
def r_rgb(w, h):
    big = np.asarray(Image.fromarray(head(20)).resize((768, 768), Image.LANCZOS))
    img, cr = crop_fit(big, w, h, centre=(384, 400))
    wi = Image.fromarray(wrist(20)).resize((int(h * 0.55), int(h * 0.55)), Image.LANCZOS)
    pad = int(h * 0.05)
    d = ImageDraw.Draw(img)
    x0, y0 = w - wi.width - pad, h - wi.height - pad
    d.rectangle([x0 - 2, y0 - 2, x0 + wi.width + 1, y0 + wi.height + 1], fill="white")
    img.paste(wi, (x0, y0))
    return img, dict(obs=20, head_crop=cr, content="head RGB (obs 20) with the left-wrist RGB of the same instant as inset")


def scatter3d(ax, P, c, s, a=1.0):
    ax.scatter(*np.asarray(P).T, s=s, c=c, alpha=a, depthshade=False, linewidths=0, rasterized=True)


def setup3d(fig, lo, hi, elev=24, azim=8):
    ax = fig.add_axes([-0.42, -0.62, 1.84, 2.2], projection="3d")
    ax.set_axis_off(); ax.patch.set_alpha(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2]); ax.set_box_aspect(np.asarray(hi) - np.asarray(lo))
    ax.view_init(elev=elev, azim=azim)
    return ax


def inbox(P, lo, hi):
    return np.all((P >= lo) & (P <= hi), axis=1)


def r_p1(w, h):
    z = chunk(30)
    lo, hi = np.array([0.30, -0.30, -0.01]), np.array([0.62, 0.06, 0.24])
    C = np.asarray(z["filtered_cloud"], np.float64); sup = np.asarray(z["support_mask"], bool)
    T = np.asarray(z["target_points"], np.float64); seeds = np.asarray(z["attention_cloud"], np.float64)[np.asarray(z["seed_indices"])]
    fig = mpl_canvas(w, h); ax = setup3d(fig, lo, hi)
    m = inbox(C, lo, hi)
    d, _ = cKDTree(T).query(C, distance_upper_bound=0.006)
    scatter3d(ax, C[m & sup][::2], "#C9CED6", 0.8)
    scatter3d(ax, C[m & ~sup & ~np.isfinite(d)], BLUE, 2.0, 0.55)
    scatter3d(ax, T, ORANGE, 5.0, 0.95)
    scatter3d(ax, seeds, GOLD, 9.0, 1.0)
    return mpl_to_img(fig), dict(chunk=SEG[OBS.index(30)], call=OBS.index(30), obs=30, n_target=int(len(T)), n_seeds=int(len(seeds)),
                                 content="grey = support points, blue = other observed points, orange = target points, gold = seeds")


def r_p2(w, h):
    lo, hi = np.array([0.30, -0.24, -0.01]), np.array([0.78, 0.30, 0.30])
    z0 = chunk(30)
    C = np.asarray(z0["filtered_cloud"], np.float64); sup = np.asarray(z0["support_mask"], bool)
    fig = mpl_canvas(w, h); ax = setup3d(fig, lo, hi, elev=30, azim=10)
    m = inbox(C, lo, hi)
    scatter3d(ax, C[m & sup][::3], "#D3D7DE", 0.6)
    scatter3d(ax, C[m & ~sup], BLUE, 1.2, 0.25)
    cents, used = [], []
    for k, obs in enumerate((30, 50)):
        T = np.asarray(chunk(obs)["target_points"], np.float64)
        scatter3d(ax, T, ORANGE, 3.0, 0.35 + 0.3 * k); cents.append(T.mean(0)); used.append(obs)
    for k, obs in enumerate((70, 90, 110, 130)):
        A = np.asarray(chunk(obs)["attached_query_spheres"], np.float64).reshape(-1, 4)[:, :3]
        scatter3d(ax, A, ORANGE, 3.0, 0.30 + 0.17 * k); cents.append(A.mean(0)); used.append(obs)
    cents = np.asarray(cents)
    ax.plot(*cents.T, ls=(0, (3, 2)), color="#1F4E79", lw=1.2)
    ax.scatter(*cents[-1], marker=">", s=40, c="#1F4E79", depthshade=False)
    return mpl_to_img(fig), dict(obs=used, chunks=[SEG[OBS.index(o)] for o in used],
                                 content="orange = target points (obs 30, 50) and attached-sphere centres while held (obs 70-130), "
                                         "opacity increasing with time; dashed line through their centroids")


def r_p3(w, h):
    z = chunk(50)
    D = np.asarray(z["esdf_distance"], np.float32); F = np.asarray(z["esdf_free_distance"], np.float32)
    o, vs = np.asarray(z["esdf_origin"]), float(z["esdf_voxel_size"])
    zc = 0.062; k = int(round((zc - o[2]) / vs))
    xr, yr = (0.29, 0.65), (-0.33, 0.06)
    i0, i1 = int((xr[0] - o[0]) / vs), int((xr[1] - o[0]) / vs); j0, j1 = int((yr[0] - o[1]) / vs), int((yr[1] - o[1]) / vs)
    Ds, Fs = D[i0:i1, j0:j1, k], F[i0:i1, j0:j1, k]
    cmap = LinearSegmentedColormap.from_list("fld", ["#FFFFFF", "#EAF2FA", "#C7DCEF"])
    fig = mpl_canvas(w, h, bg="#FFFFFF")
    for n, (fld, title) in enumerate(((Ds, "C1"), (Fs, "C2"))):
        ax = fig.add_axes([0.01 + n * 0.5, 0.02, 0.48, 0.96]); ax.set_axis_off()
        ax.imshow(np.clip(fld, 0, 0.12), cmap=cmap, vmin=0, vmax=0.12, interpolation="bilinear")
        rgba = np.zeros(fld.shape + (4,))
        occ = fld <= 0.0025
        rgba[occ] = list(np.array([214, 69, 69]) / 255) + [0.9]
        if n == 1:
            freed = (Ds <= 0.0025) & ~occ
            rgba[freed] = list(np.array([232, 113, 26]) / 255) + [0.95]
        ax.imshow(rgba, interpolation="nearest")
        ax.contour(fld, levels=[0.02, 0.04, 0.06], colors=["#9DB9D6"], linewidths=0.4)
        ax.contour(fld, levels=[0.0025], colors=["#8E1F1F"], linewidths=0.6)
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.04, 0.96], color="#9AA3B2", lw=0.8))
    return mpl_to_img(fig), dict(chunk=SEG[OBS.index(50)], call=OBS.index(50), obs=50, slice_z_m=zc, x_range_m=xr, y_range_m=yr,
                                 content="left: esdf_distance (all links, C1); right: esdf_free_distance (contact links, C2); "
                                         "red = occupied (keep clear), orange = occupied in C1 but free in C2; light blue = "
                                         "distance, iso-lines at 2/4/6 cm; camera-aligned (right = +y, down = +x)")


def r_p5(w, h):
    obs = 110
    z = chunk(obs); Sj = json.loads(str(z["summary_json"]))
    names = list(Sj.get("sphere_link_names") or [])
    sc, sr = np.asarray(z["sphere_centres"], np.float64), np.asarray(z["sphere_radii"], np.float64)
    hand = np.array([n in fs.HAND_LINKS for n in names])
    U = 3
    img = Image.fromarray(head(obs)).resize((256 * U, 256 * U), Image.LANCZOS).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    uv, zz = project(sc[hand])
    gold = tuple(int(GOLD[i:i + 2], 16) for i in (1, 3, 5))
    for (u, v), d0, r in zip(uv, zz, sr[hand]):
        rp = K[0, 0] * r / d0 * U; x, y = (u + 0.5) * U, (v + 0.5) * U
        dr.ellipse([x - rp, y - rp, x + rp, y + rp], fill=gold + (55,), outline=gold + (200,), width=2)
    path = np.array([eef_base(t) for t in range(85, 140)])
    uv, _ = project(path)
    green = tuple(int(GREEN[i:i + 2], 16) for i in (1, 3, 5))
    pts = [((u + 0.5) * U, (v + 0.5) * U) for u, v in uv]
    dr.line(pts, fill=green + (230,), width=5)
    for x, y in pts[::6]:
        dr.ellipse([x - 6, y - 6, x + 6, y + 6], fill=green + (255,))
    x, y = pts[-1]; dr.ellipse([x - 10, y - 10, x + 10, y + 10], fill=green + (255,), outline=(255, 255, 255, 255), width=3)
    full = np.asarray(Image.alpha_composite(img, ov).convert("RGB"))
    c = (np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts]))
    out, cr = zoom(full, 1.0, c, w, h)
    return out, dict(obs=obs, chunk=SEG[OBS.index(obs)], path_obs=[85, 139], crop=cr,
                     content="gold = hand collision spheres (contact links) of call 18; green = executed end-effector path "
                             "(robosuite eef position, obs 85-139) projected with the recorded camera")


def r_strip(key, w, h):
    obs = STRIP_OBS[key]
    big = np.asarray(Image.fromarray(head(obs)).resize((768, 768), Image.LANCZOS))
    uv, _ = project(eef_base(obs)[None])
    c = (uv[0, 0] * 3 + 1.5, uv[0, 1] * 3 + 25)            # grip site, slightly lowered to include the bowl
    out, cr = zoom(big, 1.05, c, w, h)
    return out, dict(obs=obs, crop=cr, content="head RGB centred on the end effector")



# ------------------------------------------------------------------------------------------- variant B: scene overlays
# Same recorded arrays as p1-p3, drawn on the head camera instead of in 3D / as a field slice. Each depth pixel of the
# recorded head depth is lifted to the robot base frame and coloured by the recorded set it lies on (fig1_v4_assets.py).
import fig1_v4_assets as sc  # noqa: E402  (lift, near, in_spheres, up_mask, paint, sphere_mask; same camera model)

HEX = lambda c: tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))  # noqa: E731


def _obstacles(z, P, valid, exclude):
    """Observed non-support surface (>8 mm above the recorded table plane, within 15 mm of the recorded non-support cloud),
    outside the robot spheres (+12 mm) and outside ``exclude``."""
    cloud = np.asarray(z["filtered_cloud"]); sup = np.asarray(z["support_mask"], bool)
    plane = np.asarray(z["support_planes"])[0]
    above = (P.reshape(-1, 3) @ plane[:3] + plane[3]).reshape(256, 256) > 0.008
    Sp = np.c_[np.asarray(z["sphere_centres"]), np.asarray(z["sphere_radii"])]
    robot = sc.in_spheres(P, valid, Sp, pad=0.012)
    return sc.near(P, valid, cloud[~sup], 0.015) & above & ~robot & ~exclude


def r_p1_scene(w, h):
    obs = 30; z = chunk(obs); P, valid = sc.lift(obs)
    T = np.asarray(z["target_points"], np.float64)
    Sp = np.c_[np.asarray(z["sphere_centres"]), np.asarray(z["sphere_radii"])]
    tgt = sc.near(P, valid, T, 0.008) & ~sc.in_spheres(P, valid, Sp, pad=0.006)
    obst = _obstacles(z, P, valid, tgt)
    img = sc.big(sc.cam(EP["obs_agentview_image"][obs]))
    img = sc.paint(img, sc.up_mask(obst), HEX(BLUE), alpha=0.28, edge_w=2)
    img = sc.paint(img, sc.up_mask(tgt), HEX(ORANGE), alpha=0.62, edge_w=2)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    seeds_all = np.asarray(z["attention_cloud"], np.float64)[np.asarray(z["seed_indices"])]
    ds, _ = cKDTree(T).query(seeds_all)
    seeds = seeds_all[ds <= 0.02]            # display choice (user, 2026-11-09): seeds off the target surface not drawn
    uv, d0 = project(seeds)
    for (u, v), dd in zip(uv, d0):
        if dd > 0:
            x, y = (u + 0.5) * 3, (v + 0.5) * 3
            dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=HEX(GOLD) + (255,), outline=(80, 60, 0, 255), width=1)
    full = np.asarray(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))
    c = project(T.mean(0)[None])[0][0] * 3
    out, cr = zoom(full, 1.55, (c[0] + 60, c[1] + 10), w, h)
    return out, dict(obs=obs, chunk=SEG[OBS.index(obs)], crop=cr, n_seeds_recorded=int(len(seeds_all)),
                     n_seeds_drawn=int(len(seeds)), seeds_omitted_dist_m=np.sort(ds[ds > 0.02]).round(3).tolist(),
                     content="orange = depth pixels within 8 mm of the recorded target points; gold = seed points within "
                             "2 cm of the target points (the remaining recorded seeds, on the bottle and the cabinet, are not "
                             "drawn); light blue = other observed surface (kept as obstacle)")


def r_p2_scene(w, h):
    obs_bg = 130
    img = sc.big(sc.cam(EP["obs_agentview_image"][obs_bg])).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    o = HEX(ORANGE); cents = []
    steps = [(30, "target_points"), (50, "target_points"), (70, "attached"), (90, "attached"), (110, "attached"), (130, "attached")]
    for k, (obs, kind) in enumerate(steps):
        z = chunk(obs)
        Pk = (np.asarray(z["target_points"], np.float64) if kind == "target_points"
              else np.asarray(z["attached_query_spheres"], np.float64).reshape(-1, 4)[:, :3])
        if k == 0:                                                       # where the target was first grounded
            uv, d0 = project(Pk)
            for (u, v), dd in zip(uv, d0):
                if dd > 0:
                    x, y = (u + 0.5) * 3, (v + 0.5) * 3
                    dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=o + (150,))
        cents.append(project(Pk.mean(0)[None])[0][0] * 3 + 1.5)
    cents = np.asarray(cents)
    navy = (31, 78, 121, 255)
    for (p0, p1) in zip(cents[:-1], cents[1:]):                       # dashed polyline through the recorded centroids
        L = np.linalg.norm(p1 - p0); n = max(int(L // 14), 1)
        for t in range(n):
            if t % 2 == 0:
                a0, a1 = p0 + (p1 - p0) * t / n, p0 + (p1 - p0) * min((t + 1) / n, 1)
                dr.line([tuple(a0), tuple(a1)], fill=navy, width=5)
    for k, cxy in enumerate(cents):                                   # one marker per recorded step, opacity rising
        al = int(110 + 145 * k / (len(cents) - 1)); r = 9
        dr.ellipse([cxy[0] - r, cxy[1] - r, cxy[0] + r, cxy[1] + r], fill=o + (al,), outline=(255, 255, 255, 230), width=3)
    d = cents[-1] - cents[-2]; d = d / max(np.linalg.norm(d), 1e-6); nrm = np.array([-d[1], d[0]])
    tip = cents[-1]; dr.polygon([tuple(tip + 16 * d), tuple(tip - 8 * d + 10 * nrm), tuple(tip - 8 * d - 10 * nrm)], fill=navy)
    full = np.asarray(Image.alpha_composite(img, ov).convert("RGB"))
    cc = cents.mean(0)
    out, cr = zoom(full, 1.25, (cc[0], cc[1] + 20), w, h)
    return out, dict(background_obs=obs_bg, steps=[s[0] for s in steps], crop=cr,
                     content="background = head RGB at obs 130; orange dots = recorded target points of obs 30 (first "
                             "grounding); orange markers = projected centroids of the target points (obs 30, 50) and of the "
                             "attached spheres (obs 70-130), opacity rising with time; dashed line through them")


def r_p3_scene(w, h):
    obs = 50; z = chunk(obs); P, valid = sc.lift(obs)
    Sj = json.loads(str(z["summary_json"])); names = list(Sj.get("sphere_link_names") or [])
    Sp = np.c_[np.asarray(z["sphere_centres"]), np.asarray(z["sphere_radii"])]
    hand = np.array([n in fs.HAND_LINKS for n in names])
    o, vs = np.asarray(z["target_free_trace_origin"]), float(z["target_free_trace_voxel_size"])
    rem = o + (np.asarray(z["target_free_removed_ijk"], np.float64) + 0.5) * vs
    robot = sc.in_spheres(P, valid, Sp, pad=0.006)
    freed = sc.near(P, valid, rem, 0.5 * vs * 1.8) & ~robot
    obst = _obstacles(z, P, valid, freed)
    base = sc.big(sc.cam(EP["obs_agentview_image"][obs]))
    c1 = sc.paint(base, sc.up_mask(obst | freed), HEX(RED), alpha=0.45, edge_w=2)           # all links: everything is kept clear
    c2 = sc.paint(base, sc.up_mask(obst), HEX(RED), alpha=0.45, edge_w=2)                   # contact links: target removed
    c2 = sc.paint(c2, sc.up_mask(freed), HEX(ORANGE), alpha=0.65, edge_w=2)
    c2 = sc.paint(c2, sc.sphere_mask(Sp[hand]), HEX(GOLD), alpha=0.35, edge_w=3)
    ctr = project(rem.mean(0)[None])[0][0] * 3
    hw = w // 2 - 2 * S
    a1, cr = zoom(np.asarray(c1), 2.1, (ctr[0] + 45, ctr[1] - 10), hw, h)
    a2, _ = zoom(np.asarray(c2), 2.1, (ctr[0] + 45, ctr[1] - 10), hw, h)
    out = Image.new("RGB", (w, h), (255, 255, 255)); out.paste(a1, (0, 0)); out.paste(a2, (w - hw, 0))
    return out, dict(obs=obs, chunk=SEG[OBS.index(obs)], crop_each=cr,
                     content="left (C1, all links): red = observed non-support surface incl. the bowl; right (C2, contact "
                             "links): red = same minus the bowl, orange = depth pixels within 0.9 voxel of the voxels removed "
                             "from the target-free field, gold = hand collision spheres")

# ------------------------------------------------------------------------------------------- module 2, revised ("mem")
# What object memory recorded in this episode (summary_json.manipulated / target_exclusion.geometry_version):
#   calls 0-9 (obs 20-65)   state visible, association accepted (shift 0.35-1.26 mm): the re-grounded target is the
#                           same object; target_points are the current observation.
#   calls 10-23 (obs 70-135) grounding no_seed / no_cluster, state occluded (age 1-14); exclusion source "attached":
#                           the geometry observed at call 9 (observed_frame 9, 222 points) is kept and carried by the
#                           hand as attached_query_spheres (80 spheres, r 17.3 mm); it is not re-observed.
# The revised panel separates the two: observed = solid orange, remembered (not observed) = dashed orange outline.
MEM_SEEN, MEM_KEPT = (30, 50), (90, 130)


def _mem_masks(obs):
    z = chunk(obs)
    if "target_points" in z.files:                                   # observed target surface on the depth pixels
        P, valid = sc.lift(obs)
        Sp = np.c_[np.asarray(z["sphere_centres"]), np.asarray(z["sphere_radii"])]
        T = np.asarray(z["target_points"], np.float64)
        m = sc.up_mask(sc.near(P, valid, T, 0.008) & ~sc.in_spheres(P, valid, Sp, pad=0.006))
        return "seen", m, project(T.mean(0)[None])[0][0]
    A = np.asarray(z["attached_query_spheres"], np.float64).reshape(-1, 4)
    return "kept", sc.sphere_mask(A), project(A[:, :3].mean(0)[None])[0][0]


def r_p2_mem_scene(w, h):
    fig = mpl_canvas(w, h, bg="#FFFFFF")
    gx, gy, ww, hh = 0.012, 0.03, 0.488, 0.47
    asp = (w * ww) / (h * hh)
    meta = []
    for n, obs in enumerate(MEM_SEEN + MEM_KEPT):
        kind, m, c = _mem_masks(obs)
        r, col = divmod(n, 2)
        ax = fig.add_axes([gx + col * (ww + 0.0), 1 - (r + 1) * (hh + 0.02), ww - 0.012, hh]); ax.set_axis_off()
        img = np.asarray(sc.big(sc.cam(EP["obs_agentview_image"][obs])))
        cw = 330; ch = int(cw / asp)                                      # 768 px canvas window, same scale in all 4
        cx, cy = (c[0] + 0.5) * 3, (c[1] + 0.5) * 3 + 25
        x0 = int(np.clip(cx - cw / 2, 0, 768 - cw)); y0 = int(np.clip(cy - ch / 2, 0, 768 - ch))
        sub, ms = img[y0:y0 + ch, x0:x0 + cw], m[y0:y0 + ch, x0:x0 + cw]
        ax.imshow(sub, interpolation="lanczos")
        ov = np.zeros(ms.shape + (4,))
        if kind == "seen":
            ov[ms] = [0.91, 0.44, 0.10, 0.62]
            ax.imshow(ov, interpolation="bilinear")
            ax.contour(ms.astype(float), levels=[0.5], colors=[ORANGE], linewidths=1.0)
        else:
            ov[ms] = [0.91, 0.44, 0.10, 0.16]
            ax.imshow(ov, interpolation="bilinear")
            ax.contour(ms.astype(float), levels=[0.5], colors=[ORANGE], linewidths=1.1, linestyles=[(0, (2.2, 1.4))])
        ax.set_xlim(-0.5, cw - 0.5); ax.set_ylim(ch - 0.5, -0.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        meta.append(dict(obs=obs, call=OBS.index(obs), drawn=kind, window_768px=[x0, y0, cw, ch]))
    # row markers: solid chip = observed, dashed chip = remembered; arrow inside each row = time
    for r, kind in enumerate(("seen", "kept")):
        yc = 1 - (r + 0.5) * (hh + 0.02) - 0.01
        fig.patches.append(matplotlib.patches.FancyArrow(0.47, yc, 0.045, 0, width=0.012, head_width=0.05, head_length=0.02,
                                                         length_includes_head=True, transform=fig.transFigure,
                                                         color="#1F4E79", zorder=5))
        y_tag = 1 - r * (hh + 0.02) - 0.115
        box = matplotlib.patches.FancyBboxPatch((0.03, y_tag), 0.055, 0.075, boxstyle="round,pad=0.004,rounding_size=0.01",
                                                transform=fig.transFigure, zorder=6, mutation_aspect=w / h,
                                                facecolor=ORANGE if kind == "seen" else "#FFFFFF", edgecolor=ORANGE,
                                                linewidth=1.0, linestyle="-" if kind == "seen" else (0, (2, 1.3)))
        fig.patches.append(box)
    return mpl_to_img(fig), dict(frames=meta, remembered_geometry="attached_query_spheres (80, r 17.3 mm) = call-9 target "
                                 "geometry (observed_frame 9, 222 points) carried with the hand",
                                 content="top row (obs 30, 50): state visible, association accepted; solid orange = depth "
                                         "pixels within 8 mm of that call's target points. bottom row (obs 90, 130): state "
                                         "occluded, grounding no_seed / no_cluster; dashed orange = projected outline of the "
                                         "attached spheres (geometry last observed at obs 65), not an observation. Tags: "
                                         "filled = observed, dashed = remembered; arrows = time. Same 330 px window scale "
                                         "on the 768 px upsampled head frame in all four.")


def r_p2_mem(w, h):
    lo, hi = np.array([0.28, -0.26, -0.02]), np.array([0.84, 0.30, 0.44])     # wider than r_p2: held geometry rises to z 0.28 m
    z0 = chunk(30)
    C = np.asarray(z0["filtered_cloud"], np.float64); sup = np.asarray(z0["support_mask"], bool)
    fig = mpl_canvas(w, h); ax = setup3d(fig, lo, hi, elev=30, azim=10)
    ax.set_position([-0.40, -0.47, 1.80, 2.15])                       # lift the scene so the placed bowl is not cut
    m = inbox(C, lo, hi)
    scatter3d(ax, C[m & sup][::3], "#D3D7DE", 0.6)
    scatter3d(ax, C[m & ~sup], BLUE, 1.2, 0.25)
    for obs in MEM_SEEN:                                              # observed: solid
        scatter3d(ax, np.asarray(chunk(obs)["target_points"], np.float64), ORANGE, 3.0, 0.85)
    for obs in MEM_KEPT:                                              # remembered: hollow, dashed rim
        A = np.asarray(chunk(obs)["attached_query_spheres"], np.float64).reshape(-1, 4)[:, :3]
        ax.scatter(*A.T, s=7, facecolors="none", edgecolors=ORANGE, linewidths=0.45, alpha=0.9, depthshade=False, rasterized=True)
    path = np.array([eef_base(t) for t in range(MEM_SEEN[0], MEM_KEPT[-1] + 1)])
    ax.plot(*path.T, ls=(0, (3, 2)), color="#1F4E79", lw=1.0)
    ax.scatter(*path[-1], marker="o", s=14, c="#1F4E79", depthshade=False)
    return mpl_to_img(fig), dict(obs_seen=list(MEM_SEEN), obs_kept=list(MEM_KEPT),
                                 content="solid orange = observed target points (obs 30, 50; state visible); hollow orange = "
                                         "attached-sphere centres (obs 90, 130; state occluded: geometry last observed at obs "
                                         "65, carried with the hand, not re-observed); dashed navy = executed end-effector "
                                         "path obs 30-130")


# ------------------------------------------------------------------------------------------- template editing
def inpaint(a, box, pad=3, iters=500):
    x0, y0, x1, y1 = box
    a = a.astype(float)
    m = np.zeros(a.shape[:2], bool); m[y0 - pad:y1 + pad, x0 - pad:x1 + pad] = True
    a[m] = a[~m].mean(0)
    for _ in range(iters):
        p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
        nb = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        a[m] = nb[m]
    return np.clip(a, 0, 255).astype(np.uint8)


def round_mask(w, h, r):
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return m


def text(canvas, xy, s, size, bold=False, fill=(34, 39, 56), anchor="lm"):
    f = ImageFont.truetype(FONT_B if bold else FONT_R, size * S)
    ImageDraw.Draw(canvas).text((xy[0] * S, xy[1] * S), s, font=f, fill=fill, anchor=anchor)


# ------------------------------------------------------------------------------------------- module 4, revised ("mem")
# Schematic of the stage recognition in the manuscript (03_method.tex, HSR) and the code (ag3s/stages/subtask_probe.py):
# a linear probe on the frozen pi0.5's own hidden state (prefix K/V cache of PaliGemma = VLM backbone, layer 4: mean over
# text tokens ++ mean over image tokens, 512-d) gives pick / place / home; a second linear classifier adds the gripper
# feedback (measured opening -> grasp state z_t) and the hand-target distance (template arrow below the box).
# Diagram only: no per-frame probe output is recorded for this SafeLIBERO run (its calls record the grasp latch
# searching / latched / held from gripper feedback), so no stage is highlighted as "current".
PALM_PATH = [(978, 481, 1022, 507), (1015, 481, 1201, 523), (1196, 496, 1313, 508), (1301, 466, 1315, 507),
             (1309, 466, 1497, 477), (1482, 433, 1499, 477)]   # template px; background there is 254 +- 1 (measured)
M2_CAPTION_BOX = (786, 302, 990, 320)                  # template px of "track candidates over time"
M4_BOX = (1383, 150, 1612, 421)                         # interior of the template's module-4 card (border at x 1377/1617)
M4_BG = (248, 244, 254)
PURPLE, PURPLE_D, INK, ARROW = (150, 92, 196), (109, 56, 130), (34, 39, 56), (57, 90, 134)


def draw_m4(canvas):
    dr = ImageDraw.Draw(canvas)
    Q = lambda *v: [int(round(t * S)) for t in v]                                      # noqa: E731
    x0, y0, x1, y1 = M4_BOX
    dr.rectangle(Q(x0, y0, x1, y1), fill=M4_BG)

    def rr(b, fill, edge, w=1.2, r=6, dash=False):
        dr.rounded_rectangle(Q(*b), radius=r * S, fill=fill, outline=edge, width=max(1, int(w * S)))

    def arrow(p, q, col=ARROW, w=1.6):
        p, q = np.asarray(p, float), np.asarray(q, float)
        d = (q - p) / np.linalg.norm(q - p); n = np.array([-d[1], d[0]])
        dr.line(Q(*p, *(q - 6 * d)), fill=col, width=int(w * S))
        dr.polygon([tuple(Q(*(q))), tuple(Q(*(q - 7 * d + 4 * n))), tuple(Q(*(q - 7 * d - 4 * n)))], fill=col)

    def sub(xy, base, s, size, bold=False, fill=INK, anchor="lm"):
        """text with a subscript t-style suffix; anchor 'lm' or 'mm' on the whole string"""
        fb = ImageFont.truetype(FONT_B if bold else FONT_R, size * S)
        fs_ = ImageFont.truetype(FONT_B if bold else FONT_R, int(size * 0.68) * S)
        wb = dr.textlength(base, font=fb) / S; ws = dr.textlength(s, font=fs_) / S
        x = xy[0] - (wb + ws) / 2 if anchor == "mm" else xy[0]
        dr.text(Q(x, xy[1]), base, font=fb, fill=fill, anchor="lm")
        dr.text(Q(x + wb + 0.5, xy[1] + size * 0.28), s, font=fs_, fill=fill, anchor="lm")

    # (a) frozen pi0.5: VLM layers, layer 4 tapped
    text(canvas, (1391, 161), "Frozen VLA π", 14, bold=True, fill=PURPLE_D)
    fb = ImageFont.truetype(FONT_B, 14 * S); wpi = dr.textlength("Frozen VLA π", font=fb) / S
    text(canvas, (1391 + wpi + 0.5, 165), "0.5", 9, bold=True, fill=PURPLE_D)
    rr((1390, 172, 1478, 268), (255, 255, 255), (200, 186, 222), r=6)
    text(canvas, (1434, 181), "VLM backbone", 11, fill=(96, 90, 120), anchor="mm")
    bars = [(256, "L1", False), (237, "L4", True), (218, "⋮", None), (199, "L18", False)]
    for yb, lab, hi in bars:
        if hi is None:
            text(canvas, (1434, yb + 1), "· · ·", 12, fill=(150, 140, 170), anchor="mm"); continue
        rr((1398, yb - 7, 1470, yb + 7), PURPLE if hi else (234, 226, 246), PURPLE_D if hi else (205, 190, 228), w=1, r=3)
        text(canvas, (1434, yb), lab, 11, bold=hi, fill=(255, 255, 255) if hi else (96, 90, 120), anchor="mm")
    text(canvas, (1434, 276), "image + text tokens", 10, fill=(96, 90, 120), anchor="mm")
    # (b) hidden state read from layer 4
    arrow((1471, 237), (1497, 237), col=PURPLE_D)
    rr((1498, 206, 1608, 262), (255, 255, 255), PURPLE, w=1.4, r=6)
    text(canvas, (1553, 220), "hidden state", 14, bold=True, anchor="mm")
    sub((1553, 236), "f", "t", 14, bold=True, anchor="mm")
    text(canvas, (1553, 252), "prefix K/V, layer 4", 10, fill=(96, 90, 120), anchor="mm")
    # (c) gripper feedback
    rr((1390, 288, 1556, 322), (255, 255, 255), (186, 198, 218), w=1.2, r=6)
    g = (1404, 305)                                            # parallel-jaw icon
    dr.line(Q(g[0] - 9, g[1] - 9, g[0] + 9, g[1] - 9), fill=INK, width=2 * S)
    dr.line(Q(g[0], g[1] - 13, g[0], g[1] - 9), fill=INK, width=2 * S)
    for sx in (-1, 1):
        dr.line(Q(g[0] + 9 * sx, g[1] - 9, g[0] + 9 * sx, g[1] + 8), fill=INK, width=2 * S)
    dr.line(Q(g[0] - 5, g[1] + 2, g[0] + 5, g[1] + 2), fill=ARROW, width=S)
    for sx in (-1, 1):
        dr.polygon([tuple(Q(g[0] + 6 * sx, g[1] + 2)), tuple(Q(g[0] + 3 * sx, g[1] - 0.5)), tuple(Q(g[0] + 3 * sx, g[1] + 4.5))], fill=ARROW)
    text(canvas, (1420, 298), "gripper feedback", 14, bold=True)
    sub((1420, 313), "opening → grasp state z", "t", 11, fill=(80, 86, 104))
    # (d) probe + classifier -> pick / place / home
    arrow((1580, 263), (1580, 334), col=PURPLE_D)
    arrow((1460, 323), (1460, 334))
    rr((1408, 335, 1598, 357), (240, 232, 250), PURPLE, w=1.2, r=6)
    text(canvas, (1503, 346), "linear probe + classifier", 13, bold=True, fill=PURPLE_D, anchor="mm")
    arrow((1503, 358), (1503, 368), col=PURPLE_D)
    chips = [((1394, 369, 1460, 391), "pick", (254, 240, 197), (236, 196, 92)),
             ((1468, 369, 1534, 391), "place", (238, 217, 250), (190, 140, 230)),
             ((1542, 369, 1608, 391), "home", (254, 213, 209), (232, 150, 146))]
    for b, lab, fc, ec in chips:
        rr(b, fc, ec, w=1.2, r=6)
        text(canvas, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), lab, 14, anchor="mm")
    sub((1500, 406), "current stage s", "t", 15, anchor="mm")


def main(variant="abstract", mem=False):
    OUT.mkdir(parents=True, exist_ok=True)
    tpl = Image.open(TEMPLATE).convert("RGB")
    a = np.asarray(tpl)
    # 1) text that does not match the recorded setup (inpainted at template resolution, re-typeset after upscaling)
    edits = {"pi": (83, 120, 233, 141), "rgb": (154, 261, 218, 301), "s4cap": (1304, 709, 1355, 724),
             "s5head": (1686, 558, 1749, 580), "s5cap": (1662, 711, 1742, 728)}
    for b in edits.values():
        a = inpaint(a, b)
    # legend spheres: recolour target (teal) and neighbour (violet) spheres to the paper code
    leg = {}
    for name, win, col in (("target", (425, 760, 470, 800), ORANGE), ("neighbour", (658, 760, 705, 800), BLUE)):
        x0, y0, x1, y1 = win
        w = a[y0:y1, x0:x1].astype(int); sat = w.max(-1) - w.min(-1)
        ys, xs = np.where(sat > 40)
        bb = (x0 + xs.min(), y0 + ys.min(), x0 + xs.max(), y0 + ys.max()); leg[name] = bb
        a = inpaint(a, bb, pad=2)
    if mem:                                                   # "palm-target distance" chip + its path into module 4 removed
        a = a.copy()                                          # (user, 2026-11-10); flat fill with the measured background
        for x0, y0, x1, y1 in PALM_PATH:
            a[y0:y1 + 1, x0:x1 + 1] = 254
        a = inpaint(a, M2_CAPTION_BOX)                        # module-2 caption "track candidates over time" (user, 2026-11-10)
    canvas = Image.fromarray(a).resize((tpl.width * S, tpl.height * S), Image.LANCZOS)
    dr = ImageDraw.Draw(canvas)
    for name, col in (("target", ORANGE), ("neighbour", BLUE)):
        x0, y0, x1, y1 = [v * S for v in leg[name]]
        rgb = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
        dr.ellipse([x0, y0, x1, y1], fill=rgb, outline=tuple(int(c * 0.8) for c in rgb), width=2)
        cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2
        dr.ellipse([cx - r * 0.55, cy - r * 0.6, cx - r * 0.05, cy - r * 0.15], fill=tuple(min(255, int(c + 0.45 * (255 - c))) for c in rgb))
    text(canvas, (83, 131), "Frozen VLA π", 20, bold=True)
    fpi = ImageFont.truetype(FONT_B, 20 * S)
    wpi = ImageDraw.Draw(canvas).textlength("Frozen VLA π", font=fpi) / S
    text(canvas, (83 + wpi + 1, 136), "0.5", 13, bold=True)
    text(canvas, (154, 270), "RGB-D +", 21)
    text(canvas, (154, 294), "wrist", 21)
    text(canvas, (1329, 718), "lower onto plate", 20, anchor="mm")
    text(canvas, (1718, 569), "Success", 22, bold=True, anchor="mm")
    text(canvas, (1700, 718), "bowl on plate, episode ends", 20, anchor="mm")
    if mem:
        draw_m4(canvas)                                       # module 4 redrawn: hidden-state probe + gripper feedback
        text(canvas, (888, 311), "keep the target's identity over time", 21, anchor="mm")
    # 2) recorded content into the image slots
    rec = {}
    for key, (x0, y0, x1, y1) in SLOTS.items():
        w, h = (x1 - x0 + 1) * S, (y1 - y0 + 1) * S
        if key == "rgb":
            img, meta = r_rgb(w, h)
        elif key == "p1":
            img, meta = (r_p1_scene if variant == "scene" else r_p1)(w, h)
        elif key == "p2":
            if mem:
                img, meta = (r_p2_mem_scene if variant == "scene" else r_p2_mem)(w, h)
            else:
                img, meta = (r_p2_scene if variant == "scene" else r_p2)(w, h)
        elif key == "p3":
            img, meta = (r_p3_scene if variant == "scene" else r_p3)(w, h)
        elif key == "p5":
            img, meta = r_p5(w, h)
        else:
            img, meta = r_strip(key, w, h)
        img = img.convert("RGB").resize((w, h), Image.LANCZOS)
        canvas.paste(img, (x0 * S, y0 * S), round_mask(w, h, 9 * S))
        ImageDraw.Draw(canvas).rounded_rectangle([x0 * S, y0 * S, x1 * S + S - 1, y1 * S + S - 1], radius=9 * S,
                                                 outline=(200, 205, 213), width=S)
        rec[key] = dict(slot_px_template=[x0, y0, x1, y1], **meta)
        print("filled", key)
    stem = "fig02_method_pipeline" + ("_scene" if variant == "scene" else "") + ("_mem" if mem else "")
    png = OUT / f"{stem}.png"
    canvas.save(png, dpi=(canvas.width / WIDTH_IN,) * 2)
    pdf = OUT / f"{stem}.pdf"
    canvas.save(pdf, "PDF", resolution=canvas.width / WIDTH_IN)
    side = dict(artifact=stem, variant=variant, module2="revised: observed vs remembered" if mem else "original", png=str(png.relative_to(ROOT)), pdf=str(pdf.relative_to(ROOT)),
                size_px=list(canvas.size), size_in=[WIDTH_IN, round(WIDTH_IN * canvas.height / canvas.width, 3)],
                template=dict(file=str(TEMPLATE.relative_to(ROOT)), sha256=sha(TEMPLATE), size_px=[tpl.width, tpl.height],
                              note="AI-drawn layout supplied by the user (2026-11-09); its mock scenes are fully covered by the slots"),
                episode=dict(run=str(RUN), stem=STEM, chunk_mapping=HOW, seed=7, init=9),
                slots=rec, text_edits=dict(boxes=edits, legend_spheres=leg, removed_palm_target_path=PALM_PATH if mem else None,
                                           module2_caption=dict(box=M2_CAPTION_BOX, new="keep the target's identity over time") if mem else None,
                                           note="inpainted and re-typeset (Nimbus Sans Narrow) because the template text did "
                                                "not match the recorded setup; see module docstring"),
                font_warning="template text is raster at ~16-22 px of 1920 px; at 6.875 in this prints at ~4.1-5.7 pt "
                             "(below the 7 pt floor used for the other figures)",
                printed_numbers=[], script="paper/tools/plotting/fig2_template_fill.py",
                generated=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    (OUT / f"{stem}.json").write_text(json.dumps(side, indent=1, default=str) + "\n")
    print("wrote", png.relative_to(ROOT), canvas.size)


if __name__ == "__main__":
    for v in (sys.argv[1:] or ["abstract", "scene"]):        # suffix -mem: revised module 2 (separate output files)
        main(v.removesuffix("-mem"), mem=v.endswith("-mem"))
