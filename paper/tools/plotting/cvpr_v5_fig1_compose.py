"""CVPR 2027 Fig. 1 v5: instruction -> two lanes on the same SafeLIBERO episode.

  Prior lane : frozen VLA + an add-on open-vocabulary detector -> detector boxes -> (no target model) ->
               one end-effector ellipsoid -> bottle knocked over
  AG3S lane  : the same frozen VLA, its own noun attention -> 3D target surface -> contact-link distance field
               (bowl freed for the fingers, bottle still an obstacle) -> bowl placed, bottle upright

All rasters are recorded frames or overlays of recorded geometry (fig1_v4_assets.py, v4/assets/assets.json).
The 3D panel and the field panel are drawn directly from the recorded AG3S server chunk of call 6 (obs 50):
filtered point cloud, target-free trace voxels, hand collision spheres, and the two recorded distance fields
(esdf_distance = field for all links, esdf_free_distance = field read by the contact links).
No number is printed in the figure. Helpers are imported from cvpr_v3_fig1_compose.py.

    .venv-ag3s/bin/python paper/tools/plotting/cvpr_v5_fig1_compose.py
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import pathlib
import subprocess
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cvpr_v3_fig1_compose as v3  # noqa: E402
from cvpr_v3_fig1_compose import C, FS, PDF_DPI, PNG_DPI, REPO, STRICT_LAYOUT, TEXTWIDTH_IN, Fig, sha  # noqa: E402

sys.path.insert(0, str(REPO / "experiments/segmentation"))
import gtto_feature_scene as fs  # noqa: E402  (HAND_LINKS, read-only)

V5 = REPO / "paper/figures/fig01_teaser/v5"
A4 = REPO / "paper/figures/fig01_teaser/v4/assets"
META = json.loads((A4 / "assets.json").read_text())
AG3S_RUN = pathlib.Path(glob.glob("/mnt/dev/AG3S_VLA-storage/results/benchmark/*qe-pe123458-e1a-spatial-LI-t2-*")[0])
CHUNK = AG3S_RUN / "server_constraints/run_0000/chunk_00165.npz"        # call 6, obs 50 (closing), see assets.json
INSTR = "pick up the black bowl on the stove and place it on the plate"
INK, MUTE, LINE = "#1F2430", "#5A6070", "#C9CED8"
LANE_P, LANE_O = "#F6F6F7", "#FFF9EC"


class Used:
    def __init__(self):
        self.items = []

    def img(self, name, slot, centre, width, aspect):
        p = A4 / f"{name}.png"
        a = np.asarray(Image.open(p).convert("RGB"))
        H, W = a.shape[:2]
        w = int(round(min(width, W, H * aspect))); h = int(round(w / aspect))
        x0 = int(np.clip(centre[0] - w / 2, 0, W - w)); y0 = int(np.clip(centre[1] - h / 2, 0, H - h))
        self.items.append(dict(slot=slot, file=str(p.relative_to(REPO)), sha256=sha(p), crop_px=[x0, y0, w, h],
                               provenance=META.get(f"{name}.png", {})))
        return a[y0:y0 + h, x0:x0 + w]


def rbox(f, x, y, w, h, r, **kw):
    """Rounded rectangle in inches with true circular corners of radius r (figure-fraction coordinates)."""
    p = FancyBboxPatch((x / f.W, y / f.H), w / f.W, h / f.H, transform=f.fig.transFigure,
                       boxstyle=f"round,pad=0,rounding_size={r / f.W}", mutation_aspect=f.W / f.H, **kw)
    f.fig.add_artist(p)
    return p


def aclip(ax, w, h, r):
    """Rounded clip patch in axes coordinates for an axes of w x h inches."""
    p = FancyBboxPatch((0, 0), 1, 1, boxstyle=f"round,pad=0,rounding_size={r / w}", mutation_aspect=w / h,
                       transform=ax.transAxes, fc="none", ec="none")
    ax.add_patch(p)
    return p


def rpanel(f, rect, img, edge, badge=None, bcol=None, lw=0.9, r=0.06):
    """Image with rounded corners, a thin coloured frame and an optional pill badge (bottom left)."""
    x, y, w, h = rect
    f.panels.append(rect)
    ax = f.fig.add_axes(f.F(x, y, w, h), zorder=3)
    im = ax.imshow(img, aspect="auto", interpolation="lanczos")
    ax.set_axis_off()
    clip = aclip(ax, w, h, r); im.set_clip_path(clip)
    shadow(f, rect, r)
    rbox(f, x, y, w, h, r, fc="none", ec=edge, lw=lw, zorder=4)
    if badge:
        f.badges.append(ax.text(0.09 / w, 0.105 / h, badge, transform=ax.transAxes, fontsize=FS, fontweight="bold",
                                color="white", va="baseline", zorder=6,
                                bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.7", fc=bcol, ec="none", alpha=0.95)))
    return ax


def shadow(f, rect, r):
    x, y, w, h = rect
    rbox(f, x + 0.02, y - 0.02, w, h, r, fc="#000000", ec="none", alpha=0.09, zorder=2)


def chip(f, x, y, w, h, text, fc, ec, color=INK, ls="-", bold=True, icon=None):
    rbox(f, x, y, w, h, 0.05, fc=fc, ec=ec, lw=0.8, ls=ls, zorder=3)
    tx = x + w / 2 + (0.07 if icon else 0)
    f.text(tx, y + h / 2, text, size=FS, ha="center", fontweight="bold" if bold else "normal", color=color, zorder=6,
           linespacing=1.05)
    if icon == "eye":
        cx, cy = x + 0.12, y + h / 2
        f.fig.add_artist(Ellipse((cx / f.W, cy / f.H), 0.15 / f.W, 0.085 / f.H, transform=f.fig.transFigure, fc="white",
                                 ec=C["darkgold"], lw=0.9, zorder=6))
        f.fig.add_artist(Ellipse((cx / f.W, cy / f.H), 0.05 / f.W, 0.05 / f.H, transform=f.fig.transFigure, fc=C["darkgold"],
                                 ec="none", zorder=7))
    if icon == "lens":
        cx, cy = x + 0.115, y + h / 2 + 0.012
        f.fig.add_artist(Ellipse((cx / f.W, cy / f.H), 0.08 / f.W, 0.08 / f.H, transform=f.fig.transFigure, fc="white", ec=MUTE,
                                 lw=1.0, zorder=6))
        f.line([(cx + 0.028, cy - 0.028), (cx + 0.07, cy - 0.07)], MUTE, lw=1.4, z=6)


def person(f, x, y, s=1.0):
    """Minimal person icon (head + shoulders) with its feet at (x, y)."""
    col = "#3A4A6B"
    f.fig.add_artist(Ellipse((x / f.W, (y + 0.20 * s) / f.H), 0.10 * s / f.W, 0.10 * s / f.H, transform=f.fig.transFigure,
                             fc=col, ec="none", zorder=5))
    rbox(f, x - 0.08 * s, y, 0.16 * s, 0.13 * s, 0.06 * s, fc=col, ec="none", zorder=5)


def curve(f, a, b, color, rad=0.25, lw=1.0, ls="-"):
    f.segments.append(((a[0], a[1], b[0], b[1]), "arrow"))
    f.fig.add_artist(FancyArrowPatch((a[0] / f.W, a[1] / f.H), (b[0] / f.W, b[1] / f.H), transform=f.fig.transFigure,
                                     connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=7, color=color, lw=lw,
                                     ls=ls, zorder=5, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------------------------------- recorded geometry panels
def load_chunk():
    z = np.load(CHUNK, allow_pickle=True)
    S = json.loads(str(z["summary_json"]))
    return z, S


def panel_3d(f, rect, edge, badge, bcol):
    z, S = load_chunk()
    x, y, w, h = rect
    f.panels.append(rect)
    shadow(f, rect, 0.07)
    bg = f.fig.add_axes(f.F(x, y, w, h), zorder=3); bg.set_axis_off()
    cp = aclip(bg, w, h, 0.06); cp.set_facecolor("#FBFCFE")
    ax = f.fig.add_axes(f.F(x - 0.22, y - 0.16, w + 0.44, h + 0.36), projection="3d", zorder=4)
    ax.set_axis_off(); ax.patch.set_alpha(0)
    lo, hi = np.array([0.31, -0.27, -0.01]), np.array([0.63, 0.04, 0.26])
    cloud = np.asarray(z["filtered_cloud"], np.float64); sup = np.asarray(z["support_mask"], bool)
    o, vs = np.asarray(z["target_free_trace_origin"]), float(z["target_free_trace_voxel_size"])
    rem = o + (np.asarray(z["target_free_removed_ijk"], np.float64) + 0.5) * vs
    kep = o + (np.asarray(z["target_free_nontarget_kept_ijk"], np.float64) + 0.5) * vs
    inb = lambda P: np.all((P >= lo) & (P <= hi), axis=1)  # noqa: E731
    tbl = cloud[sup & inb(cloud)][::3]; obj = cloud[~sup & inb(cloud)]
    tree_rem = rem[inb(rem)]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(tree_rem).query(obj, distance_upper_bound=0.008)
    obj = obj[~np.isfinite(d)]
    names = list(S.get("sphere_link_names") or [])
    sc, sr = np.asarray(z["sphere_centres"], np.float64), np.asarray(z["sphere_radii"], np.float64)
    hand = np.array([n in fs.HAND_LINKS for n in names]) & inb(sc)
    kw = dict(depthshade=False, linewidths=0, rasterized=True)
    ax.scatter(*tbl.T, s=0.6, c="#D5D9E0", **kw)
    ax.scatter(*obj.T, s=1.6, c=C["neigh"], alpha=0.55, **kw)
    ax.scatter(*kep[inb(kep)].T, s=2.2, c=C["neigh"], alpha=0.95, **kw)
    ax.scatter(*tree_rem.T, s=2.2, c=C["target"], alpha=0.95, **kw)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=26, azim=8)               # from the camera side (+x) towards the robot: +y appears on the right
    rbox(f, x, y, w, h, 0.06, fc="none", ec=edge, lw=0.9, zorder=5)
    f.badges.append(f.text(x + 0.09, y + 0.105, badge, size=FS, fontweight="bold", color="white", va="baseline", zorder=8,
                           bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.7", fc=bcol, ec="none", alpha=0.95)))
    return dict(chunk=str(CHUNK), sha256=sha(CHUNK), box_lo_m=lo.tolist(), box_hi_m=hi.tolist(),
                n_table=int(len(tbl)), n_object=int(len(obj)), n_removed=int(len(tree_rem)), n_kept=int(inb(kep).sum()),
                content="grey = support points (every 3rd), blue = other observed points and non-target voxels kept, "
                        "orange = voxels removed from the contact-link field (target surface)")


def panel_field(f, rect, edge, badge, bcol):
    z, S = load_chunk()
    D = np.asarray(z["esdf_distance"], np.float32); F = np.asarray(z["esdf_free_distance"], np.float32)
    o, vs = np.asarray(z["esdf_origin"]), float(z["esdf_voxel_size"])
    zc = 0.062; k = int(round((zc - o[2]) / vs))
    xr, yr = (0.30, 0.64), (-0.31, 0.07)                 # base x (towards camera) and y (image right)
    i0, i1 = int((xr[0] - o[0]) / vs), int((xr[1] - o[0]) / vs); j0, j1 = int((yr[0] - o[1]) / vs), int((yr[1] - o[1]) / vs)
    Fs, Ds = F[i0:i1, j0:j1, k], D[i0:i1, j0:j1, k]       # rows = x increasing = down in the camera image
    x, y, w, h = rect
    ax = rpanel(f, rect, np.ones((2, 2, 3)), edge, badge=badge, bcol=bcol)
    ax.images[0].remove()
    cmap = LinearSegmentedColormap.from_list("fld", ["#0E3D66", "#2E75B6", "#9CC3E6", "#EAF3FB", "#FFFFFF"])
    ext = (0, 1, 1, 0)
    im = ax.imshow(np.clip(Fs, 0, 0.12), cmap=cmap, extent=ext, aspect="auto", interpolation="bilinear", vmin=0, vmax=0.12)
    clip = ax.patches[0]; im.set_clip_path(clip)
    occ_all, occ_free = Ds <= 0.0025, Fs <= 0.0025
    freed = occ_all & ~occ_free
    rgba = np.zeros(Fs.shape + (4,)); rgba[freed] = list(np.array([232, 113, 26]) / 255) + [0.95]
    im2 = ax.imshow(rgba, extent=ext, aspect="auto", interpolation="nearest"); im2.set_clip_path(clip)
    yy, xx = np.linspace(0, 1, Fs.shape[1]), np.linspace(0, 1, Fs.shape[0])
    cs = ax.contour(yy, xx, Fs, levels=[0.02, 0.04, 0.06, 0.08], colors="white", linewidths=0.35, alpha=0.8)
    for c in cs.collections if hasattr(cs, "collections") else [cs]:
        c.set_clip_path(clip)
    cs2 = ax.contour(yy, xx, Fs, levels=[0.0025], colors=["#0B2C4A"], linewidths=0.8)
    for c in cs2.collections if hasattr(cs2, "collections") else [cs2]:
        c.set_clip_path(clip)
    ax.set_xlim(0, 1); ax.set_ylim(1, 0)
    for (yy0, xx0, lab, du, dv) in ((-0.14, 0.40, "bowl", -0.27, -0.05), (-0.01, 0.56, "bottle", 0.0, 0.25)):
        u = (yy0 - yr[0]) / (yr[1] - yr[0]); v = 1 - (xx0 - xr[0]) / (xr[1] - xr[0])   # object centroids (base y, x)
        ax.text(u + du, v + dv, lab, transform=ax.transAxes, fontsize=FS, color=INK, ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.5", fc="white", ec="none", alpha=0.9))
    return dict(chunk=str(CHUNK), sha256=sha(CHUNK), slice_z_m=zc, slice_k=k, x_range_m=xr, y_range_m=yr,
                n_freed_px=int(freed.sum()), n_obstacle_px=int(occ_free.sum()),
                content="horizontal slice of the recorded fields at the bowl's mid-height, camera-aligned (right = +y, down = +x); "
                        "colour = esdf_free_distance (field read by the contact links), dark line = its zero level (obstacles, "
                        "incl. the bottle), orange = occupied in esdf_distance (all links) but free in esdf_free_distance "
                        "(the bowl surface the fingers may approach), white lines = iso-distance")


# ------------------------------------------------------------------------------------------- figure
def fig01_v5():
    W = TEXTWIDTH_IN
    M = 0.03
    COL0 = 1.00                                   # lane header + module chips
    GAP = 0.14                                    # space between panels (arrows)
    PW = (W - 2 * M - COL0 - 0.04 - 3 * GAP) / 4  # panel width
    PH_P, PH_O = 0.84, 1.06
    LAB = 0.19
    TOP = 0.34
    H = round(TOP + LAB + PH_P + 0.07 + 0.10 + LAB + PH_O + 0.07, 3)
    f = Fig(W, H)
    U = Used()
    side = {}
    xs = [M + COL0 + i * (PW + GAP) for i in range(4)]

    # ---- top band: person + instruction bubble + colour key
    px0, py0 = M + 0.10, H - 0.29
    person(f, px0, py0, s=1.0)
    by = H - 0.165
    f.text(0.33, by, f"\u201c{INSTR}\u201d", size=7.5, style="italic", color=INK, zorder=6,
           bbox=dict(boxstyle="round,pad=0.42,rounding_size=0.75", fc="white", ec="#9AA3B2", lw=0.8))
    f.fig.add_artist(Polygon([(0.33 / W, (by + 0.03) / H), (0.33 / W, (by - 0.05) / H), (0.24 / W, (by - 0.06) / H)],
                             closed=True, transform=f.fig.transFigure, fc="white", ec="none", zorder=7))
    f.line([(0.24, by - 0.06), (0.322, by - 0.05)], "#9AA3B2", lw=0.8, z=7)
    key = [(C["target"], "target surface"), (C["neigh"], "obstacle"), (C["gold"], "seeds")]
    offs = [0.0, 0.98, 1.66]
    kx = W - M - 2.12
    for (col, lab), dx in zip(key, offs):
        f.fig.add_artist(Ellipse(((kx + dx + 0.04) / W, by / H), 0.08 / W, 0.08 / H, transform=f.fig.transFigure, fc=col,
                                 ec="none", zorder=6))
        f.text(kx + dx + 0.12, by, lab, size=FS, color=MUTE)

    # ---- lanes
    yP = H - TOP - LAB - PH_P
    yO = yP - 0.07 - 0.10 - LAB - PH_O
    rbox(f, M, yP - 0.06, W - 2 * M, PH_P + LAB + 0.08, 0.08, fc=LANE_P, ec="#E3E5EA", lw=0.6, zorder=0)
    rbox(f, M, yO - 0.06, W - 2 * M, PH_O + LAB + 0.08, 0.08, fc=LANE_O, ec="#F0DDAE", lw=0.6, zorder=0)
    cx0, cw = M + 0.12, 0.80
    hP, hO = yP + PH_P + LAB - 0.08, yO + PH_O + LAB - 0.08
    f.text(cx0, hP, "Prior", size=7.5, fontweight="bold", color=C["red"])
    f.text(cx0, hO, "AG3S (ours)", size=7.5, fontweight="bold", color=C["green"])
    vlaP, vlaO = yP + PH_P - 0.34, yO + PH_O - 0.34
    chip(f, cx0, vlaP, cw, 0.22, "frozen VLA \u2744", "#EEF1F6", "#9AA3B2")
    chip(f, cx0, yP + 0.06, cw, 0.30, "+ external\ndetector", "#FFFFFF", "#9AA3B2", color=MUTE, ls=(0, (3, 2)), icon="lens")
    chip(f, cx0, vlaO, cw, 0.22, "frozen VLA \u2744", "#EEF1F6", "#9AA3B2")
    chip(f, cx0, yO + 0.22, cw, 0.32, "its own\nattention", "#FFF4D6", C["gold"], color=C["darkgold"], icon="eye")
    f.arrow((cx0 + cw / 2, vlaO), (cx0 + cw / 2, yO + 0.54), color=C["darkgold"], lw=0.9, ms=6)

    # person -> both VLA chips: one trunk on the left margin with two branches
    tx = M + 0.055
    f.line([(px0, py0), (px0, py0 - 0.02), (tx, py0 - 0.02), (tx, vlaO + 0.11)], "#7A8496", lw=0.9, z=5)
    f.arrow((tx, vlaP + 0.11), (cx0, vlaP + 0.11), color="#7A8496", lw=0.9, ms=6)
    f.arrow((tx, vlaO + 0.11), (cx0, vlaO + 0.11), color="#7A8496", lw=0.9, ms=6)

    # ---- prior lane
    asp = PW / PH_P
    rpanel(f, (xs[0], yP, PW, PH_P), U.img("l_boxes", "prior: detector boxes", (384, 440), 640, asp), C["red"],
           badge="\u2717 bottle unboxed", bcol=C["red"])
    rbox(f, xs[1], yP, PW, PH_P, 0.06, fc="#FFFFFF", ec="#B9BEC8", lw=0.8, ls=(0, (3, 2)), zorder=2)
    f.text(xs[1] + PW / 2, yP + PH_P / 2 + 0.07, "target surface", size=FS, ha="center", color=MUTE, fontweight="bold")
    f.text(xs[1] + PW / 2, yP + PH_P / 2 - 0.07, "not modelled", size=FS, ha="center", color=MUTE)
    rpanel(f, (xs[2], yP, PW, PH_P), U.img("l_ee", "prior: end-effector ellipsoid", (560, 390), 520, asp), C["red"],
           badge="\u2717 bowl not modelled", bcol=C["red"])
    rpanel(f, (xs[3], yP, PW, PH_P), U.img("o_aegis_final", "prior: episode end", (430, 420), 640, asp), C["red"],
           badge="\u2717 bottle toppled", bcol=C["red"])
    labsP = ["detector boxes", "target", "robot: one hand ellipsoid", "outcome"]
    labsO = ["noun attention \u2192 seeds", "3D target surface", "contact-link field", "outcome"]
    for i in range(4):
        f.text(xs[i] + 0.02, yP + PH_P + 0.10, labsP[i], size=FS, color=MUTE, fontweight="bold")
        f.text(xs[i] + 0.02, yO + PH_O + 0.10, labsO[i], size=FS, color=C["darkgold"], fontweight="bold")
    f.arrow((cx0 + cw, yP + 0.21), (xs[0] - 0.03, yP + 0.21), color="#9AA3B2", lw=0.9, ms=6, ls=(0, (3, 2)))
    for i in range(3):
        f.arrow((xs[i] + PW + 0.03, yP + PH_P / 2), (xs[i + 1] - 0.03, yP + PH_P / 2), color="#9AA3B2", lw=0.9, ms=6)

    # ---- AG3S lane
    asp = PW / PH_O
    rpanel(f, (xs[0], yO, PW, PH_O), U.img("r_identity", "ours: noun attention + seeds", (300, 330), 430, asp), C["gold"],
           badge="\u2713 seeds on bowl", bcol=C["green"])
    side["panel_3d"] = panel_3d(f, (xs[1], yO, PW, PH_O), C["gold"], "\u2713 surface, not box", C["green"])
    side["panel_field"] = panel_field(f, (xs[2], yO, PW, PH_O), C["gold"], "\u2713 only bowl freed", C["green"])
    rpanel(f, (xs[3], yO, PW, PH_O), U.img("o_ag3s_final", "ours: episode end", (430, 420), 600, asp), C["gold"],
           badge="\u2713 bottle upright", bcol=C["green"])
    f.arrow((cx0 + cw, yO + 0.38), (xs[0] - 0.03, yO + 0.38), color=C["darkgold"], lw=1.0, ms=6)
    for i in range(3):
        f.arrow((xs[i] + PW + 0.03, yO + PH_O / 2), (xs[i + 1] - 0.03, yO + PH_O / 2), color=C["darkgold"], lw=1.1, ms=7)
    save(f, U, "fig01_v5", side)


def save(f, U, stem, extra):
    mn, mx, cnt = f.audit()
    assert mn >= FS - 1e-6, f"{stem}: font below {FS} pt: {mn}"
    viol = f.layout_audit()
    n_viol = sum(len(v) for v in viol.values())
    print(f"[{stem}] {f.W:.3f} x {f.H:.3f} in, layout audit: " + ", ".join(f"{k}={len(v)}" for k, v in viol.items()))
    for k, v in viol.items():
        for item in v[:40]:
            print("   ", k, item)
    texts = [t.get_text() for t in f.fig.findobj(v3.mtext.Text) if t.get_text().strip() and t.get_visible()]
    if STRICT_LAYOUT:
        assert n_viol == 0, f"{stem}: {n_viol} layout violations"
    V5.mkdir(parents=True, exist_ok=True)
    pdf = V5 / f"{stem}.pdf"
    f.fig.savefig(pdf, dpi=PDF_DPI)
    f.fig.savefig(pdf.with_suffix(".png"), dpi=PNG_DPI)
    v3.plt.close(f.fig)
    side = dict(
        artifact=stem, pdf=str(pdf.relative_to(REPO)), png=str(pdf.with_suffix(".png").relative_to(REPO)),
        size_in=[f.W, f.H], textwidth_in=TEXTWIDTH_IN, raster_dpi_in_pdf=PDF_DPI, png_dpi=PNG_DPI,
        fonts=dict(family="DejaVu Sans", pdf_fonttype=42, min_pt=mn, max_pt=mx, n_text_items=cnt, floor_pt=FS),
        printed_text=texts, printed_numbers=[],
        layout_audit=dict(tolerance_pt=1.0, min_text_to_panel_border_in=0.05, violations={k: len(v) for k, v in viol.items()},
                          n_panels=len(f.panels), n_texts=cnt),
        image_panels=U.items, episode=META.get("_episode"),
        honesty=["all panels come from SafeLIBERO spatial Level I task 2 episode 9 (init 9, seed 7); prior lane = AEGIS run "
                 "(boxes, episode end) and KNOWS run (end-effector ellipsoid); AG3S lane = final run pe123458",
                 "the 3D and field panels plot recorded arrays of one AG3S server chunk (call 6, obs 50); no re-computation",
                 "the 'not modelled' tile is a statement about the method design (AEGIS models obstacles, not a target surface)",
                 "person icon, chips and arrows are schematic", "no number printed in the figure"],
        script="paper/tools/plotting/cvpr_v5_fig1_compose.py", generated=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
        is_measured_result=False, scope="qualitative same-episode comparison; no numbers",
    )
    side.update(extra)
    pdf.with_suffix(".json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {pdf.relative_to(REPO)}  min font {mn} pt")


if __name__ == "__main__":
    fig01_v5()
