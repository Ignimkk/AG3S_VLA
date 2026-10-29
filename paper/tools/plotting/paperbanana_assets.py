"""Generate the PaperBanana reference assets for F1-F3 from one recorded AG3S closed-loop run.

Source: T53 serving run C1a (AG3S on, current serving configuration of 2026-10-13), test episode
1842 ("Put the pear in the basket.", right arm, neighbour orange next to the target pear).
Every panel is drawn from recorded arrays (client step records, server ``--record-constraints``
chunk dumps, GT replay labels, the third-person video). Nothing is simulated or re-run here.
The ownership panel calls the serving function ``_ball_owned_split`` on recorded points.

Inputs are read-only. Output: paper/figures/paperbanana/figure{1,2,3}/assets/*.png, one JSON
sidecar per PNG, and a reference_montage.png per figure.

Usage:
    .venv-ag3s/bin/python paper/tools/plotting/paperbanana_assets.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "AG3S"))
sys.path.insert(0, str(REPO / "AG3S/ag3s/experiments/studies"))
sys.path.insert(0, str(REPO / "AG3S/ag3s/experiments"))
from ag3s.fields.curobo_builder import OwnedEvidence, TargetBall, _ball_owned_split  # noqa: E402
import t53_records  # noqa: E402

T53 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/"
                   "20261013T100210+0900_4a540f1a18d7_cb037c425d2f/verify/T53")
KEY = "ep1842"
RUN = T53 / f"runs/C1a/R/C1a_{KEY}_s18421"
OUT = REPO / "paper/figures/paperbanana"

# chunk indices (== client step index) chosen from the recorded grasp timeline of this run
CH_APPROACH = 12   # latched, pre-grasp, target visible
CH_GROUND = 17     # last pre-grasp chunk: target-free layer built with ball_owned
CH_CLOSE = 18      # latch CLOSING; GT fingers touch the pear
CH_HELD = 19       # latch HELD + attach (attached sphere is stored at its attach-time base-frame pose)
# third-person video frame (= control step) for the keyframe strip
KEYFRAMES = {"approach": 84, "grasp": 160, "carry": 200, "release": 268}
VIDEO_CROP = (40, 32, 600, 362)        # x0, y0, x1, y1 in 640x480 (drops the debug HUD)
NEIGHBOUR_RADIUS_M = 0.045             # colour-only: cloud points this close to a GT fruit centre

# one palette for all three figures
C_TARGET = "#E8711A"
C_NEIGH = "#2E75B6"
C_SCENE = "#BDBDBD"
C_CONTACT = "#C99A06"
C_LINK = "#7F7F7F"
C_ATTACH = "#7030A0"
C_HOLD = "#C00000"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "savefig.dpi": 220, "axes.spines.top": False,
                     "axes.spines.right": False})


# ---------------------------------------------------------------- loading
def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load():
    steps = sorted((RUN / "ag3s/run_0000").glob("step_*.npz"))
    acts = [np.load(f)["actions"] for f in steps]
    segs = t53_records.segments(sorted(T53.glob("runs/C1a/server_*/server_constraints/run_*")))
    i = t53_records.pair_segment(segs, acts)
    if i is None:
        raise SystemExit("no server segment pairs with the client planning rows")
    return steps, list(segs[i]), np.load(T53 / f"gt/{KEY}.npz"), np.load(T53 / f"gt/{KEY}.grasp.npz")


def chunk(chunks, j):
    z = np.load(chunks[j], allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["summary"] = json.loads(str(d.pop("summary_json")))
    d["path"] = chunks[j]
    return d


SOURCES: dict[str, dict] = {}


def save(fig, fig_no: int, name: str, sources: list[pathlib.Path], note: str, extra: dict | None = None):
    path = OUT / f"figure{fig_no}/assets/{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    meta = {"asset": str(path.relative_to(REPO)), "run": "T53 C1a (AG3S serving, 2026-10-13)",
            "episode": KEY, "script": "paper/tools/plotting/paperbanana_assets.py",
            "sources": {str(s): sha(s) for s in sources}, "note": note,
            "is_measured_result": False, "generated": True, **(extra or {})}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    SOURCES[f"figure{fig_no}/{name}"] = meta
    print("wrote", path.relative_to(REPO))
    return path


# ---------------------------------------------------------------- helpers
def video_frame(idx: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        f = pathlib.Path(tmp) / "f.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(RUN / "third_person_front.mp4"),
                        "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1", str(f)], check=True)
        return np.asarray(Image.open(f).convert("RGB").crop(VIDEO_CROP))


def overlay(rgb: np.ndarray, att: np.ndarray, ax, title: str):
    a = np.asarray(att, np.float64)
    a = (a - a.min()) / max(a.max() - a.min(), 1e-12)
    up = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(rgb.shape[1::-1], Image.BILINEAR)) / 255.0
    ax.imshow(rgb)
    ax.imshow(up, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def neighbour_mask(cloud: np.ndarray, gt, j: int) -> tuple[np.ndarray, str]:
    """Colour-only label: points near the GT centre of the non-target fruit closest to the target."""
    fruits = [str(f) for f in gt["fruits"]]
    pos = gt["fruit_pos_base"][j]
    t = fruits.index(str(gt["target"]))
    d = [np.linalg.norm(pos[k] - pos[t]) if k != t else np.inf for k in range(len(fruits))]
    k = int(np.argmin(d))
    return np.linalg.norm(cloud - pos[k][None], axis=1) <= NEIGHBOUR_RADIUS_M, fruits[k]


def link_masks(c):
    names = np.array(c["summary"]["sphere_link_names"])
    excl = np.array(c["summary"]["target_permission"]["target_field_exclude"], bool)
    return names, excl


def window(points: np.ndarray, centre: np.ndarray, half: float) -> np.ndarray:
    return np.all(np.abs(points - centre[None]) <= half, axis=1)


def esdf_slice_z(c, z: float):
    o, vs = np.asarray(c["esdf_origin"], float), float(c["esdf_voxel_size"])
    k = int(round((z - o[2]) / vs - 0.5))
    sl = np.asarray(c["esdf_distance"], np.float32)[:, :, k]          # [i(x), j(y)]
    ext = (o[0], o[0] + vs * sl.shape[0], o[1], o[1] + vs * sl.shape[1])
    return sl, ext, o[2] + (k + 0.5) * vs


def esdf_slice_y(c, y: float):
    o, vs = np.asarray(c["esdf_origin"], float), float(c["esdf_voxel_size"])
    j = int(round((y - o[1]) / vs - 0.5))
    sl = np.asarray(c["esdf_distance"], np.float32)[:, j, :]          # [i(x), k(z)]
    ext = (o[0], o[0] + vs * sl.shape[0], o[2], o[2] + vs * sl.shape[1])
    return sl, ext, o[1] + (j + 0.5) * vs


def draw_field(ax, sl, ext, vmax=0.06):
    ax.imshow(np.clip(sl.T, -0.01, vmax), origin="lower", extent=ext, cmap="Blues_r", vmin=-0.01, vmax=vmax)
    xs = np.linspace(ext[0], ext[1], sl.shape[0])
    ys = np.linspace(ext[2], ext[3], sl.shape[1])
    ax.contour(xs, ys, sl.T, levels=[0.0], colors="black", linewidths=0.8)


# ---------------------------------------------------------------- figure 1
def figure1(steps, chunks, gt, grasp):
    src_v = RUN / "third_person_front.mp4"
    fig, axs = plt.subplots(1, 4, figsize=(8.0, 1.9))
    for ax, (label, idx) in zip(axs, KEYFRAMES.items()):
        ax.imshow(video_frame(idx))
        ax.set_title(f"{label} (step {idx})")
        ax.axis("off")
    save(fig, 1, "f1_keyframes", [src_v],
         "Third-person keyframes of the closed-loop AG3S run (approach, grasp, carry, release).",
         {"frames": KEYFRAMES, "crop_xyxy": VIDEO_CROP,
          "grasp_state_gt": {k: str(grasp["grasp_state"][min(v // 8, 74)]) for k, v in KEYFRAMES.items()}})

    st = np.load(steps[CH_APPROACH])
    c = chunk(chunks, CH_APPROACH)
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.2))
    for ax, (img, att, name) in zip(axs, [("image_cam_high", "attention_head", "head camera"),
                                          ("image_cam_left_wrist", "attention_left_wrist", "left wrist"),
                                          ("image_cam_right_wrist", "attention_right_wrist", "right wrist (active)")]):
        overlay(st[img], c[att], ax, name)
    save(fig, 1, "f1_rgb_attention", [steps[CH_APPROACH], c["path"]],
         "Policy RGB (224x224) with the served VLA attention map (layer 8, head 2, last action query, "
         "16x16, min-max scaled per view for display).", {"chunk": CH_APPROACH})

    c = chunk(chunks, CH_GROUND)
    cloud = np.asarray(c["filtered_cloud"], float)
    keep = ~np.asarray(c["support_mask"], bool)
    tgt = np.asarray(c["target_points"], float)
    centre = tgt.mean(0)
    win = window(cloud, centre, 0.13) & keep
    nb, nb_name = neighbour_mask(cloud, gt, CH_GROUND)
    att = np.asarray(c["attention_values"], float)
    fig = plt.figure(figsize=(7.2, 3.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    lo, hi = win & (att < 0.5), win & (att >= 0.5)
    ax.scatter(*cloud[lo].T, c=C_SCENE, s=2.0, depthshade=False)
    ax.scatter(*cloud[hi].T, c=att[hi], cmap="YlOrRd", s=4.0, vmin=0.3, vmax=1, depthshade=False)
    ax.set_title("fused cloud (attention >= 0.5 coloured)")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(*cloud[win & ~nb].T, c=C_SCENE, s=2.0, depthshade=False)
    ax2.scatter(*cloud[win & nb].T, c=C_NEIGH, s=4.0, depthshade=False)
    ax2.scatter(*tgt.T, c=C_TARGET, s=8.0, depthshade=False)
    ax2.set_title("grounded target vs. neighbour")
    for a in (ax, ax2):
        a.view_init(elev=35, azim=-60)
        a.set_xticks([]); a.set_yticks([]); a.set_zticks([])
        a.set_box_aspect((1, 1, 0.6))
    fig.legend(handles=[Line2D([], [], ls="", marker="o", color=C_TARGET, label="target (ATS output)"),
                        Line2D([], [], ls="", marker="o", color=C_NEIGH, label=f"neighbour {nb_name} (GT colour only)"),
                        Line2D([], [], ls="", marker="o", color=C_SCENE, label="other observed surface")],
               loc="lower center", ncol=3, frameon=False)
    save(fig, 1, "f1_grounding_3d", [c["path"], T53 / f"gt/{KEY}.npz"],
         "Fused multi-view cloud near the target and the ATS target cluster. Neighbour colour uses the "
         f"GT fruit centre (radius {NEIGHBOUR_RADIUS_M} m) for display only; the method never sees GT.",
         {"chunk": CH_GROUND, "n_target_points": int(len(tgt)), "neighbour": nb_name})

    c = chunk(chunks, CH_CLOSE)
    names, excl = link_masks(c)
    sc, sr = np.asarray(c["sphere_centres"], float), np.asarray(c["sphere_radii"], float)
    tgt = np.asarray(c["target_points"], float)
    cloud = np.asarray(c["filtered_cloud"], float)
    nb, _ = neighbour_mask(cloud, gt, CH_CLOSE)
    centre = tgt.mean(0)
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    m = window(cloud, centre, 0.12) & ~np.asarray(c["support_mask"], bool)
    ax.scatter(cloud[m & nb, 0], cloud[m & nb, 2], s=2, c=C_NEIGH)
    ax.scatter(tgt[:, 0], tgt[:, 2], s=4, c=C_TARGET)
    near = window(sc, centre, 0.12)
    for (x, _, z), r, e in zip(sc[near], sr[near], excl[near]):
        ax.add_patch(Circle((x, z), r, fill=False, lw=0.8, ec=C_CONTACT if e else C_LINK))
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title("grasp contact (side view)")
    ax.legend(handles=[Line2D([], [], color=C_CONTACT, label="contact-link spheres"),
                       Line2D([], [], ls="", marker="o", color=C_TARGET, label="target"),
                       Line2D([], [], ls="", marker="o", color=C_NEIGH, label="neighbour")],
              loc="upper right", frameon=False)
    save(fig, 1, "f1_grasp_contact", [c["path"], T53 / f"gt/{KEY}.grasp.npz"],
         "Robot query spheres at the closing chunk; GT records finger contact with the pear at this step.",
         {"chunk": CH_CLOSE, "gt_touching": str(grasp["touching"][CH_CLOSE]),
          "gt_finger_contact_lr": grasp["finger_contact_lr"][CH_CLOSE].tolist(),
          "latch_state": c["summary"]["grasp"]["state"]})


# ---------------------------------------------------------------- figure 2
def figure2(steps, chunks, gt):
    c = chunk(chunks, CH_GROUND)
    s = c["summary"]
    tf = s["esdf_stats"]["target_free"]
    tf = ast.literal_eval(tf) if isinstance(tf, str) else tf   # dumped as a dict repr string
    ball = tf["ball"]
    bc, br = np.asarray(ball["centre_m"]), ball["radius_mm"] / 1000.0
    tgt = np.asarray(c["target_points"], float)
    cloud = np.asarray(c["filtered_cloud"], float)
    support = np.asarray(c["support_mask"], bool)
    nb, nb_name = neighbour_mask(cloud, gt, CH_GROUND)
    # non-target evidence as the pipeline builds it for `proximity`: fused cloud - target - support
    tset = {tuple(np.round(p, 5)) for p in tgt}
    is_t = np.array([tuple(np.round(p, 5)) in tset for p in cloud])
    nontarget = cloud[~support & ~is_t]
    sl, ext, zs = esdf_slice_z(c, bc[2])
    names, excl = link_masks(c)
    sc, sr = np.asarray(c["sphere_centres"], float), np.asarray(c["sphere_radii"], float)
    half = 0.11
    lim = (bc[0] - half, bc[0] + half, bc[1] - half, bc[1] + half)

    def base(ax, title):
        draw_field(ax, sl, ext)
        ax.set_xlim(lim[:2]); ax.set_ylim(lim[2:]); ax.set_aspect("equal")
        ax.set_title(title); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        band = np.abs(cloud[:, 2] - zs) < 0.01
        ax.scatter(cloud[band & nb, 0], cloud[band & nb, 1], s=2, c=C_NEIGH)
        tb = np.abs(tgt[:, 2] - zs) < 0.01
        ax.scatter(tgt[tb, 0], tgt[tb, 1], s=3, c=C_TARGET)

    rz = np.sqrt(max(br ** 2 - (zs - bc[2]) ** 2, 0.0))
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    base(ax, "(a) scene field: every surface")
    save(fig, 2, "f2a_all_surfaces", [c["path"]],
         "Recorded 5 mm scene ESDF (main layer) sliced at the target height. Target and neighbour are both obstacles.",
         {"chunk": CH_GROUND, "slice_z_m": zs})

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    base(ax, "(b) ball exemption")
    ax.add_patch(Circle(bc[:2], rz, fill=True, fc=C_HOLD, alpha=0.12, ec=C_HOLD, lw=1.0, ls="--"))
    inball = (np.linalg.norm(nontarget - bc[None], axis=1) <= br) & (nontarget[:, 2] >= ball["z_min_m"])
    band = np.abs(nontarget[:, 2] - zs) < 0.01
    ax.scatter(nontarget[inball & band, 0], nontarget[inball & band, 1], s=5, marker="x", c=C_HOLD, lw=0.6)
    save(fig, 2, "f2b_ball", [c["path"]],
         "The recorded target ball (surface-fit centre, quantile-1.0 radius + 1 voxel) drawn on the same slice. "
         "Red x = observed non-target points inside the ball, which a plain ball exemption would erase.",
         {"chunk": CH_GROUND, "ball": ball, "n_nontarget_points_in_ball": int(inball.sum())})

    # ownership panel: the serving split function on a slice grid, recorded target / non-target points
    tb = TargetBall(centre=bc, radius=br, z_min=ball["z_min_m"])
    owned = s["target_exclusion"]["owned"]
    ev = OwnedEvidence(target_points=tgt, nontarget_points=nontarget, tie_voxels=owned["tie_voxels"],
                       max_dist_voxels=owned["max_dist_voxels"], unowned=owned["unowned"])
    vs = 0.005
    g = np.arange(-br, br + vs, vs)
    gx, gy = np.meshgrid(bc[0] + g, bc[1] + g, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, zs)], 1)
    inside = (np.linalg.norm(pts - bc[None], axis=1) <= br) & (pts[:, 2] >= ball["z_min_m"])
    pts = pts[inside]
    from scipy.spatial import cKDTree
    surf = np.concatenate([tgt, nontarget])
    seed = cKDTree(surf).query(pts)[0] <= vs        # surface-adjacent cells stand in for TSDF seed voxels
    res = _ball_owned_split(pts, seed, tb, ev, vs)
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    base(ax, "(c) ownership inside the ball")
    ax.add_patch(Circle(bc[:2], rz, fill=False, ec="black", lw=0.8, ls="--"))
    for key, col, lab in [("target", C_TARGET, "removed: target-owned"),
                          ("unowned", "#9E9E9E", "removed: unowned"),
                          ("nontarget", C_NEIGH, "kept: neighbour-owned"),
                          ("ambiguous", "#9ECAE1", "kept: ambiguous")]:
        m = res[key]
        ax.scatter(pts[m, 0], pts[m, 1], s=11, marker="s", c=col, label=lab, edgecolors="black", lw=0.2, zorder=4)
    ax.legend(loc="lower left", frameon=True, framealpha=0.85, fontsize=5.5, markerscale=0.8)
    save(fig, 2, "f2c_ownership", [c["path"]],
         "Serving function _ball_owned_split (T44 I1) applied on a 5 mm slice grid inside the recorded ball, "
         "with the recorded target points and non-target evidence (fused cloud - target - support). "
         "Grid cells within one voxel of an observed point stand in for TSDF seed voxels (display approximation). "
         "Serving counts for this chunk are in target_exclusion of the dump.",
         {"chunk": CH_GROUND, "owned_params": owned,
          "serving_counts_this_chunk": {k: s["target_exclusion"][k] for k in
                                        ("n_seeds_in_ball", "n_removed", "n_kept_nontarget", "n_ambiguous",
                                         "n_unowned_removed", "n_target_owned")},
          "grid_counts": {k: int(res[k].sum()) for k in ("target", "nontarget", "ambiguous", "unowned")}})

    # link-conditioned queries: which spheres read the target-free layer (top view, both hands)
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    for (x, y, _), r, e in zip(sc, sr, excl):
        ax.add_patch(Circle((x, y), r, fill=True, alpha=0.35, fc=C_CONTACT if e else C_LINK, ec="none"))
    ax.scatter(tgt[:, 0], tgt[:, 1], s=3, c=C_TARGET)
    m = window(cloud, bc, 0.12) & nb
    ax.scatter(cloud[m, 0], cloud[m, 1], s=2, c=C_NEIGH)
    allx = np.concatenate([sc[:, 0], tgt[:, 0]]); ally = np.concatenate([sc[:, 1], tgt[:, 1]])
    ax.set_xlim(allx.min() - 0.25, allx.max() + 0.25); ax.set_ylim(ally.min() - 0.08, ally.max() + 0.08)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("(c) link-specific queries (gripper scope)")
    ax.legend(handles=[Patch(fc=C_CONTACT, alpha=0.5, label="active hand: target-free field"),
                       Patch(fc=C_LINK, alpha=0.5, label="other hand: scene field")],
              loc="center", frameon=False, fontsize=5.5)
    save(fig, 2, "f2c_link_queries", [c["path"]],
         "Recorded query spheres coloured by target_permission.target_field_exclude "
         f"({s['target_permission']['n_rows_excluded']} of {s['target_permission']['n_rows']} rows; "
         f"links {s['target_permission']['excluded_links']}). Evaluated scope queries gripper links of both hands only "
         "(--links gripper); arm and wrist links are not queried in this configuration.",
         {"chunk": CH_GROUND, "excluded_links": s["target_permission"]["excluded_links"]})


# ---------------------------------------------------------------- figure 3
def figure3(steps, chunks, gt, grasp):
    st = np.load(steps[CH_APPROACH])
    c = chunk(chunks, CH_APPROACH)
    fig, axs = plt.subplots(2, 3, figsize=(5.4, 3.7))
    for k, (img, att, name) in enumerate([("image_cam_high", "attention_head", "head"),
                                          ("image_cam_left_wrist", "attention_left_wrist", "left wrist"),
                                          ("image_cam_right_wrist", "attention_right_wrist", "right wrist")]):
        axs[0, k].imshow(st[img]); axs[0, k].set_title(name); axs[0, k].axis("off")
        axs[1, k].imshow(np.asarray(c[att], float), cmap="inferno"); axs[1, k].axis("off")
    axs[1, 0].set_title("16x16 attention", loc="left")
    save(fig, 3, "f3_policy_inputs_attention", [steps[CH_APPROACH], c["path"]],
         "Policy inputs (3 RGB views) and the raw 16x16 action-to-image attention per view from the extraction pass.",
         {"chunk": CH_APPROACH})

    c = chunk(chunks, CH_GROUND)
    cloud = np.asarray(c["filtered_cloud"], float)
    att = np.asarray(c["attention_values"], float)
    seeds = np.asarray(c["seed_indices"], int)
    tgt = np.asarray(c["target_points"], float)
    win = window(cloud, tgt.mean(0), 0.3) & ~np.asarray(c["support_mask"], bool)
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.5))
    axs[0].scatter(cloud[win, 0], cloud[win, 1], c=att[win], cmap="YlOrRd", s=1.0, vmin=0, vmax=1)
    axs[0].scatter(cloud[seeds, 0], cloud[seeds, 1], s=3, facecolors="none", edgecolors="#00A3A3", lw=0.6)
    axs[0].set_title("fused attention cloud + seeds"); axs[0].set_aspect("equal")
    axs[1].scatter(cloud[win, 0], cloud[win, 1], c=C_SCENE, s=0.8)
    axs[1].scatter(tgt[:, 0], tgt[:, 1], c=C_TARGET, s=3)
    axs[1].set_title("grown, filtered target cluster"); axs[1].set_aspect("equal")
    for a in axs:
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
    save(fig, 3, "f3_lift_seed_cluster", [c["path"]],
         "Top view of the fused cloud with normalised attention, top-5% seeds (teal rings), and the selected target cluster.",
         {"chunk": CH_GROUND, "n_seeds": int(len(seeds)), "n_target_points": int(len(tgt)),
          "grounding_status": c["summary"]["grounding_status"]})

    sl, ext, ys = esdf_slice_y(c, tgt.mean(0)[1])
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    draw_field(ax, sl, ext, vmax=0.1)
    ax.set_title("scene ESDF (5 mm), x-z slice"); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    save(fig, 3, "f3_scene_esdf", [c["path"]],
         "Recorded cuRobo main ESDF layer (5 mm, 128^3) sliced through the target; black = zero level set.",
         {"chunk": CH_GROUND, "slice_y_m": ys})

    c = chunk(chunks, CH_HELD)
    names, excl = link_masks(c)
    sc, sr = np.asarray(c["sphere_centres"], float), np.asarray(c["sphere_radii"], float)
    att_sph = c["summary"]["grasp"]["attached"]["held"]["spheres"]
    centre = np.asarray(att_sph[0]["centre_m"])
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    hand = window(sc, centre, 0.2)
    for (x, _, z), r in zip(sc[hand], sr[hand]):
        ax.add_patch(Circle((x, z), r, fill=True, alpha=0.3, fc=C_LINK, ec="none"))
    for sp in att_sph:
        ax.add_patch(Circle((sp["centre_m"][0], sp["centre_m"][2]), sp["radius_mm"] / 1000.0,
                            fill=True, alpha=0.35, fc=C_ATTACH, ec=C_ATTACH))
    ax.set_xlim(centre[0] - 0.2, centre[0] + 0.2); ax.set_ylim(centre[2] - 0.12, centre[2] + 0.25)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title("attach: target becomes robot geometry")
    ax.legend(handles=[Patch(fc=C_ATTACH, alpha=0.5, label="attached object (scene-field query)"),
                       Patch(fc=C_LINK, alpha=0.5, label="robot spheres")], loc="upper left", frameon=False, fontsize=5.5)
    save(fig, 3, "f3_attached_object", [c["path"]],
         "After the grasp latch reaches HELD, the target becomes an attached sphere on the hand that queries the scene field.",
         {"chunk": CH_HELD, "attached": c["summary"]["grasp"]["attached"]["held"]["spheres"],
          "parent_link": c["summary"]["grasp"]["attached"]["parent_link"]})

    # CCR: chunk with the largest refined-reference deviation over the planned rows (right arm)
    best, bj = -1.0, None
    for j in range(len(chunks)):
        z = np.load(chunks[j])
        d = float(np.abs(z["refined_chunk"][:16, 8:15] - z["reference_chunk"][:16, 8:15]).max())
        if d > best:
            best, bj = d, j
    c = chunk(chunks, bj)
    ref, ref2 = np.asarray(c["reference_chunk"], float), np.asarray(c["refined_chunk"], float)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    rows = np.arange(16)
    ax.axvspan(-0.5, 7.5, color="#EEEEEE", label="executed rows (K=8)")
    for k, col in zip(range(8, 15), plt.cm.tab10.colors):
        ax.plot(rows, np.degrees(ref[:16, k]), ls="--", lw=0.8, color=col)
        ax.plot(rows, np.degrees(ref2[:16, k]), ls="-", lw=1.0, color=col)
    ax.set_xlabel("chunk row"); ax.set_ylabel("joint angle (deg)")
    ax.set_title("CCR: reference (dashed) vs refined")
    ax.legend(loc="best", frameon=False, fontsize=5.5)
    save(fig, 3, "f3_ccr_refinement", [c["path"]],
         "Right-arm joints of the policy reference chunk and the SQP-refined chunk over the planned rows "
         "(8 executed + 8 look-ahead), chunk with the largest deviation in this run.",
         {"chunk": bj, "max_abs_deviation_rad": best, "to_status": c["summary"]["to"]["status"],
          "verdict": [r["kind"] for r in c["summary"]["verdict"]["reasons"]]})

    # per-chunk timeline: grasp latch, HSR probabilities, exclusion source, execute/hold
    states, p, src, hold = [], [], [], []
    for j in range(len(chunks)):
        s = json.loads(str(np.load(chunks[j])["summary_json"]))
        states.append(s.get("grasp", {}).get("state"))
        sp = s.get("subtask", {}).get("p") or {}
        p.append([sp.get("pick", np.nan), sp.get("place", np.nan), sp.get("home", np.nan)])
        src.append(s.get("exclusion", {}).get("source"))
        hold.append(any(r.get("action") == "hold" for r in s.get("verdict", {}).get("reasons", [])))
    p = np.asarray(p, float)
    n = len(chunks)
    order = ["searching", "latched", "closing", "held", "placed"]
    cols = {"searching": "#F2F2F2", "latched": "#DEEBF7", "closing": "#FFE699", "held": "#E4D3F0", "placed": "#E2F0D9"}
    fig, axs = plt.subplots(3, 1, figsize=(6.4, 2.9), sharex=True, gridspec_kw={"height_ratios": [1, 1.6, 0.7]})
    for j, sname in enumerate(states):
        axs[0].axvspan(j - 0.5, j + 0.5, color=cols.get(sname, "white"), lw=0)
    for sname in order:
        idx = [j for j, v in enumerate(states) if v == sname]
        if idx:
            axs[0].text(np.mean(idx), 0.5, sname, ha="center", va="center", fontsize=6.5)
    axs[0].set_yticks([]); axs[0].set_ylabel("grasp\nlatch", rotation=0, ha="right", va="center")
    for k, (lab, col) in enumerate(zip(["pick", "place", "home"], ["#5B9BD5", "#ED7D31", "#70AD47"])):
        axs[1].plot(np.arange(n), p[:, k], color=col, lw=1.0, label=lab)
    axs[1].set_ylabel("sub-task\nprobe p", rotation=0, ha="right", va="center")
    axs[1].legend(loc="center right", frameon=False, ncol=3)
    srcmap = {"manipulated": (C_TARGET, "target-free layer"), "attached": (C_ATTACH, "attached object"), None: ("#FFFFFF", "none")}
    for j, v in enumerate(src):
        axs[2].axvspan(j - 0.5, j + 0.5, color=srcmap.get(v, ("#FFFFFF", ""))[0], alpha=0.6, lw=0)
    hj = [j for j, h in enumerate(hold) if h]
    axs[2].scatter(hj, [0.5] * len(hj), marker="v", c=C_HOLD, s=12, zorder=3)
    axs[2].set_yticks([]); axs[2].set_ylabel("contact\nhandling", rotation=0, ha="right", va="center")
    axs[2].set_xlabel("chunk index (8 control steps each)")
    axs[2].legend(handles=[Patch(fc=C_TARGET, alpha=0.6, label="target-free layer for contact links"),
                           Patch(fc=C_ATTACH, alpha=0.6, label="attached object"),
                           Line2D([], [], ls="", marker="v", color=C_HOLD, label="HOLD chunk")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.9), ncol=3, frameon=False)
    save(fig, 3, "f3_state_timeline", list(chunks[:1]),
         "Per-chunk recorded grasp-latch state, sub-task probe probabilities, contact-handling source and HOLD "
         "chunks over the whole episode (all chunk dumps of this run).",
         {"n_chunks": n, "states": states, "exclusion_source": src, "hold_chunks": hj})


def montage(fig_no: int, names: list[str], cols: int, title: str):
    ims = [Image.open(OUT / f"figure{fig_no}/assets/{n}.png").convert("RGB") for n in names]
    h = 520
    ims = [im.resize((int(im.width * h / im.height), h)) for im in ims]
    rows = [ims[i:i + cols] for i in range(0, len(ims), cols)]
    W = max(sum(im.width for im in r) + 20 * (len(r) - 1) for r in rows)
    H = len(rows) * h + 20 * (len(rows) - 1) + 60
    canvas = Image.new("RGB", (W + 40, H + 40), "white")
    from PIL import ImageDraw
    from PIL import ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    ImageDraw.Draw(canvas).text((20, 15), title, fill="black", font=font)
    y = 70
    for r in rows:
        x = 20
        for im in r:
            canvas.paste(im, (x, y)); x += im.width + 20
        y += h + 20
    path = OUT / f"figure{fig_no}/reference_montage.png"
    canvas.save(path)
    print("wrote", path.relative_to(REPO))


def main():
    steps, chunks, gt, grasp = load()
    figure1(steps, chunks, gt, grasp)
    figure2(steps, chunks, gt)
    figure3(steps, chunks, gt, grasp)
    montage(1, ["f1_keyframes", "f1_rgb_attention", "f1_grounding_3d", "f1_grasp_contact"], 2,
            "F1 reference (AG3S closed loop, T53 C1a ep1842): keyframes | RGB+attention | 3D grounding | grasp contact")
    montage(2, ["f2a_all_surfaces", "f2b_ball", "f2c_ownership", "f2c_link_queries"], 4,
            "F2 reference (same chunk): (a) scene field | (b) ball exemption | (c) ownership + link-specific queries")
    montage(3, ["f3_policy_inputs_attention", "f3_lift_seed_cluster", "f3_scene_esdf",
                "f3_attached_object", "f3_ccr_refinement", "f3_state_timeline"], 3,
            "F3 reference (module outputs from the same run)")
    (OUT / "assets_index.json").write_text(json.dumps(SOURCES, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
