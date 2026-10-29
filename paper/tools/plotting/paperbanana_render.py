"""High-resolution replay renders for the PaperBanana F1-F3 references.

Every frame is the recorded simulator state of one AG3S closed-loop run (T53, serving arm C1a,
episode 1842) re-rendered with MuJoCo at a higher resolution and from chosen viewpoints. Nothing is
re-simulated: ``qpos`` comes from the client step records. Overlays come from the server chunk dumps
(target cluster, fused cloud and attention, query spheres, ball, ESDF voxels) or from FK of the recorded
reference / refined chunks. The ownership split is the serving function ``_ball_owned_split``.

Run (CPU renderer, no GPU):
    MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/paperbanana_render.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "AG3S"))
sys.path.insert(0, str(REPO / "AG3S/ag3s/experiments/studies"))
sys.path.insert(0, str(REPO / "AG3S/ag3s/experiments"))
from ag3s.fields.curobo_builder import (OwnedEvidence, TargetBall, _ball_owned_split,  # noqa: E402
                                        nontarget_surface_evidence)
import t53_records  # noqa: E402

RUNROOT = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/"
                       "20261013T100210+0900_4a540f1a18d7_cb037c425d2f")
T53 = RUNROOT / "verify/T53"
XML = RUNROOT / "snapshot/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
KEY = "ep1842"
RUN = T53 / f"runs/C1a/R/C1a_{KEY}_s18421"
OUT = REPO / "paper/figures/paperbanana"

# chunk indices (record step j == control step 8j), chosen from the recorded grasp timeline
CH = {"approach": 10, "ground": 17, "contact": 18, "attach": 19, "grasp": 20,
      "carry": 26, "release": 34, "ccr": 11}

# palette shared by the three figures (RGBA 0-1)
TARGET = (0.91, 0.44, 0.10, 1.0)
NEIGH = (0.18, 0.46, 0.71, 1.0)
GOLD = (0.95, 0.72, 0.05, 1.0)
GREY = (0.55, 0.55, 0.55, 1.0)
PURPLE = (0.44, 0.19, 0.63, 0.45)
RED = (0.80, 0.10, 0.10, 1.0)
CONTACT_BODIES = ("EE_BODY_R", "ee_finger_r1", "ee_finger_r2")
OTHER_HAND = ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2")
FRUITS = ("pear", "orange", "apple", "banana")

SIDE: dict[str, dict] = {}


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ data
class Run:
    def __init__(self):
        self.steps = sorted((RUN / "ag3s/run_0000").glob("step_*.npz"))
        acts = [np.load(f)["actions"] for f in self.steps]
        segs = t53_records.segments(sorted(T53.glob("runs/C1a/server_*/server_constraints/run_*")))
        i = t53_records.pair_segment(segs, acts)
        if i is None:
            raise SystemExit("no server segment pairs with the client planning rows")
        self.chunks = list(segs[i])
        self.gt = np.load(T53 / f"gt/{KEY}.npz")
        self.grasp = np.load(T53 / f"gt/{KEY}.grasp.npz")

    def step(self, j):
        return np.load(self.steps[j])

    def chunk(self, j):
        z = np.load(self.chunks[j], allow_pickle=False)
        d = {k: z[k] for k in z.files}
        s = json.loads(str(d.pop("summary_json")))
        tf = s.get("esdf_stats", {}).get("target_free")
        if isinstance(tf, str):
            s["esdf_stats"]["target_free"] = ast.literal_eval(tf)
        d["summary"], d["path"] = s, self.chunks[j]
        return d


# ------------------------------------------------------------------ renderer
class Scene:
    def __init__(self, W=1600, H=1200):
        self.m = mujoco.MjModel.from_xml_path(str(XML))
        self.m.vis.global_.offwidth = max(W, 1600)
        self.m.vis.global_.offheight = max(H, 1200)
        self.d = mujoco.MjData(self.m)
        self.base_rgba = self.m.geom_rgba.copy()
        self.r = mujoco.Renderer(self.m, H, W, max_geom=30000)
        self.W, self.H = W, H

    def body_geoms(self, names):
        ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
        return np.where(np.isin(self.m.geom_bodyid, [i for i in ids if i >= 0]))[0]

    def reset_colours(self):
        self.m.geom_rgba[:] = self.base_rgba

    def tint(self, names, rgba=None, alpha=None, mix=0.65):
        g = self.body_geoms(names)
        if rgba is not None:
            self.m.geom_rgba[g, :3] = (1 - mix) * self.base_rgba[g, :3] + mix * np.asarray(rgba[:3])
        if alpha is not None:
            self.m.geom_rgba[g, 3] = alpha

    def set(self, qpos):
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)

    def resize(self, W, H):
        self.r.close()
        self.r = mujoco.Renderer(self.m, H, W, max_geom=30000)
        self.W, self.H = W, H

    def render(self, cam, extras=()):
        if isinstance(cam, str):
            self.r.update_scene(self.d, cam)
        else:
            c = mujoco.MjvCamera()
            c.lookat[:], c.distance, c.azimuth, c.elevation = cam["lookat"], cam["dist"], cam["az"], cam["el"]
            self.r.update_scene(self.d, c)
        sc = self.r.scene
        for kind, a, b, rgba in extras:
            if sc.ngeom >= sc.maxgeom:
                break
            g = sc.geoms[sc.ngeom]
            if kind == "sphere":
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([b, 0, 0]), np.asarray(a, float),
                                    np.eye(3).ravel(), np.asarray(rgba, np.float32))
            elif kind == "box":
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([b, b, b]), np.asarray(a, float),
                                    np.eye(3).ravel(), np.asarray(rgba, np.float32))
            elif kind == "line":
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                    np.asarray(rgba, np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, b, np.asarray(a[0], float), np.asarray(a[1], float))
            sc.ngeom += 1
        return self.r.render().copy()


def save_png(img, fig_no, name, sources, note, extra=None):
    path = OUT / f"figure{fig_no}/assets/{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    meta = {"asset": str(path.relative_to(REPO)), "run": "T53 C1a (AG3S serving, 2026-10-13)", "episode": KEY,
            "script": "paper/tools/plotting/paperbanana_render.py", "renderer": f"MuJoCo {mujoco.__version__} replay of recorded qpos",
            "sources": {str(s): sha(s) for s in sources}, "note": note, "is_measured_result": False, **(extra or {})}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str) + "\n")
    SIDE[f"figure{fig_no}/{name}"] = meta
    print("wrote", path.relative_to(REPO))


# ------------------------------------------------------------------ geometry helpers
def neighbour_mask(run, cloud, j):
    fruits = [str(f) for f in run.gt["fruits"]]
    pos = run.gt["fruit_pos_base"][j]
    t = fruits.index(str(run.gt["target"]))
    dist = [np.linalg.norm(pos[k] - pos[t]) if k != t else np.inf for k in range(len(fruits))]
    k = int(np.argmin(dist))
    return np.linalg.norm(cloud - pos[k][None], axis=1) <= 0.045, fruits[k]


def voxelise(points, vs):
    keys = np.unique(np.floor(points / vs).astype(np.int64), axis=0)
    return (keys + 0.5) * vs


def ownership(c, cloud_nt, vs=0.01, surface=False):
    """Serving ownership split on the voxelised observed surface inside the recorded ball.

    `surface=True` first drops target-surface points from the non-target evidence with the serving
    T46 F1 function (`--nontarget-evidence surface`, shell 2 voxels, tie = owned tie)."""
    s = c["summary"]
    ball = s["esdf_stats"]["target_free"]["ball"]
    tb = TargetBall(centre=np.asarray(ball["centre_m"]), radius=ball["radius_mm"] / 1000.0, z_min=ball["z_min_m"],
                    provenance={"centre_method": ball["centre_method"], "fit": ball["fit"]})
    own = s["target_exclusion"]["owned"]
    tgt = np.asarray(c["target_points"], float)
    info = None
    surface_pts = np.concatenate([tgt, cloud_nt])        # voxels = every observed surface; evidence may shrink
    if surface:
        drop, info = nontarget_surface_evidence(cloud_nt, tgt, tb, shell_m=2 * 0.005, tie_m=own["tie_voxels"] * 0.005)
        cloud_nt = cloud_nt[~drop]
    ev = OwnedEvidence(target_points=tgt, nontarget_points=cloud_nt, tie_voxels=own["tie_voxels"],
                       max_dist_voxels=own["max_dist_voxels"], unowned=own["unowned"])
    vox = voxelise(surface_pts, vs)
    inside = (np.linalg.norm(vox - tb.centre[None], axis=1) <= tb.radius) & (vox[:, 2] >= tb.z_min)
    res = _ball_owned_split(vox[inside], np.ones(int(inside.sum()), bool), tb, ev, 0.005)
    return vox, inside, res, tb, info


def split_cloud(c):
    cloud = np.asarray(c["filtered_cloud"], float)
    support = np.asarray(c["support_mask"], bool)
    tgt = np.asarray(c["target_points"], float)
    tree = cKDTree(tgt)
    is_t = tree.query(cloud)[0] < 1e-6
    return cloud, support, tgt, cloud[~support & ~is_t]


def near(points, centre, r):
    return np.linalg.norm(points - np.asarray(centre)[None], axis=1) <= r


def ee_path(sc: Scene, qpos, chunk_rows):
    """FK of the right TCP site for each recorded chunk row (right-arm columns 8..14)."""
    adr = [sc.m.jnt_qposadr[mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_JOINT, f"right_arm_{k}")] for k in range(7)]
    site = mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_SITE, "right_ee")
    out = []
    for row in chunk_rows:
        q = qpos.copy()
        q[adr] = row[8:15]
        sc.set(q)
        out.append(sc.d.site_xpos[site].copy())
    sc.set(qpos)
    return np.asarray(out)


def polyline(points, width, rgba, dashed=False):
    ex = []
    for i in range(len(points) - 1):
        if dashed and i % 2:
            continue
        ex.append(("line", (points[i], points[i + 1]), width, rgba))
    return ex


# ------------------------------------------------------------------ cameras
WIDE = {"lookat": [0.56, -0.14, 1.06], "dist": 1.28, "az": 200, "el": -24}
CLOSE = {"lookat": [0.51, -0.32, 0.875], "dist": 0.42, "az": 150, "el": -34}
CLOSE_SIDE = {"lookat": [0.50, -0.32, 0.89], "dist": 0.36, "az": 140, "el": -14}
F2CAM = {"lookat": [0.51, -0.32, 0.875], "dist": 0.40, "az": 150, "el": -42}


# ------------------------------------------------------------------ figure 1
def figure1(run: Run, sc: Scene):
    for label in ("approach", "grasp", "carry", "release"):
        j = CH[label]
        sc.reset_colours(); sc.set(run.step(j)["qpos"])
        img = sc.render(WIDE)
        save_png(img, 1, f"f1_key_{label}", [run.steps[j]], f"Replay of recorded state, chunk {j} (control step {8 * j}).",
                 {"chunk": j, "gt_grasp_state": str(run.grasp["grasp_state"][j])})

    # attention on a high-resolution re-render of the active wrist camera (same intrinsics, 4:3)
    j = CH["approach"]
    c = run.chunk(j)
    sc.resize(1280, 960); sc.reset_colours(); sc.set(run.step(j)["qpos"])
    rgb = sc.render("wrist_cam_r")
    att = np.asarray(c["attention_right_wrist"], float)
    att = (att - att.min()) / max(att.max() - att.min(), 1e-12)
    up = np.asarray(Image.fromarray((att * 255).astype(np.uint8)).resize((1280, 960), Image.BICUBIC)) / 255.0
    col = plt.cm.inferno(up)[..., :3]
    alpha = np.clip((up - 0.08) / 0.6, 0, 1)[..., None] * 0.85
    blend = (rgb / 255.0 * (1 - alpha) + col * alpha)
    save_png((blend * 255).astype(np.uint8), 1, "f1_wrist_attention", [run.steps[j], c["path"]],
             "Active (right) wrist camera re-rendered at 1280x960 from the recorded state; the served 16x16 "
             "attention (layer 8, head 2) is bicubic-upsampled onto it (the grid maps linearly onto the 4:3 frame).",
             {"chunk": j})
    sc.resize(1600, 1200)

    # 3D grounding in the scene: target cluster vs neighbour, fruit meshes faded
    j = CH["ground"]
    c = run.chunk(j)
    cloud, support, tgt, nt = split_cloud(c)
    nb, nb_name = neighbour_mask(run, cloud, j)
    centre = tgt.mean(0)
    sc.reset_colours(); sc.set(run.step(j)["qpos"]); sc.tint(FRUITS, alpha=0.25)
    ex = [("sphere", p, 0.0042, NEIGH) for p in cloud[nb & ~support]]
    ex += [("sphere", p, 0.0045, TARGET) for p in tgt]
    ex += [("sphere", p, 0.0030, (0.75, 0.75, 0.75, 0.9)) for p in cloud[~support & ~nb & near(cloud, centre, 0.14)]
           if not np.any(np.linalg.norm(tgt - p, axis=1) < 1e-6)]
    img = sc.render(CLOSE, ex)
    save_png(img, 1, "f1_grounding_scene", [c["path"], run.steps[j], T53 / f"gt/{KEY}.npz"],
             "Recorded ATS target cluster (orange) and the observed neighbour surface (blue) drawn in the replayed scene; "
             "fruit meshes faded. Neighbour colour uses the GT fruit centre for display only.",
             {"chunk": j, "n_target_points": int(len(tgt)), "neighbour": nb_name})

    # intended contact: contact links tinted gold, target points, neighbour intact
    j = CH["contact"]
    c = run.chunk(j)
    cloud, support, tgt, nt = split_cloud(c)
    nb, _ = neighbour_mask(run, cloud, j)
    sc.reset_colours(); sc.set(run.step(j)["qpos"]); sc.tint(CONTACT_BODIES, GOLD, mix=0.7); sc.tint(FRUITS, alpha=0.35)
    ex = [("sphere", p, 0.0042, NEIGH) for p in cloud[nb & ~support]] + [("sphere", p, 0.0045, TARGET) for p in tgt]
    img = sc.render(CLOSE_SIDE, ex)
    save_png(img, 1, "f1_contact_scene", [c["path"], run.steps[j], T53 / f"gt/{KEY}.grasp.npz"],
             "Closing chunk: contact links (gold) on the target cluster; GT records finger contact with the pear here.",
             {"chunk": j, "gt_touching": str(run.grasp["touching"][j]), "latch": c["summary"]["grasp"]["state"]})


# ------------------------------------------------------------------ figure 2
def figure2(run: Run, sc: Scene):
    j = CH["ground"]
    c = run.chunk(j)
    cloud, support, tgt, nt = split_cloud(c)
    vox, inside, res, tb, _ = ownership(c, nt)
    vs = 0.01
    sel = near(vox, tb.centre, 0.16)
    tset = cKDTree(tgt)
    is_tvox = tset.query(vox)[0] <= 0.0075
    BLUE_T = (0.18, 0.46, 0.71, 0.85)
    GHOST = (0.85, 0.15, 0.15, 0.22)
    sc.reset_colours(); sc.set(run.step(j)["qpos"]); sc.tint(FRUITS, alpha=0.18)
    cam = F2CAM

    # (a) every observed surface is an obstacle
    ex = [("box", p, vs * 0.42, BLUE_T) for p in vox[sel]]
    sc.tint(CONTACT_BODIES, GOLD, mix=0.7)
    save_png(sc.render(cam, ex), 2, "f2a_scene_render", [c["path"], run.steps[j]],
             "Observed surfaces near the target voxelised at 1 cm (fused cloud of this chunk): all are obstacles.",
             {"chunk": j})

    # (b) ball removes everything inside, including part of the neighbour
    ex = [("box", p, vs * 0.42, GHOST if inside_i else BLUE_T) for p, inside_i in zip(vox[sel], inside[sel])]
    ex.append(("sphere", tb.centre, tb.radius, (0.85, 0.15, 0.15, 0.12)))
    save_png(sc.render(cam, ex), 2, "f2b_ball_render", [c["path"], run.steps[j]],
             "Recorded ball (centre and radius from the dump) removes every voxel inside it for every link (red ghosts), "
             "including the neighbour's near side.",
             {"chunk": j, "ball_radius_mm": tb.radius * 1000, "n_vox_inside": int(inside.sum()),
              "n_vox_inside_not_target": int((inside & ~is_tvox).sum())})

    # (c) ownership split with the adopted F1 surface evidence (offline, serving functions); as-run variant kept
    for variant, surface in (("f2c_ownership_render", True), ("f2c_ownership_asrun_proximity", False)):
        vox, inside, res, tb, info = ownership(c, nt, surface=surface)
        _render_ownership(sc, c, run, j, cam, vox, inside, res, tb, info, variant, sel_centre=tb.centre)


def _render_ownership(sc, c, run, j, cam, vox, inside, res, tb, info, name, sel_centre):
    vs = 0.01
    BLUE_T = (0.18, 0.46, 0.71, 0.85)
    sel = near(vox, sel_centre, 0.16)
    idx_in = np.where(inside)[0]
    removed = np.zeros(len(vox), bool); removed[idx_in[res["target"] | res["unowned"]]] = True
    ambiguous = np.zeros(len(vox), bool); ambiguous[idx_in[res["ambiguous"]]] = True
    ex = []
    for p, r_, a_ in zip(vox[sel], removed[sel], ambiguous[sel]):
        ex.append(("box", p, vs * 0.42, (0.91, 0.44, 0.10, 0.25) if r_ else ((0.55, 0.75, 0.90, 0.85) if a_ else BLUE_T)))
    ex.append(("sphere", tb.centre, tb.radius, (0.3, 0.3, 0.3, 0.06)))
    note = ("Serving _ball_owned_split on the 1 cm voxelised observed surface inside the recorded ball: target-owned "
            "voxels removed (faint orange), neighbour-owned and ambiguous kept (blue / light blue). ")
    note += ("Non-target evidence first filtered by the serving T46 F1 function nontarget_surface_evidence (adopted "
             "method, T46c), applied offline to this recorded chunk; the T53 run itself used proximity evidence."
             if info is not None else "As run in T53 C1a (proximity evidence): target-surface residue stays ambiguous (T45 D-TF).")
    save_png(sc.render(cam, ex), 2, name, [c["path"], run.steps[j]], note,
             {"chunk": j, "counts": {k: int(res[k].sum()) for k in ("target", "nontarget", "ambiguous", "unowned")},
              "surface_evidence": info,
              "serving_counts_this_chunk": {k: c["summary"]["target_exclusion"][k] for k in
                                            ("n_seeds_in_ball", "n_removed", "n_kept_nontarget", "n_ambiguous")}})


def figure2_links(run: Run, sc: Scene):
    j = CH["ground"]
    c = run.chunk(j)
    cloud, support, tgt, nt = split_cloud(c)
    # link-conditioned queries: both hands, recorded query spheres coloured by target_field_exclude
    s = c["summary"]
    excl = np.asarray(s["target_permission"]["target_field_exclude"], bool)
    sph, rad = np.asarray(c["sphere_centres"], float), np.asarray(c["sphere_radii"], float)
    sc.reset_colours(); sc.set(run.step(j)["qpos"])
    sc.tint(CONTACT_BODIES, GOLD, mix=0.7, alpha=0.25); sc.tint(OTHER_HAND, (0.4, 0.4, 0.4), mix=0.5, alpha=0.25)
    ex = [("sphere", p, r * 1.15, (0.95, 0.70, 0.0, 0.95) if e else (0.30, 0.30, 0.30, 0.95)) for p, r, e in zip(sph, rad, excl)]
    ex += [("sphere", p, 0.0045, TARGET) for p in tgt]
    save_png(sc.render({"lookat": [0.50, 0.0, 1.0], "dist": 1.10, "az": 180, "el": -40}, ex), 2,
             "f2c_link_render", [c["path"], run.steps[j]],
             "Recorded query spheres: active-hand contact links (gold, query the target-free layer) vs the other hand "
             "(grey, scene field). Evaluated scope queries gripper links of both hands only.",
             {"chunk": j, "n_excluded": int(excl.sum()), "n_rows": int(len(excl)),
              "excluded_links": s["target_permission"]["excluded_links"]})


# ------------------------------------------------------------------ figure 3
def figure3(run: Run, sc: Scene):
    j = CH["approach"]
    c = run.chunk(j)
    sc.resize(960, 720); sc.reset_colours(); sc.set(run.step(j)["qpos"])
    for cam, key, name in [("zed_left", "attention_head", "head"), ("wrist_cam_l", "attention_left_wrist", "left_wrist"),
                           ("wrist_cam_r", "attention_right_wrist", "right_wrist")]:
        rgb = sc.render(cam)
        save_png(rgb, 3, f"f3_in_{name}", [run.steps[j]], f"{cam} re-rendered at 960x720 from the recorded state.", {"chunk": j})
        att = np.asarray(c[key], float)
        att = (att - att.min()) / max(att.max() - att.min(), 1e-12)
        up = np.asarray(Image.fromarray((att * 255).astype(np.uint8)).resize((960, 720), Image.BICUBIC)) / 255.0
        alpha = np.clip((up - 0.08) / 0.6, 0, 1)[..., None] * 0.85
        blend = rgb / 255.0 * (1 - alpha) + plt.cm.inferno(up)[..., :3] * alpha
        save_png((blend * 255).astype(np.uint8), 3, f"f3_att_{name}", [run.steps[j], c["path"]],
                 "Served 16x16 attention over the re-rendered view (min-max per view for display).", {"chunk": j})
    sc.resize(1600, 1200)

    # lifted attention in 3D
    j = CH["ground"]
    c = run.chunk(j)
    cloud, support, tgt, nt = split_cloud(c)
    att = np.asarray(c["attention_values"], float)
    seeds = np.asarray(c["seed_indices"], int)
    centre = tgt.mean(0)
    m = ~support & near(cloud, centre, 0.35)
    sc.reset_colours(); sc.set(run.step(j)["qpos"]); sc.tint(FRUITS, alpha=0.3)
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("att", ["#D0D0D0", "#FDB863", "#E8711A", "#B2182B"])
    ex = [("sphere", p, 0.004 + 0.002 * a, tuple(cm(a)[:3]) + (1.0,)) for p, a in zip(cloud[m], att[m])]
    ex += [("sphere", p, 0.0065, (0.0, 0.75, 0.75, 1.0)) for p in cloud[seeds]]
    cam_mid = {"lookat": [0.53, -0.20, 0.88], "dist": 0.72, "az": 160, "el": -42}
    save_png(sc.render(cam_mid, ex), 3, "f3_lifted_attention", [c["path"], run.steps[j]],
             "Fused multi-view cloud coloured by normalised attention; top-5% seeds in teal.",
             {"chunk": j, "n_seeds": int(len(seeds))})
    ex = [("sphere", p, 0.003, (0.75, 0.75, 0.75, 1.0)) for p in cloud[m]] + [("sphere", p, 0.0045, TARGET) for p in tgt]
    save_png(sc.render(cam_mid, ex), 3, "f3_target_cluster", [c["path"], run.steps[j]],
             "Grown, candidate-filtered and confirmed target cluster (orange) among observed surfaces.",
             {"chunk": j, "n_target_points": int(len(tgt))})

    # scene field: recorded 5 mm ESDF, near-surface voxels around the target
    o, vs = np.asarray(c["esdf_origin"], float), float(c["esdf_voxel_size"])
    dist = np.asarray(c["esdf_distance"], np.float32)
    ii = np.argwhere(np.abs(dist) <= 0.004)
    pts = o[None] + (ii + 0.5) * vs
    keep = near(pts, centre, 0.17) & (pts[:, 2] > 0.80)
    sc.reset_colours(); sc.set(run.step(j)["qpos"]); sc.tint(FRUITS, alpha=0.15)
    ex = [("box", p, vs * 0.45, (0.18, 0.46, 0.71, 0.75)) for p in pts[keep][::2]]
    save_png(sc.render(CLOSE, ex), 3, "f3_scene_field", [c["path"], run.steps[j]],
             "Recorded cuRobo scene ESDF (main 5 mm layer): voxels with |d| <= 4 mm near the target (every second voxel).",
             {"chunk": j, "n_voxels_drawn": int(keep.sum() // 2 + keep.sum() % 2)})

    # attached object during carry: attach-time sphere carried rigidly by the hand (FK)
    ja, jc = CH["attach"], CH["carry"]
    ca = run.chunk(ja)
    att_sph = ca["summary"]["grasp"]["attached"]["held"]["spheres"]
    hand = mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_BODY, "EE_BODY_R")
    sc.set(run.step(ja)["qpos"])
    R0, p0 = sc.d.xmat[hand].reshape(3, 3).copy(), sc.d.xpos[hand].copy()
    local = [(R0.T @ (np.asarray(sp["centre_m"]) - p0), sp["radius_mm"] / 1000.0) for sp in att_sph]
    sc.reset_colours(); sc.set(run.step(jc)["qpos"]); sc.tint(CONTACT_BODIES, GOLD, mix=0.5)
    R1, p1 = sc.d.xmat[hand].reshape(3, 3), sc.d.xpos[hand]
    ex = [("sphere", p1 + R1 @ lc, r, PURPLE) for lc, r in local]
    save_png(sc.render({"lookat": list(p1 + np.array([0.0, 0.0, -0.05])), "dist": 0.5, "az": 150, "el": -15}, ex), 3,
             "f3_attached_object", [ca["path"], run.steps[ja], run.steps[jc]],
             "Attached-object sphere recorded at attach (chunk 19), carried rigidly with the hand link (FK) and shown during carry.",
             {"attach_chunk": ja, "carry_chunk": jc, "spheres": att_sph, "parent_link": ca["summary"]["grasp"]["attached"]["parent_link"]})

    # CCR: right TCP path of the recorded reference vs refined chunk (16 planned rows)
    best = (-1.0, None)
    for jj in range(len(run.chunks)):
        z = np.load(run.chunks[jj])
        if np.abs(z["refined_chunk"][:8, 8:15] - z["reference_chunk"][:8, 8:15]).max() < 1e-4:
            continue
        qq = run.step(jj)["qpos"]
        a = ee_path(sc, qq, np.asarray(z["reference_chunk"], float)[:8])
        b = ee_path(sc, qq, np.asarray(z["refined_chunk"], float)[:8])
        sh = float(np.linalg.norm(a - b, axis=1).max())
        if sh > best[0]:
            best = (sh, jj)
    j = best[1]
    c = run.chunk(j)
    q = run.step(j)["qpos"]
    sc.reset_colours(); sc.set(q); sc.tint(CONTACT_BODIES, GOLD, mix=0.4)
    ref = ee_path(sc, q, np.asarray(c["reference_chunk"], float)[:16])
    refd = ee_path(sc, q, np.asarray(c["refined_chunk"], float)[:16])
    ex = polyline(ref, 0.0040, (0.25, 0.25, 0.25, 1.0), dashed=True) + polyline(refd, 0.0050, GOLD)
    ex += [("sphere", p, 0.0075, GOLD) for p in refd[:8]] + [("sphere", p, 0.005, (0.3, 0.3, 0.3, 1.0)) for p in ref[:8]]
    cloud, support, tgt, nt = split_cloud(c)
    ex += [("sphere", p, 0.0045, TARGET) for p in tgt]
    mid = (ref.mean(0) + tgt.mean(0)) / 2
    save_png(sc.render({"lookat": list(mid), "dist": 0.45, "az": 150, "el": -20}, ex), 3, "f3_ccr_paths",
             [c["path"], run.steps[j]],
             "Right TCP (site right_ee) through the 16 planned rows: policy reference (dashed grey) vs SQP-refined (gold; "
             "dots = 8 executed rows), by FK of the recorded chunks.",
             {"chunk": j, "selection": "chunk with the largest executed-row TCP shift in this run",
              "max_tcp_shift_mm": float(np.linalg.norm(ref - refd, axis=1).max() * 1000),
              "verdict": [r["kind"] for r in c["summary"]["verdict"]["reasons"]]})


def main():
    run, sc = Run(), Scene()
    figure1(run, sc)
    figure2(run, sc)
    figure2_links(run, sc)
    figure3(run, sc)
    (OUT / "render_index.json").write_text(json.dumps(SIDE, indent=1, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
