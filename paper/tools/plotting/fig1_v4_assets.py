"""Fig. 1 v4 assets: one SafeLIBERO scene, prior safety layers vs AG3S, all from recorded runs (CPU, read-only).

Scene: SafeLIBERO spatial Level I, task 2, episode 9 (init state 9, seed 7, obstacle wine bottle). Every arm below ran
this exact episode; the head camera and robot base are static, so one camera model (K, T_base_cam from the AG3S
geometry record) serves all runs.

Ours (AG3S final run pe123458), pixel-accurate re-rendering of the recorded constraint geometry ("cleaned" panels):
each head-camera depth pixel is lifted to 3D with the recorded depth, K and T_base_cam, and coloured by the recorded
point / voxel / sphere set it lies on. Nothing is re-computed by the method.
  r_main.png      obs 110 (call 18, held): attached bowl (purple, inside the recorded attached spheres), contact-link
                  spheres (gold), every other observed non-support surface kept as obstacle (blue)
  r_identity.png  obs 30 (call 2, latched): per-point noun attention on surface pixels + seed points (gold)
  r_extent.png    obs 50 (call 6, closing): voxels removed from the target-free field (orange) and the non-target
                  voxels kept next to them (blue)
  r_permission.png obs 90 (call 14, held): contact-link spheres (gold) and attached spheres (purple)
Prior (same episode, baseline runs of 2026-11-04):
  l_main.png      AEGIS run frame at obs 110 with the AEGIS detector boxes it used (dashed); the bottle has no box
  l_boxes.png     AEGIS run frame at obs 20 (first call) with the same boxes
  l_ee.png        KNOWS run frame at obs 110 with the KNOWS robot model: one end-effector ellipsoid, semi-axes
                  (0.04, 0.04, 0.07) m in the gripper frame (Baseline/paper_baselines/methods/knows/adapter.py)
Frames are 256 px renders upsampled 3x (768 px); overlays are drawn at 768 px. Sidecar: assets.json.

    .venv-ag3s/bin/python paper/tools/plotting/fig1_v4_assets.py
"""
import glob, hashlib, json, pathlib, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/segmentation"))
import gtto_feature_scene as fs  # noqa: E402  (read-only reuse: load_calls, map_chunks, HAND_LINKS)

BENCH = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
OUT = ROOT / "paper/figures/fig01_teaser/v4/assets"
STEM, TASK, EP = "safelibero_spatial_t2_e9", 2, 9
UP = 3
ORANGE, BLUE, GOLD, PURPLE, RED = (232, 113, 26), (46, 117, 182), (217, 164, 6), (112, 48, 160), (198, 40, 40)
KNOWS_EE_SEMI = (0.04, 0.04, 0.07)


def run_dir(pattern):
    hits = sorted(glob.glob(str(BENCH / pattern)))
    assert len(hits) == 1, (pattern, hits)
    return pathlib.Path(hits[0])


RUNS = {"ag3s": run_dir(f"*qe-pe123458-e1a-spatial-LI-t{TASK}-*"), "vla": run_dir(f"*qe-vla-e1a-spatial-LI-t{TASK}-*"),
        "aegis": run_dir("*_e1a-aegis-spatial-LI-LI-aegis"), "knows": run_dir("*_e1a-knows-spatial-LI-LI-knows")}


def episode(run):
    for line in (run / "episodes.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d["task_id"] == TASK and d["episode_idx"] == EP:
            return d
    raise KeyError(run)


def frames(run):
    return np.load(run / episode(run)["npz"])["obs_agentview_image"]


def cam(raw):                       # robosuite opengl rows -> OpenCV rows
    return np.ascontiguousarray(raw[::-1])


def big(img):
    return Image.fromarray(img).resize((256 * UP, 256 * UP), Image.LANCZOS)


GEOM = np.load(RUNS["ag3s"] / "episodes" / f"{STEM}_geom.npz", allow_pickle=True)
K = np.asarray(GEOM["K/head"][0]); T_BC = np.asarray(GEOM["T_base_cam/head"][0]); T_WB = np.asarray(GEOM["T_world_base"][0])
assert np.ptp(np.asarray(GEOM["T_base_cam/head"]), axis=0).max() < 1e-9, "head camera moved"


def project(P):
    P = np.asarray(P, np.float64).reshape(-1, 3)
    Tc = np.linalg.inv(T_BC)
    pc = P @ Tc[:3, :3].T + Tc[:3, 3]
    z = pc[:, 2]
    uv = pc[:, :2] / np.maximum(z[:, None], 1e-6) * K[0, 0] + K[:2, 2]
    return uv, z


def lift(obs):
    """Head depth pixels of the AG3S run at ``obs`` in the robot base frame, (256, 256, 3) + valid mask."""
    d = np.asarray(GEOM["depth_mm/head"][obs], np.float64) / float(GEOM["depth_scale_mm"])
    v, u = np.mgrid[0:256, 0:256].astype(np.float64)
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
    P = pc @ T_BC[:3, :3].T + T_BC[:3, 3]
    return P, d > 0


def near(P, valid, pts, thr):
    m = np.zeros(P.shape[:2], bool)
    if len(pts) == 0:
        return m
    dist, _ = cKDTree(np.asarray(pts, np.float64)).query(P.reshape(-1, 3), distance_upper_bound=thr)
    return (np.isfinite(dist).reshape(P.shape[:2])) & valid


def in_spheres(P, valid, S, pad=0.0):
    m = np.zeros(P.shape[:2], bool)
    flat = P.reshape(-1, 3)
    for c, r in zip(S[:, :3], S[:, 3]):
        m |= (np.linalg.norm(flat - c, axis=1) <= r + pad).reshape(P.shape[:2])
    return m & valid


def up_mask(m):
    """256 px boolean mask -> smooth 768 px mask (bilinear + threshold)."""
    f = Image.fromarray(np.uint8(m) * 255).resize((256 * UP, 256 * UP), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.2))
    return np.asarray(f) >= 128


def paint(img, m, col, alpha=0.5, edge=True, edge_w=2):
    """Fill mask ``m`` (768 px) with ``col`` at ``alpha`` and draw a crisp outline."""
    a = np.asarray(img, np.float64)
    a[m] = a[m] * (1 - alpha) + np.asarray(col, np.float64) * alpha
    if edge:
        e = m & ~binary_erosion(m, iterations=edge_w)
        a[e] = np.asarray(col, np.float64) * 0.85
    return Image.fromarray(np.uint8(np.clip(a, 0, 255)))


def sphere_mask(S):
    """Union of the projected discs of spheres S (n, 4) at 768 px."""
    im = Image.new("L", (256 * UP, 256 * UP), 0); dr = ImageDraw.Draw(im)
    uv, z = project(S[:, :3])
    for (u, v), zz, r in zip(uv, z, S[:, 3]):
        if zz <= 0:
            continue
        rp = K[0, 0] * r / zz * UP; x, y = (u + 0.5) * UP, (v + 0.5) * UP
        dr.ellipse([x - rp, y - rp, x + rp, y + rp], fill=255)
    return np.asarray(im) > 0


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {}
    ag = RUNS["ag3s"]
    calls = fs.load_calls(ag, STEM)
    seg, how = fs.map_chunks(ag, STEM, calls)
    obs_of = [int(c["obs_index"]) for c in calls]
    grasp = [((c.get("grasp") or {}).get("state") or "") for c in calls]
    A = {k: frames(RUNS[k]) for k in ("ag3s", "aegis", "knows", "vla")}

    def chunk(i):
        return np.load(seg[i], allow_pickle=True)

    def save(name, img, **meta):
        p = OUT / name; img.save(p); rec[name] = dict(meta, sha256=sha(p)); print("wrote", p.relative_to(ROOT))

    def robot_and_support(z, P, valid):
        S = np.c_[np.asarray(z["sphere_centres"]), np.asarray(z["sphere_radii"])]
        return in_spheres(P, valid, S, pad=0.012)

    # ---------------------------------------------------------------- ours: main (held, obs 110)
    i = obs_of.index(110); z = chunk(i); P, valid = lift(110)
    names = list(json.loads(str(z["summary_json"])).get("sphere_link_names") or [])
    S = np.c_[np.asarray(z["sphere_centres"]), np.asarray(z["sphere_radii"])]
    hand = np.array([n in fs.HAND_LINKS for n in names])
    att = np.asarray(z["attached_query_spheres"], np.float64).reshape(-1, 4)
    held = in_spheres(P, valid, att, pad=0.004)
    robot = in_spheres(P, valid, S, pad=0.012)
    cloud = np.asarray(z["filtered_cloud"]); sup = np.asarray(z["support_mask"], bool)
    plane = np.asarray(z["support_planes"])[0]                    # recorded table plane (n, d), base frame
    above = (P.reshape(-1, 3) @ plane[:3] + plane[3]).reshape(256, 256) > 0.008
    obst = near(P, valid, cloud[~sup], 0.015) & above & ~held & ~robot    # stored cloud spacing ~7.6 mm
    img = big(cam(A["ag3s"][110]))
    img = paint(img, up_mask(obst), BLUE, alpha=0.35, edge_w=2)
    img = paint(img, up_mask(held), PURPLE, alpha=0.60, edge_w=2)
    img = paint(img, sphere_mask(S[hand]), GOLD, alpha=0.30, edge_w=3)
    save("r_main.png", img, source_chunk=seg[i], call_seq=i, obs_index=110, grasp=grasp[i], run=str(ag),
         content="blue = depth pixels more than 8 mm above the recorded support plane and within 15 mm of a non-support "
                 "point of the recorded filtered cloud (median spacing 7.6 mm), i.e. observed surface kept as obstacle, "
                 "outside the robot spheres (+12 mm) and the attached spheres; purple = depth pixels inside the recorded "
                 "attached spheres (+4 mm); gold = projected union of the hand / finger collision spheres (contact links)",
         n_attached_spheres=int(len(att)), n_hand_spheres=int(hand.sum()))

    # ---------------------------------------------------------------- ours: identity (obs 30)
    i = obs_of.index(30); z = chunk(i); P, valid = lift(30)
    C = np.asarray(z["attention_cloud"], np.float64); val = np.asarray(z["attention_values"], np.float64)
    dist, idx = cKDTree(C).query(P.reshape(-1, 3), distance_upper_bound=0.006)
    ok = np.isfinite(dist)
    w = np.zeros(256 * 256); w[ok] = val[idx[ok]]
    hi = np.quantile(val, 0.995); w = np.clip(w / max(hi, 1e-12), 0, 1).reshape(256, 256) * valid
    W = np.asarray(Image.fromarray(np.uint8(w * 255)).resize((768, 768), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2)), np.float64) / 255
    base = np.asarray(big(cam(A["ag3s"][30])), np.float64)
    col = np.stack([np.full_like(W, 255), 200 - 120 * W, 30 * (1 - W)], -1)        # gold -> orange-red
    a = (0.85 * np.clip(W, 0, 1) ** 0.6)[..., None]
    img = Image.fromarray(np.uint8(base * (1 - a) + col * a)).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    uv, zz = project(C[np.asarray(z["seed_indices"])])
    for (u, v), d0 in zip(uv, zz):
        if d0 > 0:
            x, y = (u + 0.5) * UP, (v + 0.5) * UP
            dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=GOLD + (255,), outline=(70, 50, 0, 255), width=1)
    save("r_identity.png", Image.alpha_composite(img, ov).convert("RGB"), source_chunk=seg[i], call_seq=i, obs_index=30,
         grasp=grasp[i], n_seeds=int(len(z["seed_indices"])),
         content="per-point noun attention (attention_values on attention_cloud) copied to the depth pixels within 6 mm of "
                 "a point, normalised by the 99.5 % quantile; gold dots = seed points. The 16x16 attention_head map is not used.")

    # ---------------------------------------------------------------- ours: extent (obs 50)
    i = obs_of.index(50); z = chunk(i); P, valid = lift(50)
    o, vs = np.asarray(z["target_free_trace_origin"]), float(z["target_free_trace_voxel_size"])
    rem = o + (np.asarray(z["target_free_removed_ijk"], np.float64) + 0.5) * vs
    kep = o + (np.asarray(z["target_free_nontarget_kept_ijk"], np.float64) + 0.5) * vs
    S = np.c_[np.asarray(z["sphere_centres"]), np.asarray(z["sphere_radii"])]
    robot = in_spheres(P, valid, S, pad=0.006)
    m_rem = near(P, valid, rem, 0.5 * vs * 1.8) & ~robot
    m_kep = near(P, valid, kep, 0.5 * vs * 1.8) & ~robot & ~m_rem
    img = big(cam(A["ag3s"][50]))
    img = paint(img, up_mask(m_kep), BLUE, alpha=0.65, edge_w=2)
    img = paint(img, up_mask(m_rem), ORANGE, alpha=0.60, edge_w=2)
    save("r_extent.png", img, source_chunk=seg[i], call_seq=i, obs_index=50, grasp=grasp[i], run=str(ag),
         n_removed_voxels=int(len(rem)), n_kept_voxels=int(len(kep)),
         content="orange = depth pixels within 0.9 voxel of a voxel removed from the target-free field; blue = within 0.9 "
                 "voxel of a non-target voxel kept next to it; robot pixels (inside its spheres + 6 mm) excluded")

    # ---------------------------------------------------------------- ours: permission (obs 90)
    i = obs_of.index(90); z = chunk(i); P, valid = lift(90)
    names = list(json.loads(str(z["summary_json"])).get("sphere_link_names") or [])
    S = np.c_[np.asarray(z["sphere_centres"]), np.asarray(z["sphere_radii"])]
    hand = np.array([n in fs.HAND_LINKS for n in names])
    att = np.asarray(z["attached_query_spheres"], np.float64).reshape(-1, 4)
    img = big(cam(A["ag3s"][90]))
    img = paint(img, up_mask(in_spheres(P, valid, att, pad=0.004)), PURPLE, alpha=0.65, edge_w=2)
    img = paint(img, sphere_mask(S[hand]), GOLD, alpha=0.40, edge_w=3)
    save("r_permission.png", img, source_chunk=seg[i], call_seq=i, obs_index=90, grasp=grasp[i], run=str(ag),
         n_attached_spheres=int(len(att)), n_hand_spheres=int(hand.sum()),
         content="gold = projected union of hand / finger collision spheres (contact links); purple = depth pixels inside "
                 "the recorded attached spheres (+4 mm)")

    # ---------------------------------------------------------------- prior: AEGIS detector boxes (same episode)
    rows = [json.loads(l) for l in (RUNS["aegis"] / "method_calls/calls.jsonl").read_text().splitlines()]
    rows = [d for d in rows if d["task_id"] == TASK and d["episode_idx"] == EP]
    boxes = rows[0]["filter_perception"]["detector"]["boxes"]["head"]
    assert all(r["filter_perception"]["detector"]["boxes"]["head"] == boxes for r in rows), "boxes change over the episode"
    used = [b for b in boxes if b.get("name")]

    def draw_boxes(img):
        dr = ImageDraw.Draw(img)
        for b in used:
            x0, y0, x1, y1 = [c * UP for c in b["box"]]
            for (p, q) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
                n = int(max(abs(q[0] - p[0]), abs(q[1] - p[1])) // 14) + 1
                for k in range(n):
                    if k % 2:
                        continue
                    t0, t1 = k / n, min((k + 1) / n, 1)
                    dr.line([(p[0] + (q[0] - p[0]) * t0, p[1] + (q[1] - p[1]) * t0),
                             (p[0] + (q[0] - p[0]) * t1, p[1] + (q[1] - p[1]) * t1)], fill=(255, 255, 255), width=5)
                    dr.line([(p[0] + (q[0] - p[0]) * t0, p[1] + (q[1] - p[1]) * t0),
                             (p[0] + (q[0] - p[0]) * t1, p[1] + (q[1] - p[1]) * t1)], fill=(60, 60, 60), width=3)
        return img
    box_meta = dict(boxes_used=used, boxes_all=boxes, variant=rows[0]["variant"],
                    n_intervened_calls=int(sum(bool(r.get("call_intervened")) for r in rows)), n_calls=len(rows),
                    note="detector boxes are computed once and reused for every call of the episode (asserted)")
    save("l_main.png", draw_boxes(big(cam(A["aegis"][110]))), run=str(RUNS["aegis"]), obs_index=110, **box_meta)
    save("l_boxes.png", draw_boxes(big(cam(A["aegis"][20]))), run=str(RUNS["aegis"]), obs_index=20, **box_meta)

    # ---------------------------------------------------------------- prior: KNOWS end-effector ellipsoid (obs 110)
    ek = np.load(RUNS["knows"] / episode(RUNS["knows"])["npz"])
    pw, q = np.asarray(ek["obs_robot0_eef_pos"][110], np.float64), np.asarray(ek["obs_robot0_eef_quat"][110], np.float64)
    x, y, zq, wq = q                                                             # robosuite quaternion order (x, y, z, w)
    R = np.array([[1 - 2 * (y * y + zq * zq), 2 * (x * y - zq * wq), 2 * (x * zq + y * wq)],
                  [2 * (x * y + zq * wq), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - x * wq)],
                  [2 * (x * zq - y * wq), 2 * (y * zq + x * wq), 1 - 2 * (x * x + y * y)]])
    Twb_inv = np.linalg.inv(T_WB)
    cb = Twb_inv[:3, :3] @ pw + Twb_inv[:3, 3]; Rb = Twb_inv[:3, :3] @ R
    g = np.random.default_rng(0).normal(size=(6000, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    surf = cb + (g * np.asarray(KNOWS_EE_SEMI)) @ Rb.T
    uv, zz = project(surf)
    from scipy.spatial import ConvexHull
    pts = (uv[zz > 0] + 0.5) * UP
    hull = pts[ConvexHull(pts).vertices]
    img = big(cam(A["knows"][110])).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    dr.polygon([tuple(p) for p in hull], fill=RED + (70,), outline=RED + (255,), width=4)
    save("l_ee.png", Image.alpha_composite(img, ov).convert("RGB"), run=str(RUNS["knows"]), obs_index=110,
         eef_pos_world=pw.round(4).tolist(), eef_quat_xyzw=q.round(4).tolist(), semi_axes_m=list(KNOWS_EE_SEMI),
         content="KNOWS robot model: one ellipsoid on the end effector (gripper frame, robosuite eef pose of the KNOWS run "
                 "at obs 110), projected with the shared head camera; semi-axes from knows/adapter.py GRIPPER_SEMI_AXES")

    # ---------------------------------------------------------------- outcome frames (no overlay)
    for k in ("aegis", "ag3s"):                                    # last recorded frame of each run
        s = len(A[k]) - 1
        save(f"o_{k}_final.png", big(cam(A[k][s])), run=str(RUNS[k]), obs_index=s)

    eps = {k: episode(RUNS[k]) for k in RUNS}
    rec["_episode"] = {"stem": STEM, "chunk_mapping": how, "camera": "head (agentview), static; K / T_base_cam from AG3S geom",
                       "arms": {k: dict(run=str(RUNS[k]), seed=e["seed"], init=e["init_state_index"],
                                        success=e["official_success"], safelibero=e.get("safelibero"))
                                for k, e in eps.items()}}
    (OUT / "assets.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
