"""T58 adjacent-post failure scene for CVPR v1 Figure 7 (AG3S ON, ep1830, first robot-post contact step).

Reconstruction follows the verifier's diag_cover_t58.py exactly: the scene is the captured MuJoCo model of the episode
(verify/T58/raw/capture/1830.mjb + reset state) with post_0 placed by the snapshot's PickPlaceObstacleManager from
configs/t58_ep1830.json (profile t58_ep1830_pick); the robot pose is the recorded qpos of control row t = 55 (first
MuJoCo robot-post contact row of ON_ep1830_s18301); the left-hand constraint query spheres (ee_left, ee_finger_l1/l2;
135 of 270) come from the server chunk npz chunk_00016 (planning row t_plan = 48) and are carried rigidly with body
EE_BODY_L to t = 55. The red point is the real closest hand point to the post (mj_geomDistance from-to point, geom of
EE_BODY_L); the red segment goes to the nearest point of the sphere union. The recomputed distances are asserted
against verify/T58/diag/cover_ON_ep1830.json before anything is written.

    MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_fig7_scene.py
"""
from __future__ import annotations

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
RUN = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261013T161809+0900_cf38c9dbca7b_6ee6157e1d6c")
SNAP, V = RUN / "snapshot", RUN / "verify/T58"
sys.path.insert(0, str(SNAP / "integration/rby1/manipulation/src"))
from rby1_manipulation.simulation import pick_place_obstacles as PPO  # noqa: E402

OUT = REPO / "paper/figures/cvpr_v1/assets"
EP, SEED, ARM, T_CONTACT, CHUNK, T_PLAN = 1830, 18301, "ON", 55, 16, 48
RD = V / f"runs/obs/{ARM}_ep{EP}_s{SEED}"
SRV = V / "runs/server_ON_8295_20261013T074151/server_constraints/run_0000" / f"chunk_{CHUNK:05d}.npz"
CFG, PROFILE = V / "configs/t58_ep1830.json", "t58_ep1830_pick"
COVER = V / "diag/cover_ON_ep1830.json"
CAM = {"az": 330, "el": 3, "dist": 0.33}
GOLD = (0.95, 0.70, 0.02, 0.30)
RED = (0.85, 0.05, 0.05, 1.0)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    assert PPO.__file__.startswith(str(SNAP))
    cap = V / "raw/capture"
    m = mujoco.MjModel.from_binary_path(str(cap / f"{EP}.mjb")); d = mujoco.MjData(m)
    z = np.load(cap / f"{EP}_reset.npz")
    d.qpos[:] = z["qpos"]; d.qvel[:] = z["qvel"]; d.mocap_pos[:] = z["mocap_pos"]; d.mocap_quat[:] = z["mocap_quat"]; d.ctrl[:] = z["ctrl"]
    mujoco.mj_forward(m, d)
    cfg = PPO.load_pick_place_obstacle_config(str(CFG))
    mgr = PPO.PickPlaceObstacleManager(m, d, cfg); mgr.activate(PROFILE)
    OB = sorted(mgr.collision_geom_ids); assert len(OB) == 1
    OB = OB[0]
    ctrl = {r["t_step"]: r for r in (json.loads(l) for l in (RD / "frames.jsonl").read_text().splitlines() if l.strip()) if r["kind"] == "control"}
    zc = np.load(SRV, allow_pickle=True)
    sj = json.loads(str(zc["summary_json"]))
    assert int(sj["t_step"]) == CHUNK - 9 and 8 * (int(sj["t_step"]) - 1) == T_PLAN, sj["t_step"]
    names = np.array(sj["sphere_link_names"])
    sel = np.array([n in ("ee_left", "ee_finger_l1", "ee_finger_l2") for n in names])
    C, R = zc["sphere_centres"][sel], zc["sphere_radii"][sel]
    eb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "EE_BODY_L")
    hand = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) in ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2")]

    def setq(t):
        d.qpos[:] = ctrl[t]["qpos"]; mgr.activate(PROFILE); mujoco.mj_kinematics(m, d)

    setq(T_PLAN); loc = (C - d.xpos[eb]) @ d.xmat[eb].reshape(3, 3)
    setq(T_CONTACT); W = loc @ d.xmat[eb].reshape(3, 3).T + d.xpos[eb]
    best = None
    for g in hand:
        ft = np.zeros(6); dist = mujoco.mj_geomDistance(m, d, g, OB, 0.2, ft)
        if best is None or dist < best[0]:
            best = (dist, g, ft.copy())
    dist, g, ft = best; p = ft[:3]
    gap = np.linalg.norm(W - p, axis=1) - R; k = int(np.argmin(gap))
    q = W[k] + R[k] * (p - W[k]) / np.linalg.norm(p - W[k])
    ref = [r for r in json.loads(COVER.read_text())["rows"] if r["t"] == T_CONTACT][0]
    assert abs(1000 * dist - ref["real_hand_post_mm"]) < 0.05 and abs(1000 * gap[k] - ref["closest_point_outside_spheres_mm"]) < 0.05, (dist, gap[k], ref)
    assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) == ref["real_body"]
    # render
    m.vis.global_.offwidth, m.vis.global_.offheight = 1400, 1400
    r = mujoco.Renderer(m, 1400, 1400, max_geom=20000)
    cam = mujoco.MjvCamera(); cam.lookat[:] = p + np.array([0, 0, 0.01]); cam.distance, cam.azimuth, cam.elevation = CAM["dist"], CAM["az"], CAM["el"]
    r.update_scene(d, cam); sc = r.scene
    for c_, rad in zip(W, R):
        gg = sc.geoms[sc.ngeom]
        mujoco.mjv_initGeom(gg, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0, 0]), c_, np.eye(3).ravel(), np.array(GOLD, np.float32)); sc.ngeom += 1
    gg = sc.geoms[sc.ngeom]
    mujoco.mjv_initGeom(gg, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(RED, np.float32))
    mujoco.mjv_connector(gg, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025, p, q); sc.ngeom += 1
    gg = sc.geoms[sc.ngeom]
    mujoco.mjv_initGeom(gg, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.007, 0, 0]), p, np.eye(3).ravel(), np.array(RED, np.float32)); sc.ngeom += 1
    img = r.render().copy()
    # project p and q for vector annotation (pixel coords)
    def project(x):
        fovy = np.deg2rad(m.vis.global_.fovy); cz = np.deg2rad(cam.azimuth); ce = np.deg2rad(cam.elevation)
        fwd = np.array([np.cos(ce) * np.cos(cz), np.cos(ce) * np.sin(cz), np.sin(ce)])
        eye = np.asarray(cam.lookat) - cam.distance * fwd
        up0 = np.array([0, 0, 1.0]); right = np.cross(fwd, up0); right /= np.linalg.norm(right); up = np.cross(right, fwd)
        v = np.asarray(x) - eye; zc_ = v @ fwd; xc = v @ right; yc = v @ up
        f_ = 0.5 / np.tan(fovy / 2)
        return float(0.5 + f_ * xc / zc_) * img.shape[1], float(0.5 - f_ * yc / zc_) * img.shape[0]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "t58_ep1830_contact.png"
    Image.fromarray(img).save(out)
    meta = dict(asset=str(out.relative_to(REPO)), run=str(RD), episode=EP, seed=SEED, arm=ARM, control_step=T_CONTACT, chunk_file=CHUNK, t_plan=T_PLAN,
                post_geom=int(OB), n_hand_spheres=int(sel.sum()), real_hand_post_mm=1000 * float(dist), closest_point_outside_spheres_mm=1000 * float(gap[k]),
                nearest_sphere_link=str(names[sel][k]), real_body=ref["real_body"], hand_point_xyz=p.tolist(), sphere_point_xyz=q.tolist(),
                hand_point_px=project(p), sphere_point_px=project(q), camera=CAM, image_px=[img.shape[1], img.shape[0]],
                cover_rows=json.loads(COVER.read_text())["rows"],
                sources={str(x): sha(x) for x in (RD / "frames.jsonl", SRV, CFG, COVER, cap / f"{EP}.mjb", cap / f"{EP}_reset.npz")},
                renderer=f"MuJoCo {mujoco.__version__} replay of the recorded qpos (no re-simulation); OSMesa, CPU",
                method="verify/T58/p/diag_cover_t58.py reconstruction (spheres carried rigidly with EE_BODY_L from t_plan to t)",
                script="paper/tools/plotting/cvpr_v1_fig7_scene.py")
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print("wrote", out.relative_to(REPO), f"hand-post {1000 * dist:.2f} mm, outside {1000 * gap[k]:.2f} mm, px {meta['hand_point_px']}")


if __name__ == "__main__":
    main()
