"""T40 A figure 1 (real scene): t = 0 scene of the episode (E3b planning row 0 qpos) with the obstacle placed via mocap,
and the recorded apple-centre paths of E0 and E3b drawn as 3-D polylines (MuJoCo user geoms) over [lift_t, place_t].
Renderer: MuJoCo osmesa, two cameras: 'front' preset of pi05_infer.py (lookat 0.45 0 0.85, distance 1.7, azimuth 180,
elevation -18) and an oblique top view (lookat at the obstacle, distance 0.9, azimuth 160, elevation -55).
usage: render_scene.py OUT_PNG_PREFIX REPLAY_JSON SLOT  -> OUT_front.png, OUT_top.png (numpy arrays only, no labels)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, sys
import numpy as np, mujoco
import t40a_lib as L

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402


def hex_rgba(h, a=1.0):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a], np.float32)


def add_path(scn, P, rgba, width=0.004):
    for a, b in zip(P[:-1], P[1:]):
        if scn.ngeom >= scn.maxgeom or np.linalg.norm(b - a) < 1e-5:
            continue
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scn.ngeom += 1


def add_sphere(scn, p, r, rgba):
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), p, np.eye(3).ravel(), rgba)
    scn.ngeom += 1


def render(q0, slot, xy, yaw, paths, marks, out_prefix, W=960, H=720):
    L.set_q(q0); L.place(slot, xy, yaw); mujoco.mj_forward(L.m, L.d)
    L.m.vis.global_.offwidth = max(L.m.vis.global_.offwidth, W); L.m.vis.global_.offheight = max(L.m.vis.global_.offheight, H)
    r = mujoco.Renderer(L.m, height=H, width=W)
    cams = {}
    c = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(c); c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = [0.45, 0.0, 0.85]; c.distance = 1.7; c.azimuth = 180.0; c.elevation = -18.0; cams["front"] = c
    c2 = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(c2); c2.type = mujoco.mjtCamera.mjCAMERA_FREE
    c2.lookat[:] = [xy[0] - 0.02, xy[1] - 0.08, L.TABLE_Z + 0.12]; c2.distance = 0.9; c2.azimuth = 160.0; c2.elevation = -55.0
    cams["top"] = c2
    outs = {}
    for name, cam in cams.items():
        r.update_scene(L.d, camera=cam)
        for P, col in paths:
            add_path(r.scene, P, hex_rgba(col, 1.0))
        for p, col in marks:
            add_sphere(r.scene, p, 0.008, hex_rgba(col, 1.0))
        img = r.render()
        outs[name] = img
        from PIL import Image
        Image.fromarray(img).save(f"{out_prefix}_{name}.png")
    r.close()
    return outs
