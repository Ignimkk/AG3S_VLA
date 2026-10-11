"""T43 P: render the chosen placement per episode at t=0 (recorded planning row 0 of E0 = state after the 1000-step warmup; hurdle
via manager.activate = the client call) with the recorded T39 apple-centre paths over [lift_t, place_t] (E0 CATEGORICAL[0],
E3b CATEGORICAL[1]).  Views: front preset of pi05_infer (lookat 0.45 0 0.85, distance 1.7, azimuth 180, elevation -18) and an
oblique close view of the hurdle + reset gripper.  usage: render_p.py TAG...  (RP_KIND=main|relaxed|tradeoff; tradeoff = select tradeoff d_best_given_abc, a placement that does NOT pass all
criteria, for episodes without one)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
import numpy as np, mujoco
from PIL import Image
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p')); sys.path.insert(0, _ag3s_workspace_path(''))
import flib as F
from AG3S.ag3s.experiments.common import figstyle
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P/render')); OUT.mkdir(parents=True, exist_ok=True)
KIND = os.environ.get("RP_KIND", "main")


def rgba(h):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [1.0], np.float32)


def add_path(scn, P, col, w=0.004):
    for a, b in zip(P[:-1], P[1:]):
        if scn.ngeom >= scn.maxgeom or np.linalg.norm(b - a) < 1e-5:
            continue
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), col)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, w, a, b); scn.ngeom += 1


for tag in sys.argv[1:]:
    s = json.loads(pathlib.Path(f"{_ag3s_workspace_path()}/outputs/verify/T43/P/raw/select_p_{tag}.json").read_text())
    e = {"main": s.get("chosen"), "relaxed": s.get("chosen_relaxed"), "tradeoff": (s["tradeoff"].get("d_best_given_abc") or {})}.get(KIND) or None
    if not e:
        print(tag, "no placement", KIND); continue
    c = F.Ctx(tag)
    paths = {}
    for cond, R in c.runs.items():
        mk = (R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"]); paths[cond] = R["Q"][mk][:, 38:41]
    c.set_hurdle(e["x"], e["y"], e["yaw"], e["h"], exact=True)
    c.d.qpos[:] = c.runs["E0"]["Q"][0]; mujoco.mj_forward(c.m, c.d)
    c.m.vis.global_.offwidth = max(c.m.vis.global_.offwidth, 960); c.m.vis.global_.offheight = max(c.m.vis.global_.offheight, 720)
    r = mujoco.Renderer(c.m, height=720, width=960)
    for name, (la, dist, az, el) in {"front": ([0.45, 0.0, 0.85], 1.7, 180.0, -18.0),
                                     "oblique": ([e["x"], e["y"] + 0.03, 0.98], 0.85, 220.0, -22.0)}.items():
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = la; cam.distance = dist; cam.azimuth = az; cam.elevation = el
        r.update_scene(c.d, camera=cam)
        for k, cond in enumerate(("E0", "E3b")):
            add_path(r.scene, paths[cond], rgba(figstyle.CATEGORICAL[k]))
        Image.fromarray(r.render()).save(OUT / f"{tag}_{KIND}_{name}.png")
    r.close(); c.restore(); print(tag, "ok", flush=True)
