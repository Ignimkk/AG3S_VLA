"""T43 P2: render the chosen overbar per episode at t=0 (recorded planning row 0 of E0 = state after the 1000-step warmup) with the
written config (configs/t43p2_*.json, red rgba) activated by manager.activate (the client call), plus the recorded apple-centre
paths over [lift_t, place_t] (E0 CATEGORICAL[0], E3b CATEGORICAL[1]).  Views: front preset of pi05_infer (lookat 0.45 0 0.85,
distance 1.7, azimuth 180, elevation -18) and an oblique close view of the bar, the crate and the reset gripper.
usage: render_p2.py TAG...  -> render/<TAG>_{front,oblique}.png"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco
from PIL import Image
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P2/p')); sys.path.insert(0, _ag3s_workspace_path(''))
import p2lib as F
from AG3S.ag3s.experiments.common import figstyle
OUT = F.ROOT / "render"; OUT.mkdir(parents=True, exist_ok=True)


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
    s = json.loads((F.RAW / f"select_p2_{tag}.json").read_text())
    if not s.get("config"):
        print(tag, "no config"); continue
    c = F.Ctx(tag); e = s["chosen"]
    paths = {}
    for cond, R in c.runs.items():
        mk = (R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"]); paths[cond] = R["Q"][mk][:, 38:41]
    c.mgr.config = F.PPO.load_pick_place_obstacle_config(s["config"])
    c.d.qpos[:] = c.runs["E0"]["Q"][0]; c.mgr.activate("t43p2_overbar"); mujoco.mj_forward(c.m, c.d)
    c.m.vis.global_.offwidth = max(c.m.vis.global_.offwidth, 960); c.m.vis.global_.offheight = max(c.m.vis.global_.offheight, 720)
    r = mujoco.Renderer(c.m, height=720, width=960)
    side = 1.0 if s["side"] > 0 else -1.0
    for name, (la, dist, az, el) in {"front": ([0.45, 0.0, 0.85], 1.7, 180.0, -18.0),
                                     "oblique": ([e["x"], e["y"] + side * 0.05, 0.97], 0.80, 180.0 + side * 40.0, -28.0)}.items():
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = la; cam.distance = dist; cam.azimuth = az; cam.elevation = el
        r.update_scene(c.d, camera=cam)
        for k, cond in enumerate(("E0", "E3b")):
            add_path(r.scene, paths[cond], rgba(figstyle.CATEGORICAL[k]))
        Image.fromarray(r.render()).save(OUT / f"{tag}_{name}.png")
    r.close(); c.mgr.park_all(); c.restore(); print(tag, "ok", flush=True)
