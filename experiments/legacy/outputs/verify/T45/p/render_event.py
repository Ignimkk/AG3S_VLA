"""MuJoCo render (osmesa) of palm-contact event frames: recorded control-row qpos, free camera aimed at the contact point."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, numpy as np, mujoco, pathlib
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T45/p'))
import common as C
ev = json.load(open(_ag3s_workspace_path('outputs/verify/T45/events.json')))
picks = [("C1a_ep54_s542", 156), ("C1a_ep3_s31", 233)]
r = mujoco.Renderer(C.M, 480, 640)
out = {}
for name, t0 in picks:
    e = [x for x in ev if x["name"] == name and x["t_first"] == t0][0]
    p = max(e["points"], key=lambda q: -q["dist_mm"])
    rows = {x["t_step"]: x for x in (json.loads(l) for l in open(pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL/runs')) / e["arm"] / "R" / name / "frames.jsonl")) if x["kind"] == "control"}
    C.forward(rows[p["t"]]["qpos"])
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = p["point_world"]; cam.distance = 0.32; cam.azimuth = 200; cam.elevation = -25
    opt = mujoco.MjvOption()
    r.update_scene(C.D, cam, opt); img = r.render()
    np.save(f"${AG3S_PROJECT_ROOT}/outputs/verify/T45/render_{name}_t{p['t']}.npy", img)
    out[f"{name}_t{p['t']}"] = {"lookat": p["point_world"], "geom": p["mesh"], "body": p["mjcf_body"], "fruit_geom": p["fruit_geom"], "dist_mm": p["dist_mm"]}
    print(name, p["t"], img.shape)
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T45/render_meta.json'), "w"), indent=1)
