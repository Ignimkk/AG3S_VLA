"""T43 S2: grasp-pose stills.  For each episode and run (DEMO replay, first E0 and first E3b policy run with a close command), the
recorded control-row qpos at the settled row (first close command + 16 control rows; analyze_s2.py) is set on the T42-1 client
model (${AG3S_PROJECT_ROOT}-t42 model_transport.xml) and rendered (mj_forward, no stepping) with a close camera looking at the target
fruit from the robot's side (azimuth 180 = from +x toward -x, i.e. the robot's view direction reversed; elevation -30, distance
0.45 m).  usage: render_s2.py EP [EP ...]  -> render/<EP>_<RUN>_settled.png"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco
from PIL import Image
S2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S2')); OUT = S2 / "render"; OUT.mkdir(parents=True, exist_ok=True)
XML = "${AG3S_PROJECT_ROOT}-t42/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
A = json.loads((S2 / "raw/s2_analysis.json").read_text())["episodes"]
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
m.vis.global_.offwidth = max(m.vis.global_.offwidth, 640); m.vis.global_.offheight = max(m.vis.global_.offheight, 480)
r = mujoco.Renderer(m, height=480, width=640)
idx = {}
for ep in sys.argv[1:]:
    e = A[ep]; runs = [("DEMO", e["demo_replay"])]
    for cond in ("E0", "E3b"):
        rr = [x for x in e["policy_runs"] if x["cond"] == cond and x["geom"].get("settled")]
        if rr:
            runs.append((cond + "_" + pathlib.Path(rr[0]["run_dir"]).name.split("_")[-1], rr[0]))
    for name, run in runs:
        g = run["geom"].get("settled")
        if not g:
            continue
        rows = [json.loads(l) for l in (pathlib.Path(run["run_dir"]) / "frames.jsonl").read_text().splitlines() if l.strip()]
        q = next(np.asarray(c["qpos"], float) for c in rows if c["kind"] == "control" and c["t_step"] == g["t"])
        d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        f = np.asarray(g["fruit_pos"])
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = f + np.array([0, 0, 0.03]); cam.distance = 0.45; cam.azimuth = 180.0; cam.elevation = -30.0
        r.update_scene(d, camera=cam)
        p = OUT / f"{ep}_{name}_settled.png"; Image.fromarray(r.render()).save(p)
        idx[f"{ep}/{name}"] = {"image": str(p), "t": g["t"], "run_dir": run["run_dir"]}
        print(ep, name, "ok", flush=True)
(S2 / "render/index.json").write_text(json.dumps(idx, indent=1))
