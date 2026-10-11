"""T40 E smoke (implementer): run pi05_infer.main() for episode 1807 with a hurdle config up to the point where the
policy would be contacted (no server, no GPU), then render third-person images of the scene main() built.
usage: smoke_hurdle.py CONFIG PROFILE OUT_DIR"""
from common.workspace import workspace_path as _ag3s_workspace_path
import inspect, json, os, pathlib, sys
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, _ag3s_workspace_path('integration/rby1/bringup'))
import mujoco, numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import pi05_infer as P  # noqa: E402

cfg, prof, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
CAP = {}


class Reached(Exception):
    pass


def stub(remote):
    CAP.update(inspect.currentframe().f_back.f_locals); raise Reached


P.load_remote_policy = stub
sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1",
            "--episode-index", "1807", "--headless", "--obstacle-config", cfg, "--obstacle-profile", prof]
try:
    P.main()
except Reached:
    pass
m, d, mgr = CAP["m"], CAP["d"], CAP["obstacle_manager"]
summary = {"model_xml": CAP["model_xml"], "profile": prof, "active_slots": list(mgr.active_slots),
           "robot_clearance_m": mgr.robot_clearance(), "object_clearance_m": mgr.object_clearance()}
bar = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pick_obstacle_hurdle_0_collision_bar")
summary["bar_centre_world_m"] = d.geom_xpos[bar].tolist()
summary["apple_world_m"] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")].tolist()
summary["crate_world_m"] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")].tolist()
r = mujoco.Renderer(m, height=600, width=800)
views = {"front": dict(lookat=[0.5, 0.1, 0.9], distance=1.25, azimuth=180.0, elevation=-25.0),
         "oblique": dict(lookat=[0.55, 0.15, 0.9], distance=0.95, azimuth=225.0, elevation=-28.0)}
for name, v in views.items():
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = v["lookat"]; cam.distance = v["distance"]; cam.azimuth = v["azimuth"]; cam.elevation = v["elevation"]
    r.update_scene(d, camera=cam); Image.fromarray(r.render()).save(out / f"smoke_ep1807_hurdle_{name}.png")
r.update_scene(d, camera="zed_left"); Image.fromarray(r.render()).save(out / "smoke_ep1807_hurdle_zed_left.png")
(out / "smoke_ep1807_hurdle.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
