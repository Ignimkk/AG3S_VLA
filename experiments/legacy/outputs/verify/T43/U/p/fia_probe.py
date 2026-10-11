"""T43 U — fidelity probe of the TO-row rebuild (FIA helpers, main tree) on one chunk."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
from AG3S.trajopt.experiments import fixed_input_ablation as FIA
from AG3S.trajopt.types import ChunkLayout
from AG3S.trajopt.linearize import CollisionLinearizer
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap

class _Args:
    links = None; sphere_spacing = None; max_spheres_per_capsule = None; capsule_radius_scale = None
    max_sphere_radius = None; esdf_margin = None; target_field_policy = None; phase = None
    manipulators = None; model_xml = None; hand = None; allow_uncertified = False; rows_per_step = None

rd, npz, seq = sys.argv[1], sys.argv[2], int(sys.argv[3])
srv = pathlib.Path(npz).parents[2]
log = FIA.parse_server_log(str(srv / "serve_safe.log"))
print({k: v for k, v in log.items() if k not in ("cost",)})
man = json.loads((pathlib.Path(rd) / "manifest.json").read_text())
man["scene"]["model_xml"] = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
st = FIA.resolve_settings(_Args, log, man)
print(st.links, st.sphere_options, st.esdf_margin, st.target_field_policy, st.phase, st.manipulators)
robot, qadr, _ = FIA.build_robot(st)
layout = ChunkLayout.rby1(robot.joint_names)
lin = CollisionLinearizer(robot, layout, 8)
rows = [json.loads(l) for l in open(pathlib.Path(rd) / "frames.jsonl") if '"planning"' in l[:40]]
plan = {r["seq"]: r for r in rows if r.get("kind") == "planning"}
z = np.load(npz)
s = json.loads(str(z["summary_json"]))
qn = FIA.q_now_from_qpos(np.asarray(plan[seq]["qpos"], float), qadr)
refc = np.asarray(z["refined_chunk"], float)
fmap = FingerJointMap(robot)
meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
path, _ = fmap.path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, 8, default=robot.param_vector())
lin.set_joint_parameter_path(path)
traj = layout.chunk_to_trajectory(refc)[:, :8]
field = FIA.build_field(z, 0.5)
plain = FIA.make_scene(field, lin.robot_radii, st.esdf_margin)
cent = lin.sphere_states(traj, qn)[0]
print("n_spheres", lin.n_spheres, "npz", z["sphere_radii"].shape, "radii diff", np.abs(lin.robot_radii[:122] - z["sphere_radii"][:122]).max())
print("centre step0 diff mm", 1e3 * np.abs(cent[0, :122] - z["sphere_centres"][:122]).max())
mine = lin.esdf_clearance(traj, qn, plain)[..., 0]
rec = z["clearance"]
print("record clearance diff mm", 1e3 * np.abs(mine[:, :rec.shape[1]] - rec).max(), rec.shape)
