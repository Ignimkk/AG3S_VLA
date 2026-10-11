"""T30 pre-E3 check on the first chunks of the first E3a run (server 8204, chunk k <-> local planning seq k+1).
For each chunk: rebuild the TO scene with the T24 tool (same as off1/f1_counterfactual.py) and compare the RECORDED
to.metrics.clearance_m / worst row with the reconstruction under the new rule (target_free_exempt=None) and the old rule
(target_free_exempt=zeros). Also: rows of authorized (mask) spheres whose old/new clearance differ (= would have received the phase
margin) and the recorded settle_epsilon, finger source, stamp mode / camera skew (local observation rows).
usage: check_first_chunks.py COND PORT RUN N"""
from common.workspace import workspace_path as _ag3s_workspace_path
import dataclasses, json, pathlib, sys, ast
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T30/off1'))
import AG3S.trajopt.experiments.fixed_input_ablation as T
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap

COND, PORT, RUN, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
REC = pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T30/{COND}_rec")
run_dir = REC / RUN
from AG3S.trajopt.limits import build_limits
from AG3S.trajopt.types import ChunkLayout
args = T.build_parser().parse_args(["--constraints", str(REC / f"server_{PORT}/server_constraints/run_0000"), "--frames", str(run_dir / "frames.jsonl"),
                                    "--server-log", str(REC / f"serve_safe_{PORT}.log"), "--out", "/dev/null"])
manifest = json.loads((run_dir / "manifest.json").read_text())
settings = T.resolve_settings(args, T.parse_server_log(args.server_log), manifest)
robot, qadr, _ = T.build_robot(settings)
layout = ChunkLayout.rby1(robot.joint_names)
_, _, _, maxd, unk = T.contact_policy(settings, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
planning = T.to_config(settings, collision=True, cost=dict(settings.cost), sqp={})
meter = T.Meter(robot, layout, planning.horizon.planned, settings.hand, build_limits(robot, layout, dt=planning.horizon.dt, config=planning.limits))
lin = meter.lin; fmap = FingerJointMap(robot)
rows = [json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines() if l.strip()]
plan = {r["seq"]: r for r in rows if r["kind"] == "planning"}
obs = [r for r in rows if r["kind"] == "observation"]
out = {"settings": settings.to_json(), "chunks": [],
       "observations": [{"t": o["t_step"], "stamp_mode": o.get("stamp_mode"), "camera_skew_sec": o.get("camera_skew_sec"), "render_spread_sec": o.get("render_spread_sec")} for o in obs[:N]]}
for k in range(N):
    f = REC / f"server_{PORT}/server_constraints/run_0000/chunk_{k:05d}.npz"
    z = np.load(f); s = json.loads(str(z["summary_json"]))
    pr = plan[k + 1]
    q_now = T.q_now_from_qpos(pr["qpos"], qadr)
    field = T.build_field(z, outside_distance=maxd if unk == "free" else -maxd)
    tp = z["target_points"] if "target_points" in z.files else None
    margin, mask, auth, _, _ = T.contact_policy(settings, robot.sphere_link_names, has_target=bool(s.get("has_target")) and tp is not None,
                                                has_free=bool(field.target_free_layers), attached=False)
    new = T.make_scene(field, lin.robot_radii, settings.esdf_margin, target_points=tp, margin=margin, mask=mask)
    old = dataclasses.replace(new, target_free_exempt=np.zeros(lin.robot_radii.size, bool))
    refc = np.asarray(z["reference_chunk"], float)
    Pp, _ = fmap.path((s.get("finger_joints") or {}).get("opening_norm") or {}, {"left": refc[:, 7], "right": refc[:, 15]}, lin.horizon, default=robot.param_vector())
    lin.set_joint_parameter_path(Pp)
    traj = layout.chunk_to_trajectory(np.asarray(z["refined_chunk"], float))[:, :lin.horizon]
    cn, pn = lin.worst_row(traj, q_now, new); co, po = lin.worst_row(traj, q_now, old)
    cln = lin.esdf_clearance(traj, q_now, new)[..., 0]; clo = lin.esdf_clearance(traj, q_now, old)[..., 0]
    maskv = np.zeros(cln.shape[1], bool) if mask is None else mask
    diff = np.abs(cln - clo) > 1e-9
    tm = (s.get("to") or {}).get("metrics") or {}
    rp = tm.get("max_violation_pair"); rp = ast.literal_eval(rp) if isinstance(rp, str) else rp
    g = s.get("grasp") or {}
    out["chunks"].append({"chunk": k, "t": int(z["t_step"]), "recorded_clearance_mm": 1000 * tm["clearance_m"], "recorded_row": [rp.get("step"), rp.get("query"), rp.get("tier")],
                          "new_rule_clearance_mm": 1000 * cn, "new_row": [pn["step"], pn["query"], pn["tier"]], "old_rule_clearance_mm": 1000 * co, "old_row": [po["step"], po["query"], po["tier"]],
                          "abs_diff_recorded_vs_new_mm": abs(1000 * (tm["clearance_m"] - cn)), "abs_diff_recorded_vs_old_mm": abs(1000 * (tm["clearance_m"] - co)),
                          "n_mask_rows_where_old_new_differ": int((diff & maskv[None, :]).sum()), "max_old_new_diff_mask_rows_mm": float(1000 * np.abs(cln - clo)[:, maskv].max()) if maskv.any() else None,
                          "authorized_links": auth, "mask_links": sorted({str(robot.sphere_link_names[i]) for i in np.flatnonzero(maskv)}),
                          "finger_source": (s.get("finger_joints") or {}).get("source"), "settle_epsilon": (g.get("evidence") or {}).get("settle_epsilon"),
                          "anchor_present": bool((s.get("manipulated") or {}).get("anchor")), "association": ((s.get("manipulated") or {}).get("association") or {}).get("outcome")})
    lin.set_joint_parameter_path(None)
    print(json.dumps(out["chunks"][-1]))
print(json.dumps(out["observations"]))
pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T30/precheck/first_chunks_{COND}_{PORT}_{RUN}.json").write_text(json.dumps(out, indent=1, default=float))
