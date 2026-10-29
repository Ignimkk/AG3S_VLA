"""hold_drift.json → 스텝별 지표 (손가락 구 중심 드리프트, q_hold 대비 관절 편차, gripper ctrl, T16 기록 대조)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.trajopt.experiments import fixed_input_ablation as fia
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T23/m3'))
RUN = pathlib.Path(_ag3s_workspace_path('outputs/live_test/20260928_t16'))
d = json.loads((OUT / "hold_drift.json").read_text())
args = fia.build_parser().parse_args(["--constraints", str(RUN / "server_constraints/run_0000"), "--frames", str(RUN / "execute_ep1807/frames.jsonl"), "--server-log", str(RUN / "serve_safe.log"), "--out", "/dev/null"])
settings = fia.resolve_settings(args, fia.parse_server_log(args.server_log), json.loads((RUN / "execute_ep1807/manifest.json").read_text()))
robot, qadr, _ = fia.build_robot(settings)
names = [str(x) for x in robot.sphere_link_names]
fi = np.asarray([i for i, x in enumerate(names) if x in ("ee_finger_l1", "ee_finger_l2")])
LEFT = [f"left_arm_{i}" for i in range(7)]; RIGHT = [f"right_arm_{i}" for i in range(7)]


def finger(qpos):
    c, _ = robot.sphere_centers_numeric(fia.q_now_from_qpos(qpos, qadr))
    return np.asarray(c)[fi].mean(axis=0)


def finger_from_arm(qpos_template, arm14):
    q = np.asarray(qpos_template, np.float64).copy()
    for j, v in zip(LEFT + RIGHT, arm14):
        q[qadr[j]] = v
    return finger(q)


def arm_of(qpos):
    return np.asarray([qpos[qadr[j]] for j in LEFT + RIGHT])


res = {"mujoco": d["mujoco"], "runs": {}}
for key, r in d["runs"].items():
    log = r["log"]
    fc = np.asarray([e["finger_centre_base"] for e in log])
    entry = fc[0]
    first_hold = next(e for e in log if e["hold"] is not None)
    qh = np.asarray(first_hold["hold"]["q_hold"])
    fq = finger_from_arm(first_hold["qpos"], qh)
    arms = np.asarray([arm_of(e["qpos"]) for e in log])
    rec = [e["recorded_t16_qpos"] for e in log]
    rec_fc = np.asarray([finger(q) for q in rec if q is not None])
    n = len(rec_fc)
    apple = np.asarray([e["apple_world"] for e in log])
    out = {
        "t": [e["t"] for e in log],
        "executed": [e["executed"] for e in log],
        "finger_drift_from_entry_mm": (np.linalg.norm(fc - entry, axis=1) * 1000).tolist(),
        "finger_dist_to_fk_q_hold_mm": (np.linalg.norm(fc - fq, axis=1) * 1000).tolist(),
        "finger_step_mm": np.r_[0.0, np.linalg.norm(np.diff(fc, axis=0), axis=1) * 1000].tolist(),
        "arm_max_abs_dev_from_q_hold_deg": np.degrees(np.abs(arms - qh[None, :]).max(axis=1)).tolist(),
        "arm_step_max_deg": np.r_[0.0, np.degrees(np.abs(np.diff(arms, axis=0)).max(axis=1))].tolist(),
        "hold_max_abs_dev_rad_record": [None if e["hold"] is None else e["hold"].get("max_abs_dev_rad") for e in log],
        "hold_source": first_hold["hold"]["source"], "q_hold_captured_at_t": first_hold["t"],
        "ctrl_left_grip": [e["ctrl_left_grip"] for e in log],
        "applied_gripper_left": [e["applied_gripper"][0] for e in log],
        "measured_gripper_left": [e["measured_state"][7] for e in log],
        "gripper_hold": first_hold["hold"]["gripper_hold"],
        "apple_world_disp_mm": (np.linalg.norm(apple - apple[0], axis=1) * 1000).tolist(),
        "apple_z_change_mm": ((apple[:, 2] - apple[0, 2]) * 1000).tolist(),
        "finger_centre_base": fc.tolist(),
        "prefix_qpos_max_abs_err_at_t71": r["prefix_qpos_max_abs_err_per_t"][-1],
        "prefix_qpos_max_abs_err_max": max(r["prefix_qpos_max_abs_err_per_t"]),
        "prefix_arm_err_note": "qpos 전체(물체 free joint 포함) max abs 차",
    }
    if n:
        out["t16_recorded_finger_drift_from_t72_mm"] = (np.linalg.norm(rec_fc - rec_fc[0], axis=1) * 1000).tolist()
        out["sim_vs_t16_finger_mm"] = (np.linalg.norm(fc[:n] - rec_fc, axis=1) * 1000).tolist()
        rec_arms = np.asarray([arm_of(q) for q in rec if q is not None])
        out["sim_vs_t16_arm_max_deg"] = np.degrees(np.abs(arms[:n] - rec_arms).max(axis=1)).tolist()
        out["t16_recorded_arm_step_max_deg"] = np.r_[0.0, np.degrees(np.abs(np.diff(rec_arms, axis=0)).max(axis=1))].tolist()
        out["t16_recorded_finger_centre_base"] = rec_fc.tolist()
    res["runs"][key] = out
    k = lambda a, i: round(a[i], 2)
    print(key, "src", out["hold_source"], "drift@0,8,24,63", [k(out["finger_drift_from_entry_mm"], i) for i in (0, 8, 24, 63)],
          "toFKqhold@0,8,63", [k(out["finger_dist_to_fk_q_hold_mm"], i) for i in (0, 8, 63)],
          "armdev_deg@63", k(out["arm_max_abs_dev_from_q_hold_deg"], 63), "step_mm@60-63", np.round(out["finger_step_mm"][60:64], 3).tolist(),
          "armstep_deg mean(8:)", round(float(np.mean(out["arm_step_max_deg"][8:])), 3),
          "grip ctrl uniq", sorted(set(np.round(out["ctrl_left_grip"], 5)))[:4],
          "apple disp max", round(max(out["apple_world_disp_mm"]), 2),
          "vsT16 finger @0,8,20", [k(out.get("sim_vs_t16_finger_mm", [0]*64), i) for i in (0, 8, 20)] if n else None,
          "T16 drift @20", k(out["t16_recorded_finger_drift_from_t72_mm"], 20) if n else None,
          "prefix err t71", out["prefix_qpos_max_abs_err_at_t71"])
(OUT / "hold_drift_analysis.json").write_text(json.dumps(res))
