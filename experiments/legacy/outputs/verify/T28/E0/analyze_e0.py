"""T28 P1-1 — E0 success / grasp / place / apple z timeline from frames.jsonl.

Definitions (written into the output):
  grasp  : apple z (planning-row object_poses) >= initial apple z + 0.050 m; t = first planning t_step.
  place  : at the LAST planning row the apple centre is inside the crate interior, crate-local frame,
           same rule as rby1_manipulation/evaluation/transport.py:object_in_crate
           (|x| < 0.082-0.005, |y| < 0.142-0.005, -0.052 < z < 0.075; CRATE_INTERIOR_HALF=[0.082,0.142,0.056]
           from transport_scene.py:82 == crate wall geoms in scenes/scene_transport.xml:122-131).
           place t = first planning t_step of the final uninterrupted in-crate streak.
  success: grasp and place.
Also: per control step apple z from qpos[38:41] (apple_free qposadr 38), left gripper command
(actions[:,7] of the executed rows, 1=open) and measured left gripper |qpos[27]|/0.045.
"""
import json, sys, pathlib
import numpy as np

HALF = np.array([0.082, 0.142, 0.056]); MARGIN = 0.005; ZR = (-0.052, 0.075)
APPLE_ADR, CRATE_ADR, GRIP_L_ADR = 38, 31, 27


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def in_crate(apple_pos, crate_pos, crate_quat):
    R = quat_to_R(np.asarray(crate_quat))
    local = R.T @ (np.asarray(apple_pos) - np.asarray(crate_pos))
    lim = HALF[:2] - MARGIN
    return bool(np.all(np.abs(local[:2]) < lim) and ZR[0] < local[2] < ZR[1]), local.tolist()


def analyze(run_dir):
    run_dir = pathlib.Path(run_dir)
    rows = [json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines() if l.strip()]
    man = json.loads((run_dir / "manifest.json").read_text())
    plan = [r for r in rows if r["kind"] == "planning"]
    ctrl = [r for r in rows if r["kind"] == "control"]
    z0 = plan[0]["object_poses"]["apple"]["pos"][2]
    timeline = []
    for r in plan:
        op = r["object_poses"]
        ic, local = in_crate(op["apple"]["pos"], op["crate"]["pos"], op["crate"]["quat"])
        timeline.append({"t": r["t_step"], "seq": r["seq"],
                         "apple_pos": op["apple"]["pos"], "apple_dz_mm": 1000 * (op["apple"]["pos"][2] - z0),
                         "in_crate": ic, "apple_in_crate_frame": local,
                         "infer_ms": (r.get("timing_ms") or {}).get("policy_infer"),
                         "ipc": r.get("ipc"), "safe": r.get("safe"), "hold_reason": r.get("hold_reason"),
                         "grip_l_cmd_first8": [a[7] for a in r["actions"][:8]] if r.get("actions") else None})
    grasp_t = next((e["t"] for e in timeline if e["apple_dz_mm"] >= 50.0), None)
    place = timeline[-1]["in_crate"]
    place_t = None
    if place:
        k = len(timeline) - 1
        while k > 0 and timeline[k - 1]["in_crate"]:
            k -= 1
        place_t = timeline[k]["t"]
    # per control step
    cz = [{"t": c["t_step"], "apple_dz_mm": 1000 * (c["qpos"][APPLE_ADR + 2] - z0),
           "grip_l_meas": abs(c["qpos"][GRIP_L_ADR]) / 0.045,
           "grip_l_applied": (c.get("applied_ctrl") or {}).get("gripper", [None])[0],
           "executed": c.get("executed")} for c in ctrl]
    grasp_t_ctrl = next((e["t"] for e in cz if e["apple_dz_mm"] >= 50.0), None)
    first_close_cmd_t = next((e["t"] for e in cz if e["grip_l_applied"] is not None and e["grip_l_applied"] < 0.5), None)
    inf = np.array([e["infer_ms"] for e in timeline if e["infer_ms"] is not None])
    return {
        "run_dir": str(run_dir), "episode_index": man["scene"]["episode_index"],
        "argv": man["extra"].get("argv"), "mujoco": man["versions"].get("mujoco"),
        "n_planning": len(plan), "n_control": len(ctrl),
        "apple_z0_m": z0,
        "success": bool(grasp_t is not None and place), "grasp": grasp_t is not None, "place": place,
        "grasp_t": grasp_t, "grasp_t_control_rows": grasp_t_ctrl, "place_t": place_t,
        "apple_dz_max_mm": max(e["apple_dz_mm"] for e in cz) if cz else None,
        "apple_dz_final_mm": timeline[-1]["apple_dz_mm"],
        "apple_final_in_crate_frame": timeline[-1]["apple_in_crate_frame"],
        "first_left_gripper_applied_below_0p5_t": first_close_cmd_t,
        "infer_ms": {"n": int(inf.size), "first": float(inf[0]) if inf.size else None,
                     "median_excl_first": float(np.median(inf[1:])) if inf.size > 1 else None,
                     "p95_excl_first": float(np.percentile(inf[1:], 95)) if inf.size > 1 else None,
                     "max_excl_first": float(inf[1:].max()) if inf.size > 1 else None},
        "n_ipc_not_ok_or_inprocess": sum(1 for e in timeline if e["ipc"] not in ("ok", "in-process")),
        "n_not_executed_control": sum(1 for c in ctrl if c.get("executed") is False),
        "timeline_planning": timeline, "timeline_control": cz,
    }


def determinism(a, b):
    ra = [json.loads(l) for l in (pathlib.Path(a) / "frames.jsonl").read_text().splitlines() if l.strip()]
    rb = [json.loads(l) for l in (pathlib.Path(b) / "frames.jsonl").read_text().splitlines() if l.strip()]
    pa = [r for r in ra if r["kind"] == "planning"]; pb = [r for r in rb if r["kind"] == "planning"]
    ca = [r for r in ra if r["kind"] == "control"]; cb = [r for r in rb if r["kind"] == "control"]
    # non-safe client rows carry no `actions`; compare the applied ctrl (arm+gripper) per control step
    def ac(c):
        a = c.get("applied_ctrl") or {}
        return np.array(list(a.get("arm", [])) + list(a.get("gripper", [])), dtype=float)
    n = min(len(ca), len(cb))
    act = [float(np.abs(ac(ca[i]) - ac(cb[i])).max()) for i in range(n)]
    m = min(len(ca), len(cb))
    qd = [float(np.abs(np.array(ca[i]["qpos"]) - np.array(cb[i]["qpos"])).max()) for i in range(m)]
    first_act = next((ca[i]["t_step"] for i in range(n) if act[i] > 0), None)
    return {"n_control_steps_compared": n, "max_abs_applied_ctrl_diff_per_step": act,
            "max_abs_applied_ctrl_diff": max(act), "first_t_with_nonzero_applied_ctrl_diff": first_act,
            "step0_applied_ctrl_diff": act[0],
            "n_control_compared": m, "max_abs_qpos_diff": max(qd),
            "first_control_t_with_qpos_diff_gt_1e-9": next((ca[i]["t_step"] for i in range(m) if qd[i] > 1e-9), None)}


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1])
    res = {"definitions": __doc__, "runs": {}}
    for d in sys.argv[2:]:
        r = analyze(d)
        res["runs"][pathlib.Path(d).name] = r
    runs = [pathlib.Path(d) for d in sys.argv[2:]]
    r1807 = [d for d in runs if "ep1807" in d.name]
    if len(r1807) >= 2:
        res["determinism_ep1807"] = determinism(r1807[0], r1807[1])
    out.write_text(json.dumps(res, indent=1))
    for k, r in res["runs"].items():
        print(k, "success", r["success"], "grasp_t", r["grasp_t"], "place_t", r["place_t"],
              "dz_max", round(r["apple_dz_max_mm"], 1), "dz_final", round(r["apple_dz_final_mm"], 1),
              "infer", r["infer_ms"])
    if "determinism_ep1807" in res:
        d = res["determinism_ep1807"]; print({k: v for k, v in d.items() if k != "max_abs_applied_ctrl_diff_per_step"})
