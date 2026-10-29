"""T42-1 per-run analysis (verifier, 2026-10-05) — T28 analyze_e0.py rule generalized to any target fruit / arm. CPU only.

Target fruit and arm: manifest scene.episode_target_fruit / episode_used_arm, cross-checked with the dataset record
(meta/randomized_episodes.jsonl). qpos addresses are resolved by NAME (address_check.json, written by address_check.py from
the snapshot client's OBJECT_JOINTS / CRATE_JOINT / GRIPPER_{L,R}_JOINT; apple 38, banana 45, orange 52, pear 59, crate 31,
grip left 27, right 18).
  grasp  : target z (planning-row object_poses) >= initial target z + 0.050 m; t = first planning t_step.
  place  : at the LAST planning row the target centre is inside the crate interior, crate-local frame (same rule as
           rby1_manipulation/evaluation/transport.py:object_in_crate centre test: |x| < 0.082-0.005, |y| < 0.142-0.005,
           -0.052 < z < 0.075). place t = first planning t_step of the final uninterrupted in-crate streak.
  success: grasp and place.
Also: non-target fruits in the crate at the last planning row, non-target max xy displacement from their planning-row-0
pose (m), target max dz (control rows, qpos), the arm's applied gripper (applied_ctrl gripper[0 left | 1 right]),
first applied gripper < 0.5 t, HOLD chunks (planning rows whose control rows were not all executed), gate actions,
hold kinds, ipc counts, round trip timing_ms.policy_infer (first chunk excluded).
carry path (for figures): target centre xyz per control row from qpos[adr:adr+3].
usage: analyze_t42.py OUT_JSON RUN_DIR [RUN_DIR ...]   (or import: analyze(run_dir))"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np

HALF = np.array([0.082, 0.142, 0.056]); MARGIN = 0.005; ZR = (-0.052, 0.075)
ADDR = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/1/address_check.json')).read_text())
QADR = ADDR["qposadr"]
DS = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1/meta/randomized_episodes.jsonl'))
_META = {}


def meta(ep):
    if not _META:
        for l in DS.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                _META[r["episode_index"]] = r   # every split (T42 user instruction 2026-10-05: any episode index)
    return _META[ep]


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def in_crate(pos, crate_pos, crate_quat):
    local = quat_to_R(np.asarray(crate_quat)).T @ (np.asarray(pos) - np.asarray(crate_pos))
    lim = HALF[:2] - MARGIN
    return bool(np.all(np.abs(local[:2]) < lim) and ZR[0] < local[2] < ZR[1]), local.tolist()


def analyze(run_dir, series=False):
    run_dir = pathlib.Path(run_dir)
    rows = [json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines() if l.strip()]
    man = json.loads((run_dir / "manifest.json").read_text())
    sc = man["scene"]; ep = sc["episode_index"]; rec = meta(ep)
    tf, arm = sc.get("episode_target_fruit"), sc.get("episode_used_arm")
    assert tf == rec["target_fruit"] and arm == rec["used_arm"], (run_dir, tf, arm, rec["target_fruit"], rec["used_arm"])
    plan = [r for r in rows if r["kind"] == "planning"]; ctrl = [r for r in rows if r["kind"] == "control"]
    others = [k for k in ("apple", "banana", "orange", "pear") if k != tf]
    z0 = plan[0]["object_poses"][tf]["pos"][2]
    p0 = {k: np.asarray(plan[0]["object_poses"][k]["pos"]) for k in others}
    tl = []
    for r in plan:
        op = r["object_poses"]
        ic, local = in_crate(op[tf]["pos"], op["crate"]["pos"], op["crate"]["quat"])
        tl.append({"t": r["t_step"], "seq": r["seq"], "dz_mm": 1000 * (op[tf]["pos"][2] - z0), "in_crate": ic, "local": local})
    grasp_t = next((e["t"] for e in tl if e["dz_mm"] >= 50.0), None)
    place = tl[-1]["in_crate"]; place_t = None
    if place:
        k = len(tl) - 1
        while k > 0 and tl[k - 1]["in_crate"]:
            k -= 1
        place_t = tl[k]["t"]
    opL = plan[-1]["object_poses"]
    others_in = [k for k in others if in_crate(opL[k]["pos"], opL["crate"]["pos"], opL["crate"]["quat"])[0]]
    disp = {k: float(max(np.linalg.norm(np.asarray(r["object_poses"][k]["pos"][:2]) - p0[k][:2]) for r in plan)) for k in others}
    a = QADR[tf]; gi = 0 if arm == "left" else 1; ga = QADR["grip_" + arm]
    Q = np.asarray([c["qpos"] for c in ctrl], float)
    tz = Q[:, a + 2] - z0 if len(Q) else np.zeros(0)
    # self-check: planning-row qpos[adr:adr+3] vs object_poses (body xpos) of the target
    qd = max(float(np.abs(np.asarray(r["qpos"][a:a + 3]) - np.asarray(r["object_poses"][tf]["pos"])).max()) for r in plan if r.get("qpos"))
    gapp = [((c.get("applied_ctrl") or {}).get("gripper") or [None, None])[gi] for c in ctrl]
    first_close = next((c["t_step"] for c, g in zip(ctrl, gapp) if g is not None and g < 0.5), None)
    hold = 0; gate = {}; kinds = {}; ipc = {}
    for p in plan:
        cs = [c for c in ctrl if c["chunk_seq"] == p["seq"]]
        if cs and not all(c.get("executed", True) for c in cs):
            hold += 1
        g = p.get("gate") or {}
        gate[str(g.get("action"))] = gate.get(str(g.get("action")), 0) + 1
        for k in g.get("hold_kinds") or []:
            kinds[k] = kinds.get(k, 0) + 1
        ipc[str(p.get("ipc"))] = ipc.get(str(p.get("ipc")), 0) + 1
    inf = np.array([(p.get("timing_ms") or {}).get("policy_infer") for p in plan[1:] if (p.get("timing_ms") or {}).get("policy_infer") is not None], float)
    out = {"run_dir": str(run_dir), "episode": ep, "policy_seed": man.get("policy_seed"), "target_fruit": tf, "used_arm": arm,
           "safe_manipulators": (man.get("policy") or {}).get("safe_manipulators"), "safe_remote": (man.get("policy") or {}).get("safe_remote"),
           "n_planning": len(plan), "n_control": len(ctrl), "target_z0_m": z0,
           "success": bool(grasp_t is not None and place), "grasp": grasp_t is not None, "place": bool(place),
           "grasp_t": grasp_t, "place_t": place_t, "target_dz_max_mm": float(1000 * tz.max()) if tz.size else None,
           "target_dz_final_mm": tl[-1]["dz_mm"], "target_final_in_crate_frame": tl[-1]["local"],
           "non_target_in_crate_final": others_in, "non_target_max_xy_disp_m": disp,
           "first_arm_gripper_applied_below_0p5_t": first_close,
           "planning_qpos_vs_object_pose_target_max_abs_m": qd,
           "n_hold_chunks": hold, "gate_action_counts": gate, "hold_kind_counts": kinds, "ipc_counts": ipc,
           "policy_infer_ms_excl_first": None if not inf.size else {"median": float(np.median(inf)), "p95": float(np.percentile(inf, 95)), "max": float(inf.max())}}
    if series:
        out["carry_path_xyz"] = Q[:, a:a + 3].tolist() if len(Q) else []
        out["control_t"] = [c["t_step"] for c in ctrl]
        out["grip_meas_open"] = (np.abs(Q[:, ga]) / 0.045).tolist() if len(Q) else []
        out["grip_applied"] = gapp
        out["crate_pose_plan0"] = {"pos": plan[0]["object_poses"]["crate"]["pos"], "quat": plan[0]["object_poses"]["crate"]["quat"]}
        out["object_poses_plan0"] = plan[0]["object_poses"]
    return out


if __name__ == "__main__":
    res = {"definitions": __doc__, "addresses": QADR, "runs": {}}
    for d in sys.argv[2:]:
        r = analyze(d); res["runs"][pathlib.Path(d).name] = r
        print(pathlib.Path(d).name, r["target_fruit"], r["used_arm"], "S" if r["success"] else "-", "grasp_t", r["grasp_t"],
              "place_t", r["place_t"], "hold", r["n_hold_chunks"], "others_in", r["non_target_in_crate_final"])
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
