"""T37 copy of T35diag/a_runs.py (unchanged code; run on T37 E3a/E3b records -> outputs/verify/T37/geom/a/a_runs.json). T35-diag A — per-run grasp kinematics from existing closed-loop records (frames.jsonl only, MuJoCo GT).

usage: a_runs.py OUT_JSON RUN_DIR [RUN_DIR ...]

Per control row (t, chunk_seq, step_in_chunk):
  grip_cmd        : applied_ctrl.gripper[0] (left, 1 = open) — what was sent (refined chunk col 7 for safe runs)
  grip_ref_cmd    : actions_reference[step, 7] of that chunk's planning row (E0 runs: actions = reference)
  open_meas       : |qpos[27]| / 0.045 (measured left finger opening, 1 = open)
  apple           : qpos[38:41] (GT apple centre, world); disp_xy_mm / dz_mm from the first control row
  d_exec_mm       : min mj_geomDistance(left finger geoms (ee_finger_l1, ee_finger_l2), apple_geom) at the measured qpos
  d_l1_mm/d_l2_mm : same, per finger body
  d_ref_mm        : same with the left+right arm joints (left_arm_0..6, right_arm_0..6) replaced by actions_reference[step]
                    (16D cols 0-6 / 8-14), all other qpos (fingers, torso, apple) = measured row  -> the policy's commanded pose
  d_refined_mm    : same with actions[step] (the chunk that was sent)
  E0 records have no chunk arrays (planning actions = None): reference = refined = the control row's applied_ctrl (arm 14, gripper 2)
  mid_rel_apple_mm: midpoint of the two finger-side witness points (closest point of ee_finger_l1 / ee_finger_l2 to the apple)
                    - apple centre, world frame (mm), measured / ref / refined
  body_mid_rel_apple_*: mean of the ee_finger_l1 / ee_finger_l2 BODY origins - apple centre (world mm) for the measured /
                    reference-commanded / refined-commanded arm pose (pose-only, no contact witness)
Per planning row: refined - reference per joint, rows 0-7 (deg, |.| max over rows), arm cols 0-6, 8-14.
Events:
  t_close        : first control row with grip_cmd < 0.5 (analyze_p2r definition)
  t_close_ref    : first control row with grip_ref_cmd < 0.5
  at_close       : the row values at t_close; window stats over [t_close, t_close + 16) control rows
  n_close_cycles : count of grip_cmd crossings >=0.5 -> <0.5
Success / grasp_t / place_t: ../T28/E0/analyze_e0.py:analyze (same definitions as T28-T34).
Distances: GT geometry (MuJoCo 3.11 convex-hull mesh distance; negative = penetration); invalid witness pairs -> +inf, counted.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import mujoco

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/E0'))
import p12_records as P  # noqa
import analyze_e0 as A  # noqa

ARM = list(range(0, 7)) + list(range(8, 15))
m, d, probe, obj, fing = P.build()
qadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): int(m.jnt_qposadr[i]) for i in range(m.njnt)}
ARM_ADR = [qadr[f"left_arm_{i}"] for i in range(7)] + [qadr[f"right_arm_{i}"] for i in range(7)]
F1 = [g for g in fing if m.body(m.geom_bodyid[g]).name == "ee_finger_l1"]
F2 = [g for g in fing if m.body(m.geom_bodyid[g]).name == "ee_finger_l2"]
AG = obj["apple"]
B1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_finger_l1")
B2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_finger_l2")


def dist(gs):
    """min distance (mm) and the finger-side witness point (world) of that pair."""
    best = (np.inf, None)
    for g in gs:
        for a in AG:
            ft = np.zeros(6)
            v = float(mujoco.mj_geomDistance(m, d, g, a, 1.0, ft))
            if v < 1.0 and abs(abs(v) - float(np.linalg.norm(ft[:3] - ft[3:]))) > 1e-3:
                P.N_INVALID[0] += 1
                continue
            if v < best[0]:
                best = (v, ft[:3].copy())
    return 1000 * best[0], best[1]


def measure(qpos, arm_cmd=None):
    q = np.asarray(qpos, float).copy()
    if arm_cmd is not None:
        q[ARM_ADR] = np.asarray(arm_cmd, float)[ARM]
    P.set_q(m, d, q)
    (d1, w1), (d2, w2) = dist(F1), dist(F2)
    mid = 0.5 * (w1 + w2) if (w1 is not None and w2 is not None) else np.full(3, np.nan)
    BODY_MID[0] = (1000 * (0.5 * (d.xpos[B1] + d.xpos[B2]) - q[38:41])).tolist()
    return min(d1, d2), d1, d2, (1000 * (mid - q[38:41])).tolist()


BODY_MID = [None]


def run_one(run_dir):
    run_dir = pathlib.Path(run_dir)
    rows = [json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines() if l.strip()]
    plan = {int(r["seq"]): r for r in rows if r["kind"] == "planning"}
    ctrl = [r for r in rows if r["kind"] == "control"]
    has_ref = "actions_reference" in next(iter(plan.values()))
    has_act = next(iter(plan.values())).get("actions") is not None
    plan_t = {int(r["t_step"]): r for r in plan.values()}
    a0 = np.asarray(ctrl[0]["qpos"][38:41], float)
    tl = []
    for c in ctrl:
        k = int(c["step_in_chunk"])
        pr = plan.get(int(c["chunk_seq"]))
        if pr is None or int(pr["t_step"]) != int(c["t_step"]) - k:
            pr = plan_t[int(c["t_step"]) - k]
        if has_act:
            act = np.asarray(pr["actions"], float)
        else:  # E0 records carry no chunk: the applied command row is the policy row
            act = np.zeros((k + 1, 16))
            ac_ = c["applied_ctrl"]
            act[k, 0:7] = ac_["arm"][0:7]; act[k, 8:15] = ac_["arm"][7:14]
            act[k, 7] = ac_["gripper"][0]; act[k, 15] = ac_["gripper"][1]
        ref = np.asarray(pr["actions_reference"], float) if has_ref else act
        de, d1, d2, mid = measure(c["qpos"]); bm = BODY_MID[0]
        dr, dr1, dr2, midr = measure(c["qpos"], ref[k]); bmr = BODY_MID[0]
        df, df1, df2, midf = measure(c["qpos"], act[k]); bmf = BODY_MID[0]
        ap = np.asarray(c["qpos"][38:41], float)
        tl.append({"t": c["t_step"], "seq": int(c["chunk_seq"]), "k": k, "executed": bool(c.get("executed", True)),
                   "grip_cmd": float(c["applied_ctrl"]["gripper"][0]), "grip_ref_cmd": float(ref[k, 7]),
                   "grip_refined_cmd": float(act[k, 7]),
                   "open_meas": abs(float(c["qpos"][27])) / 0.045,
                   "apple": ap.tolist(), "disp_xy_mm": float(1000 * np.linalg.norm(ap[:2] - a0[:2])), "dz_mm": float(1000 * (ap[2] - a0[2])),
                   "d_exec_mm": de, "d_l1_mm": d1, "d_l2_mm": d2, "d_ref_mm": dr, "d_ref_l1_mm": dr1, "d_ref_l2_mm": dr2,
                   "d_refined_mm": df, "mid_rel_apple_mm": mid, "mid_rel_apple_ref_mm": midr, "mid_rel_apple_refined_mm": midf,
                   "body_mid_rel_apple_mm": bm, "body_mid_rel_apple_ref_mm": bmr, "body_mid_rel_apple_refined_mm": bmf})
    # per planning row refined - reference
    rr = []
    for s, pr in sorted(plan.items()):
        if not has_act:
            break
        act = np.asarray(pr["actions"], float)
        ref = np.asarray(pr["actions_reference"], float) if has_ref else act
        dv = np.degrees(np.abs(act[:8] - ref[:8]))
        rr.append({"seq": s, "t": pr["t_step"], "per_col_deg": dv.max(axis=0).tolist(),
                   "max_arm_deg": float(dv[:, ARM].max()), "max_left_arm_deg": float(dv[:, 0:7].max()),
                   "max_right_arm_deg": float(dv[:, 8:15].max()), "grip_col7_absdiff": float(dv[:, 7].max())})
    ts = [e["t"] for e in tl]
    gc = np.array([e["grip_cmd"] for e in tl]); gr = np.array([e["grip_ref_cmd"] for e in tl])
    ic = next((i for i, g in enumerate(gc) if g < 0.5), None)
    ir = next((i for i, g in enumerate(gr) if g < 0.5), None)
    an = A.analyze(run_dir)
    out = {"run": run_dir.name, "run_dir": str(run_dir), "has_reference": has_ref, "has_actions": has_act,
           "success": an.get("success"), "grasp_t": an.get("grasp_t"), "place_t": an.get("place_t"),
           "apple_dz_max_mm": an.get("apple_dz_max_mm"),
           "t_close": ts[ic] if ic is not None else None, "t_close_ref": ts[ir] if ir is not None else None,
           "n_close_cycles": int(np.sum((gc[1:] < 0.5) & (gc[:-1] >= 0.5))),
           "n_close_cycles_ref": int(np.sum((gr[1:] < 0.5) & (gr[:-1] >= 0.5)))}
    if ic is not None:
        e = tl[ic]
        w = tl[ic:ic + 16]
        pre = tl[:ic]
        out["at_close"] = {k2: e[k2] for k2 in ("t", "seq", "k", "d_exec_mm", "d_l1_mm", "d_l2_mm", "d_ref_mm", "d_ref_l1_mm", "d_ref_l2_mm",
                                                 "d_refined_mm", "mid_rel_apple_mm", "mid_rel_apple_ref_mm", "mid_rel_apple_refined_mm",
                                                 "open_meas", "disp_xy_mm", "dz_mm")}
        out["window16"] = {"min_d_exec_mm": min(x["d_exec_mm"] for x in w), "min_d_ref_mm": min(x["d_ref_mm"] for x in w),
                           "min_d_refined_mm": min(x["d_refined_mm"] for x in w),
                           "max_l1_mm": max(x["d_l1_mm"] for x in w), "max_l2_mm": max(x["d_l2_mm"] for x in w),
                           "min_l1_mm": min(x["d_l1_mm"] for x in w), "min_l2_mm": min(x["d_l2_mm"] for x in w),
                           "min_open_meas": min(x["open_meas"] for x in w), "last_open_meas": w[-1]["open_meas"],
                           "max_disp_xy_mm": max(x["disp_xy_mm"] for x in w), "max_dz_mm": max(x["dz_mm"] for x in w)}
        out["pre_close"] = {"max_disp_xy_mm": max([x["disp_xy_mm"] for x in pre] or [0.0]),
                            "min_d_exec_mm": min([x["d_exec_mm"] for x in pre] or [np.inf]),
                            "min_d_ref_mm": min([x["d_ref_mm"] for x in pre] or [np.inf])}
    if ir is not None:
        e = tl[ir]
        out["at_close_ref"] = {k2: e[k2] for k2 in ("t", "d_exec_mm", "d_ref_mm", "d_refined_mm", "mid_rel_apple_ref_mm")}
    out["run_min"] = {"d_exec_mm": min(x["d_exec_mm"] for x in tl), "d_ref_mm": min(x["d_ref_mm"] for x in tl),
                      "d_refined_mm": min(x["d_refined_mm"] for x in tl), "max_disp_xy_mm": max(x["disp_xy_mm"] for x in tl)}
    out["refined_minus_reference"] = rr
    out["timeline"] = tl
    return out


def main():
    outp = pathlib.Path(sys.argv[1])
    res = json.loads(outp.read_text()) if outp.exists() else {"definitions": __doc__, "runs": {}}
    for rd in sys.argv[2:]:
        name = pathlib.Path(rd).parent.name + "/" + pathlib.Path(rd).name
        if name in res["runs"]:
            continue
        r = run_one(rd)
        res["runs"][name] = r
        ac = r.get("at_close") or {}
        print(name, "succ", r["success"], "t_close", r["t_close"], "t_close_ref", r["t_close_ref"],
              "d_exec", round(ac.get("d_exec_mm", np.nan), 1), "d_ref", round(ac.get("d_ref_mm", np.nan), 1),
              "d_refined", round(ac.get("d_refined_mm", np.nan), 1), "mid", np.round(ac.get("mid_rel_apple_mm", [np.nan] * 3), 1),
              "cycles", r["n_close_cycles"], flush=True)
        res["n_invalid_geomdistance"] = P.N_INVALID[0]
        tmp = outp.with_suffix(".tmp"); tmp.write_text(json.dumps(res, default=float)); tmp.replace(outp)


if __name__ == "__main__":
    main()
