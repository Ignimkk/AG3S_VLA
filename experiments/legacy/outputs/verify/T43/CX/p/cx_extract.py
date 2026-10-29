"""T43 CX — per-case extraction, V11AB vs V11ABC at the same ep:seed (verifier, CPU only, numbers only).
usage: T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v11 cx_extract.py OUT_JSON OUT_NPZ_DIR

Cases: H 1807 s18071, H 1807 s18073, H 1967 s19672, H 1995 s19953, B 1967 s19672 (V11ABC drops), H 1982 s19822, B 1982 s19822 (V11ABC no grasp).
Server pairing: V11/analysis/analysis_<ARM>.json server_segment (analyze_t41 rule: the segment whose every refined_chunk equals the run's
planning actions <= 1e-6, as many chunks as planning rows); re-checked here chunk by chunk (pair_max_abs).
Per chunk (server summary_json + npz):
  grasp.state / attach / detach / held_slip / attach_revoked; client gate action (planning row 'safe'), to.status, worst waypoint row
  (link, clearance_mm, step, tier, point_m), verdict classification collision_by_why, lookahead gate, deflection_rate {cost, max_step_change_rad},
  server servo_accel {n_over, overshoot_max_rad, boundary_velocity_ratio, boundary_velocity_overshoot_rad} (ABC only).
  C rows recomputed (both arms; for V11AB = what the C rows would have measured on the AB command): X = [u-2, u-1, Q0..Q15] with
  u = exec_feedback.applied_arm[-2:] (safe_policy._ta_history rule), Q = refined_chunk rows 0..15, 14 arm joints (cols 0-6, 8-14);
  row j = X[j+2] - 2X[j+1] + X[j] (j = 0..15; j = 0, 1 straddle the boundary); bound b = min(max(a_C, |same row on the reference|), 0.04)
  (problem.servo_accel_rows, relaxed; max_step_change = a_URDF 10 * 0.9 * (1/15)^2 = 0.04 rad/row^2); a_C from the ABC record.
  n_bind = rows with |row| >= b - 1e-6 (at or over the bound), n_over = |row| > b + 1e-9, slack_max = max(|row| - b, 0);
  split: boundary rows j in {0,1}, window rows j = 0..7, tail j = 8..15. bvel_ratio = max_j |Q0 - u-1| / max_step_j.
  refine_dev_rad = max |refined - reference| over rows 0..7, 14 arm joints; tcp_dev_mm = max over rows 0..7 |TCP(refined) - TCP(reference)|.
  closing chunks (grasp.state closing, not attached): fingertip midpoint at row 7 vs apple centre (T43W tip_radial) -> outward_mm.
Divergence AB vs ABC (by seq): qpos (planning row, 16D measured-state columns of the record) max |diff|; reference rows 0..7 max |diff|;
  refined rows 0..7 max |diff|; applied control rows of that chunk max |diff| (traj.npz executed_actions).
Replay timeline (CT/p/replay_ct.py npz, V11/replay for H, CX/replay for B): per control row t, apple centre in the TCP frame relative to the
  fingertip midpoint (rel = R_tcp^T (apple - finger_mid)), slip_mm = |rel(t) - rel(ref)| with rel(ref) = mean rel over rows grasp_t+8 .. grasp_t+15;
  apple-hand contact sub-steps, max normal force; grip_cmd; finger_open_norm; hold flag; apple speed.
  separation row t_sep = first row >= grasp_t with apple-hand contact 0 on 3 consecutive rows (drop or intended release),
  or (if none) the first row with slip_mm > 20. grasp_t = fail_stage grasp_t, or (no grasp) the first attach chunk's t.
  slip reference rows: grasp_t+8 .. grasp_t+15, or grasp_t .. grasp_t+3 when t_sep < grasp_t + 16.
Hand kinematics: commanded TCP = FK(site left_ee) of executed_actions active-arm columns on the control row qpos; measured TCP = replay tcp
  (post state of row t-1 = state at row t); a = second difference * 15^2 (m/s^2), jerk = third difference * 15^3 (m/s^3), norms.
  window = t_sep - 24 .. t_sep + 4: max |a|, max |jerk| commanded / measured; and the apple-in-hand relative acceleration (second difference of
  rel * 15^2)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import hurdle_metrics as HM  # noqa: E402

T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
V11 = T43 / "V11"
CASES = [("H", 1807, 18071), ("H", 1807, 18073), ("H", 1967, 19672), ("H", 1995, 19953), ("B", 1967, 19672), ("H", 1982, 19822), ("B", 1982, 19822)]
ARMS = ("V11AB", "V11ABC")
J14 = list(range(0, 7)) + list(range(8, 15))
MAX_STEP = np.array([np.pi, np.pi, np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2.094395102] * 2) * 0.9 / 15.0
MSC = 10.0 * 0.9 / 15.0 ** 2
LEFT_Q = list(range(20, 27))
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x
A_C = None


def fk_site(qfull, arm7):
    M = HM.model(); m, d = M["m"], M["d"]
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
    d.qpos[:] = qfull; d.qpos[LEFT_Q] = arm7; mujoco.mj_kinematics(m, d)
    return d.site_xpos[site].copy()


def tip_mid(qfull, arm7):
    M = HM.model(); m, d = M["m"], M["d"]; b = M["bid"]
    d.qpos[:] = qfull; d.qpos[LEFT_Q] = arm7; mujoco.mj_kinematics(m, d)
    return 0.5 * (d.xpos[b["ee_finger_l1"]] + d.xpos[b["ee_finger_l2"]])


def c_rows(ref, ref_f, hist, a_c):
    """ref / ref_f: (P, 14) reference / refined rows; hist (2, 14) or None. returns dict."""
    X = ref_f if hist is None else np.vstack([hist, ref_f]); Xr = ref if hist is None else np.vstack([hist, ref])
    v = X[2:] - 2 * X[1:-1] + X[:-2]; vr = Xr[2:] - 2 * Xr[1:-1] + Xr[:-2]
    b = np.minimum(np.maximum(a_c[None, :], np.abs(vr)), MSC)
    over = np.abs(v) - b
    off = 0 if hist is not None else 2   # without history row j=0 is X[0..2] = Q0..Q2 (no boundary rows)
    js = np.arange(len(v)) + off   # index j in the 'with history' numbering
    out = {}
    for name, sel in (("boundary", js <= 1), ("window", js <= 7), ("tail", js >= 8), ("all", js >= 0)):
        o = over[sel]
        out[name] = {"n_rows": int(o.size), "n_bind": int((o >= -1e-6).sum()), "n_over": int((o > 1e-9).sum()),
                     "slack_max_rad": float(max(o.max(initial=-np.inf), 0.0)) if o.size else 0.0,
                     "slack_sum_rad": float(np.clip(o, 0, None).sum()),
                     "abs_max_rad": float(np.abs(v[sel]).max(initial=0.0)), "ref_abs_max_rad": float(np.abs(vr[sel]).max(initial=0.0))}
    out["bvel_ratio"] = None if hist is None else float((np.abs(ref_f[0] - hist[-1]) / MAX_STEP).max())
    out["bvel_ratio_reference"] = None if hist is None else float((np.abs(ref[0] - hist[-1]) / MAX_STEP).max())
    return out


def load_run(arm, stage, ep, seed):
    key = f"{arm}/{stage}/{arm}_ep{ep}_s{seed}"
    A = json.load(open(V11 / f"analysis/analysis_{arm}.json"))["runs"][key]
    rd = pathlib.Path(A["run_dir"])
    rows = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
    plan = [r for r in rows if r["kind"] == "planning"]; ctrl = {r["t_step"]: r for r in rows if r["kind"] == "control"}
    z = np.load(rd / "traj.npz", allow_pickle=True)
    first = pathlib.Path(A["server_segment"]["first"]); i0 = int(first.stem.split("_")[1])
    files = [first.parent / f"chunk_{i0 + n:05d}.npz" for n in range(len(plan))]
    return key, A, rd, plan, ctrl, z, files


def chunks(arm, plan, files, apple_by_t):
    global A_C
    out = []
    for p, f in zip(plan, files):
        z = np.load(f, allow_pickle=True); s = json.loads(str(z["summary_json"]))
        assert int(z["t_step"]) == p["seq"], (f, p["seq"])
        refined = np.asarray(z["refined_chunk"], np.float64); ref = np.asarray(z["reference_chunk"], np.float64)
        pair_err = float(np.abs(refined - np.asarray(p["actions"], np.float64)).max())
        to = s.get("to") or {}; m = to.get("metrics") or {}; mv = lit(m.get("max_violation_pair")) or {}
        g = s.get("grasp") or {}; att = lit(g.get("attached"))
        sa = lit(m.get("servo_accel")) if m.get("servo_accel") is not None else None
        if sa and A_C is None:
            A_C = np.asarray(sa["a_c_rad_per_row2"], float)
        dr = lit(m.get("deflection_rate")) if m.get("deflection_rate") is not None else None
        la = lit(m.get("lookahead")) if m.get("lookahead") is not None else None
        fb = s.get("exec_feedback") or {}
        hist = None
        if fb.get("available") and fb.get("applied_arm") is not None and len(fb["applied_arm"]) >= 2:
            hist = np.asarray(fb["applied_arm"], np.float64)[-2:]
        cls = (s.get("verdict") or {}).get("classification") or {}
        e = {"seq": p["seq"], "t": p["t_step"], "pair_max_abs": pair_err, "gate_action": (p.get("gate") or {}).get("action"), "safe": p.get("safe"),
             "hold_kinds": (p.get("gate") or {}).get("hold_kinds"), "to_status": to.get("status"),
             "grasp_state": g.get("state"), "attach": bool(g.get("attach")), "detach": bool(g.get("detach")), "held_slip": g.get("held_slip"),
             "attach_revoked": g.get("attach_revoked"), "attached": bool(att),
             "wp": {"link": mv.get("link"), "clearance_mm": None if mv.get("clearance_m") is None else 1e3 * mv["clearance_m"], "step": mv.get("step"),
                    "tier": mv.get("tier"), "point_m": mv.get("point_m"), "query": mv.get("query")},
             "collision_by_why": lit(cls.get("collision_by_why")), "follow_to": ((s.get("verdict") or {}).get("follow_to") or {}).get("decision"),
             "lookahead_gate": (la or {}).get("gate"), "deflection_rate": None if not dr else {"cost": dr.get("cost"), "max_step_change_rad": dr.get("max_step_change_rad"), "boundary": dr.get("boundary")},
             "servo_accel_server": None if not sa else {k: sa.get(k) for k in ("boundary", "n_over", "overshoot_max_rad", "boundary_velocity_ratio", "boundary_velocity_overshoot_rad")},
             "exec_feedback": {"n_exec": fb.get("n_exec"), "n_hold": fb.get("n_hold"), "hold_kind": fb.get("hold_kind")},
             "ta_history": s.get("ta_history"),
             "planned_gripper_row0": (fb.get("planned_gripper") or [[None]])[0][0] if fb.get("planned_gripper") else None,
             "ref_grip_rows0_7": [float(x) for x in ref[:8, 7]], "refined_grip_rows0_7": [float(x) for x in refined[:8, 7]],
             "refine_dev_rad": float(np.abs(refined[:8][:, J14] - ref[:8][:, J14]).max())}
        e["_ref14"] = ref[:16][:, J14]; e["_fin14"] = refined[:16][:, J14]; e["_hist"] = hist
        q = np.asarray(p["qpos"], float)
        e["tcp_dev_mm"] = float(max(np.linalg.norm(fk_site(q, refined[k, 0:7]) - fk_site(q, ref[k, 0:7])) for k in range(8)) * 1e3)
        if e["grasp_state"] == "closing" and not e["attached"] and (p.get("object_poses") or {}).get("apple"):
            ap = np.asarray(p["object_poses"]["apple"]["pos"], float)
            dr_ = np.linalg.norm(tip_mid(q, refined[7, 0:7]) - ap) * 1e3; dref = np.linalg.norm(tip_mid(q, ref[7, 0:7]) - ap) * 1e3
            e["closing_tip"] = {"d_refined_mm": float(dr_), "d_reference_mm": float(dref), "outward_mm": float(dr_ - dref)}
        out.append(e)
    return out


def timeline(arm, stage, ep, seed, z, ctrl, grasp_t):
    name = f"{arm}_{stage}_{arm}_ep{ep}_s{seed}.npz"
    f = (V11 / "replay" / name) if (V11 / "replay" / name).exists() else (T43 / "CX/replay" / name)
    R = np.load(f, allow_pickle=True)
    T = 600
    ap, fm, tcp, tR = R["apple_pos"], R["finger_mid"], R["tcp"], R["tcp_R"]
    rel = np.einsum("tji,tj->ti", tR, ap - fm)   # R^T (apple - finger_mid), index = state after row t-1
    EA = np.asarray(z["executed_actions"], float)
    cmd = np.array([fk_site(np.asarray(ctrl[t]["qpos"], float), EA[t, 0:7]) for t in range(T)])
    tl = {"rel_mm": 1e3 * rel, "apple_pos": ap, "finger_mid": fm, "tcp_meas": tcp, "tcp_cmd": cmd,
          "grip_cmd": R["grip_cmd"], "finger_open": R["finger_open_norm"], "hold": R["hold"], "executed": R["executed"],
          "cnt_apple_hand": R["cnt_apple-hand"], "f_apple_hand": R["maxf_apple-hand"], "cnt_apple_obst": R["cnt_apple-obstacle"],
          "cnt_hand_obst": R["cnt_hand-obstacle"], "apple_vel": R["apple_linvel"], "source": str(f)}
    out = {"replay": str(f)}
    if grasp_t is None:
        return tl, out
    c = R["cnt_apple-hand"]; vz = R["apple_linvel"][1:, 2]
    tsep = None
    for t in range(int(grasp_t), T - 3):
        if c[t] == 0 and c[t + 1] == 0 and c[t + 2] == 0:
            tsep = t; break
    tsep_rule = "contact0x3"
    g0 = int(grasp_t) + 8; ref_rows = [g0, g0 + 8]
    if tsep is not None and tsep < int(grasp_t) + 16:
        ref_rows = [int(grasp_t), int(grasp_t) + 4]   # short hold: reference = the first 4 rows after grasp_t
    ref = rel[ref_rows[0] + 1:ref_rows[1] + 1].mean(0)   # rel index t+1 = state after row t
    slip = np.linalg.norm(rel - ref, axis=1) * 1e3; tl["slip_mm"] = slip
    if tsep is None:
        w = np.where(slip[int(grasp_t):T] > 20.0)[0]; tsep = int(grasp_t) + int(w[0]) if len(w) else None; tsep_rule = "slip>20mm" if tsep is not None else None
    ga = int(grasp_t)
    out.update({"t_sep": tsep, "t_sep_rule": tsep_rule, "rel_ref_rows": ref_rows, "rel_ref_mm": (1e3 * ref).tolist(),
                "rel_at_grasp_mm": (1e3 * rel[ga + 1:ga + 5].mean(0)).tolist(), "finger_open_at_grasp": float(R["finger_open_norm"][ga + 1:ga + 5].mean()),
                "grip_cmd_at_grasp": float(R["grip_cmd"][ga:ga + 4].mean()), "apple_hand_force_at_grasp_N": float(R["maxf_apple-hand"][ga:ga + 4].mean()),
                "rows_in_hand": None if tsep is None else int(tsep - ga),
                "hold_rows_in_hand": None if tsep is None else int(R["hold"][ga:tsep].sum())})
    if tsep is not None and tsep - ga > 24:
        s0, s1 = ref_rows[1], tsep - 2
        out["creep_mm_per_100rows"] = float((slip[s1] - slip[s0]) / max(s1 - s0, 1) * 100)
    acc = lambda P: np.linalg.norm(np.diff(P, 2, axis=0), axis=1) * 15 ** 2
    jerk = lambda P: np.linalg.norm(np.diff(P, 3, axis=0), axis=1) * 15 ** 3
    meas = tcp[1:]   # state at end of row t (index t)
    tl.update({"acc_cmd": acc(cmd), "acc_meas": acc(meas), "jerk_cmd": jerk(cmd), "jerk_meas": jerk(meas), "acc_rel": acc(rel[1:])})
    if tsep is not None:
        a0, a1 = max(tsep - 24, 2), min(tsep + 4, T - 3)
        sl = slice(a0 - 1, a1 - 1)  # diff index k ~ row k+1 centre
        out["window"] = {"rows": [a0, a1], "acc_cmd_max": float(tl["acc_cmd"][sl].max()), "acc_meas_max": float(tl["acc_meas"][sl].max()),
                         "jerk_cmd_max": float(tl["jerk_cmd"][sl].max()), "jerk_meas_max": float(tl["jerk_meas"][sl].max()),
                         "acc_rel_max": float(tl["acc_rel"][sl].max()),
                         "slip_mm_at_tsep_minus8": float(slip[max(tsep - 8, 0)]), "slip_mm_at_tsep": float(slip[tsep]),
                         "slip_first_gt5": next((int(t) for t in range(int(grasp_t) + 16, T) if slip[t] > 5), None),
                         "slip_first_gt10": next((int(t) for t in range(int(grasp_t) + 16, T) if slip[t] > 10), None),
                         "grip_cmd_rows": [float(x) for x in R["grip_cmd"][max(tsep - 24, 0):tsep + 4]],
                         "finger_open_rows": [float(x) for x in R["finger_open_norm"][max(tsep - 24, 0) + 1:tsep + 5]],
                         "hold_rows": [bool(x) for x in R["hold"][max(tsep - 24, 0):tsep + 4]],
                         "cnt_apple_hand_rows": [int(x) for x in c[max(tsep - 24, 0):tsep + 4]],
                         "apple_obstacle_contact_rows_before_tsep": int((R["cnt_apple-obstacle"][int(grasp_t):tsep] > 0).sum()),
                         "hand_obstacle_contact_rows_before_tsep": int((R["cnt_hand-obstacle"][int(grasp_t):tsep] > 0).sum())}
    # carry-wide (grasp_t+16 .. tsep or +120): peak accelerations
    e = tsep if tsep is not None else min(int(grasp_t) + 120, T - 3)
    s0 = int(grasp_t) + 16
    if e > s0 + 3:
        out["carry"] = {"rows": [s0, e], "acc_cmd_p90": float(np.percentile(tl["acc_cmd"][s0:e], 90)), "acc_cmd_max": float(tl["acc_cmd"][s0:e].max()),
                        "acc_meas_p90": float(np.percentile(tl["acc_meas"][s0:e], 90)), "acc_meas_max": float(tl["acc_meas"][s0:e].max()),
                        "jerk_cmd_max": float(tl["jerk_cmd"][s0:e].max()), "jerk_meas_max": float(tl["jerk_meas"][s0:e].max()),
                        "slip_mm_max": float(slip[s0:e].max())}
    return tl, out


def main():
    outp = pathlib.Path(sys.argv[1]); nd = pathlib.Path(sys.argv[2]); nd.mkdir(parents=True, exist_ok=True)
    VJ = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43V11ABC.verify.json')))["numbers"]["fail_stage"]["runs"]
    res = {"definitions": __doc__, "cases": {}}
    HM.set_arm("left"); HM.model()
    for stage, ep, seed in CASES:
        ck = f"{stage}:{ep}_s{seed}"; C = {}
        for arm in ARMS:
            key, A, rd, plan, ctrl, z, files = load_run(arm, stage, ep, seed)
            ch = chunks(arm, plan, files, None)
            C[arm] = {"key": key, "A": A, "plan": plan, "ctrl": ctrl, "z": z, "chunks": ch}
        for arm in ARMS:
            for e in C[arm]["chunks"]:
                e["c_rows"] = c_rows(e["_ref14"], e["_fin14"], e["_hist"], A_C)
        case = {"fail_stage": {a: {k: VJ[ck][a].get(k) for k in ("stage", "grasp_t", "place_t", "release_t", "detach_slip_chunks")} for a in ARMS}}
        # divergence by seq
        by = {a: {e["seq"]: (e, p) for e, p in zip(C[a]["chunks"], C[a]["plan"])} for a in ARMS}
        div = []
        EA = {a: np.asarray(C[a]["z"]["executed_actions"], float) for a in ARMS}
        for sq in sorted(set(by["V11AB"]) & set(by["V11ABC"])):
            (e1, p1), (e2, p2) = by["V11AB"][sq], by["V11ABC"][sq]
            t = p1["t_step"]
            div.append({"seq": sq, "t": t, "t_abc": p2["t_step"],
                        "qpos_max_abs": float(np.abs(np.asarray(p1["qpos"]) - np.asarray(p2["qpos"])).max()),
                        "ref_max_abs": float(np.abs(e1["_ref14"][:8] - e2["_ref14"][:8]).max()),
                        "refined_max_abs": float(np.abs(e1["_fin14"][:8] - e2["_fin14"][:8]).max()),
                        "applied_max_abs": float(np.abs(EA["V11AB"][t:t + 8][:, J14] - EA["V11ABC"][t:t + 8][:, J14]).max()) if t + 8 <= 600 else None,
                        "grasp_state": [e1["grasp_state"], e2["grasp_state"]], "gate": [e1["gate_action"], e2["gate_action"]]})
        fd = lambda k, tol: next(((d["seq"], d["t"]) for d in div if d[k] is not None and d[k] > tol), None)
        case["divergence"] = {"first_refined_gt_1e-4": fd("refined_max_abs", 1e-4), "first_ref_gt_1e-4": fd("ref_max_abs", 1e-4),
                              "first_qpos_gt_1e-6": fd("qpos_max_abs", 1e-6), "first_applied_gt_1e-4": fd("applied_max_abs", 1e-4),
                              "first_qpos_gt_1e-3": fd("qpos_max_abs", 1e-3), "per_seq": div[:40]}
        case["arms"] = {}
        for arm in ARMS:
            gt = case["fail_stage"][arm]["grasp_t"]; gsrc = "fail_stage.grasp_t"
            if gt is None:   # no grasp by the analyze_e0 rule: use the first attach chunk's t (server grasp.attach)
                gt = next((e["t"] for e in C[arm]["chunks"] if e["attach"]), None); gsrc = "first_attach_chunk_t"
            tl, tout = timeline(arm, stage, ep, seed, C[arm]["z"], C[arm]["ctrl"], gt)
            tout["grasp_t_used"] = gt; tout["grasp_t_source"] = gsrc
            np.savez_compressed(nd / f"tl_{arm}_{stage}_{ep}_s{seed}.npz", **{k: v for k, v in tl.items() if k != "source"})
            chs = [{k: v for k, v in e.items() if not k.startswith("_")} for e in C[arm]["chunks"]]
            case["arms"][arm] = {"key": C[arm]["key"], "server_segment": C[arm]["A"]["server_segment"], "pair_max_abs_max": max(e["pair_max_abs"] for e in chs),
                                 "timeline": tout, "chunks": chs}
        res["cases"][ck] = case
        d = case["divergence"]; print(ck, "div", d["first_refined_gt_1e-4"], d["first_ref_gt_1e-4"], d["first_qpos_gt_1e-6"],
                                    {a: case["arms"][a]["timeline"].get("t_sep") for a in ARMS}, flush=True)
    res["a_c_rad_per_row2"] = None if A_C is None else A_C.tolist(); res["max_step_change_rad"] = MSC; res["max_step_rad"] = MAX_STEP.tolist()
    outp.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
