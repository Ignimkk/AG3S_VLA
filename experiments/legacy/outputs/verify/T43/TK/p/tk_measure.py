"""T43 TK — plan (P) / command (C) / measured (M) decomposition on recorded runs (verifier, CPU only, recorded data only).

usage: T43W_SNAP=<client snapshot> tk_measure.py ARM OUT_JSON
  ARM = V7g (analysis CT/raw/analysis_v7g.json) | V8 (V8/analysis/analysis.json) | E0 (Z5/analysis_z5.json, no TO, no obstacle)
  client snapshot only provides pi05_infer.main() up to load_remote_policy (scene reset incl. hurdle; T43/W/p/hold_w.reset);
  pi05_infer.py md5 is identical in work-t43w3 / work-t43v7 / work-t43v8 (raw/code_md5.txt).

Definitions (control row t, chunk with start row t0 = chunk_start_steps, k = t - t0 = step_in_chunk):
  P(t) = refined_chunk[k] of the executing chunk (server npz of the paired segment, analyze_t41 rule = analysis.json server_segment);
         E0: predicted_chunks[i][k] (policy chunk, no TO).
  C(t) = executed_actions[t] (traj.npz) = the value written into d.ctrl at row t (client applied_ctrl; HOLD rows = q_hold).
  M(t) = measured_qpos[t] (traj.npz, rby1_state() after the row's mj_steps) = post state of row t.  M_pre(t) = frames control row t qpos
         (after apply_action, before mj_step).  Index convention = the TO's: Q[:, k] is checked as the configuration after row k.
  Joint values: the 7 joints of the active arm (ep1834 right, else left), action layout [L7, Lg, R7, Rg].
  EE: world position of body EE_BODY_L|R, FK (mj_kinematics) of the full row qpos (M_pre(t) recorded) with the active-arm joints replaced.
  phase: before (t < grasp_t), carry (grasp_t <= t <= place_t), after (t > place_t); grasp_t/place_t from the arm's analysis.json;
         grasp false -> every row 'before'.
  row classes: exec (executed true, not the obstacle-stop row), hold (executed false), stop (last row when the client log has
         '[SAFETY] obstacle stop' -> its physics broke off early; excluded from C-M statistics).
  speed_cmd_mm = |EE(C(t)) - EE(C(t-1))| (mm per control row);  speed_meas_mm = |EE(M(t)) - EE(M(t-1))|;  jspeed_cmd = max_j |C_j(t) - C_j(t-1)|.
  lag (continuous, steps): tau* = argmin_{tau in [0, 6], 0.05} |EE(M(t)) - EE_C(t - tau)|, EE_C linearly interpolated between rows
         (rows t - tau < 1 not searched); same in joint space (L2 over 7 joints).  run lag = argmin_tau median_t over exec carry rows.
  q_now check (per chunk): planning-row qpos (captured at the frozen instant, = TransportScene.robot_state source) vs control row t0
         M_pre, vs M(t0 - 1), vs inference_states[i]; q_now - C(t0 - 1) and q_now - P_prev(last executed row): the offset at the chunk start.
  refined vs reference: |EE(refined_k) - EE(reference_k)| (TO modification, mm).
Contacts (ct.json of T43/V8 = CT rule; first row of each (run, body, phase) event): at row t the bar distance (mj_geomDistance, min over
  hurdle collision geoms and the contacting body's collision geoms, signed, mm) for P(t), C(t), M_pre(t), M(t) [M(t) full state =
  control row t+1 qpos when it exists, else row t qpos with the measured arm]; apple contacts: the apple is moved rigidly with the EE
  (pose relative to EE_BODY taken from the state the variant replaces).  TO clearance of the chunk: summary to.metrics.clearance_m,
  clearance_min_per_step_mm[k], clearance[k] min over the spheres of the contacting link (sphere_link_names), max_violation_pair.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import numpy as np, mujoco  # noqa: E402
import hold_w as HW  # noqa: E402

T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
ANALYSIS = {"V7g": T43 / "CT/raw/analysis_v7g.json", "V8": T43 / "V8/analysis/analysis.json", "E0": T43 / "Z5/analysis_z5.json"}
CT = T43 / "V8/ct.json"
TAUS = np.arange(0.0, 6.0001, 0.05)


def lit(x):
    import ast
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def segment_chunks(seg):
    if not seg or seg == "None":
        return {}
    seg = lit(seg); first = pathlib.Path(seg["first"]); i0 = int(first.stem.split("_")[1]); out = {}
    for n in range(int(seg["n"])):
        p = first.parent / f"chunk_{i0 + n:05d}.npz"
        with np.load(p, allow_pickle=True) as z:
            out[int(z["t_step"])] = p
    return out


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "mean": float(x.mean()),
            "max": float(x.max()), "min": float(x.min())}


def main():
    arm, out_path = sys.argv[1], pathlib.Path(sys.argv[2])
    runs = json.load(open(ANALYSIS[arm]))["runs"]
    ct = json.load(open(CT))["runs"]
    cl = HW.load_client()
    res = {"definitions": __doc__, "arm": arm, "snapshot": str(HW.HY), "runs": {}}
    model_done = False
    only = sys.argv[3:] if len(sys.argv) > 3 else None
    for key in sorted(runs):
        if only and key not in only:
            continue
        a = runs[key]; rd = pathlib.Path(a["run_dir"])
        if not (rd / "traj.npz").exists():
            continue
        man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = int(man["scene"]["episode_index"])
        z = np.load(rd / "traj.npz", allow_pickle=True); seed = int(z["policy_seed"])
        stt = rd.parent / f"{rd.name}.start"
        cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines()
                    if l.startswith("obstacle=") and l != "obstacle=:"), None)
        L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; mg = L.get("obstacle_manager")
        hurdle = sorted(mg.collision_geom_ids) if mg is not None else []
        side = "right" if ep == 1834 else "left"; fl = side[0]
        c0 = 0 if side == "left" else 8
        qadr = np.asarray(L["idx"][f"{side}_q"][:7], int)
        ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "EE_BODY_L" if side == "left" else "EE_BODY_R")
        apple_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
        apple_q = int(m.jnt_qposadr[m.body_jntadr[apple_b]])
        coll = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
        geoms_of = lambda b: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b and coll(g)]
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"

        if not model_done:
            # actuator / joint constants of the active arm (left of ep1807) at the first run's reset state
            mujoco.mj_forward(m, d); M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, d, M)
            rows = []
            for j, aid in enumerate(L["act"]["left_a"][:7]):
                jid = int(m.actuator_trnid[aid, 0]); dof = int(m.jnt_dofadr[jid])
                kp = float(m.actuator_gainprm[aid, 0]); kv = float(-m.actuator_biasprm[aid, 2]); b = float(m.dof_damping[dof])
                I = float(M[dof, dof]); wn = np.sqrt(kp / I); zeta = (kv + b) / (2 * np.sqrt(kp * I))
                rows.append({"joint": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid), "kp": kp, "kv": kv, "damping": b,
                             "armature": float(m.dof_armature[dof]), "I_diag": I, "wn_rad_s": float(wn), "zeta": float(zeta),
                             "ramp_lag_s_(kv+b)/kp": (kv + b) / kp, "ramp_lag_steps": (kv + b) / kp / (1.0 / 15.0),
                             "ctrlrange": m.actuator_ctrlrange[aid].tolist(), "forcerange": m.actuator_forcerange[aid].tolist(),
                             "forcelimited": int(m.actuator_forcelimited[aid])})
            spa = max(1, int(round(1.0 / (cl.CTRL_HZ * m.opt.timestep))))
            res["model"] = {"timestep": float(m.opt.timestep), "integrator": int(m.opt.integrator), "substeps_per_row": spa,
                            "row_period_s": spa * float(m.opt.timestep), "CTRL_HZ": cl.CTRL_HZ, "OPEN_LOOP_HORIZON": cl.OPEN_LOOP_HORIZON,
                            "left_arm": rows}
            model_done = True

        rows_f = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
        ctrl = {r["t_step"]: r for r in rows_f if r["kind"] == "control"}
        plan = {r["seq"]: r for r in rows_f if r["kind"] == "planning"}
        chunks = segment_chunks(a.get("server_segment"))
        EA = np.asarray(z["executed_actions"], float); MQ = np.asarray(z["measured_qpos"], float)
        PC = np.asarray(z["predicted_chunks"], float); CS = np.asarray(z["chunk_start_steps"], int); IS = np.asarray(z["inference_states"], float)
        T = len(EA)
        stop = "[SAFETY] obstacle stop" in (rd.parent / f"{rd.name}.client.log").read_text(errors="replace")
        g_t = a.get("grasp_t"); p_t = a.get("place_t")
        g_t = None if g_t in (None, "None") else int(g_t); p_t = None if p_t in (None, "None") else int(p_t)

        def phase(t):
            if g_t is None or t < g_t:
                return "before"
            if p_t is None or t <= p_t:
                return "carry"
            return "after"

        def fk(qfull, arm7):
            d.qpos[:] = qfull; d.qpos[qadr] = arm7; mujoco.mj_kinematics(m, d); return d.xpos[ee].copy()

        # chunk index of every row
        ci_of = np.searchsorted(CS, np.arange(T), side="right") - 1
        # server refined / reference chunks keyed by chunk index i (seq = chunk_seq of the control row)
        ref_by_i, refd_by_i, summ_by_i = {}, {}, {}
        for i, t0 in enumerate(CS):
            if t0 not in ctrl:
                continue
            seq = int(ctrl[t0]["chunk_seq"])
            if seq in chunks:
                with np.load(chunks[seq], allow_pickle=True) as zz:
                    refd_by_i[i] = np.asarray(zz["refined_chunk"], float); ref_by_i[i] = np.asarray(zz["reference_chunk"], float)
                    s = json.loads(str(zz["summary_json"]))
                    summ_by_i[i] = {"to": s.get("to") or {}, "clear_step": s.get("clearance_min_per_step_mm"),
                                    "clear": np.asarray(zz["clearance"], float), "links": s.get("sphere_link_names"),
                                    "exec_feedback": s.get("exec_feedback") or {}, "seq": seq}

        per = []
        EE_C = np.zeros((T, 3)); EE_M = np.zeros((T, 3)); EE_P = np.full((T, 3), np.nan); EE_R = np.full((T, 3), np.nan)
        J_C = EA[:, c0:c0 + 7]; J_M = MQ[:, c0:c0 + 7]; J_P = np.full((T, 7), np.nan); J_R = np.full((T, 7), np.nan)
        for t in range(T):
            q = np.asarray(ctrl[t]["qpos"], float); i = int(ci_of[t]); k = t - int(CS[i])
            if arm == "E0":
                J_P[t] = PC[i, k, c0:c0 + 7]
            elif i in refd_by_i:
                J_P[t] = refd_by_i[i][k, c0:c0 + 7]; J_R[t] = ref_by_i[i][k, c0:c0 + 7]
            EE_C[t] = fk(q, J_C[t]); EE_M[t] = fk(q, J_M[t])
            if np.all(np.isfinite(J_P[t])):
                EE_P[t] = fk(q, J_P[t])
            if np.all(np.isfinite(J_R[t])):
                EE_R[t] = fk(q, J_R[t])
        # lag search
        sgrid = np.arange(T, dtype=float)

        def interp(arr, s):
            return np.stack([np.interp(s, sgrid, arr[:, j]) for j in range(arr.shape[1])], -1)
        for t in range(T):
            r = ctrl[t]; i = int(ci_of[t]); k = t - int(CS[i]); ex = bool(r.get("executed"))
            cls = "stop" if (stop and t == T - 1) else ("exec" if ex else "hold")
            s = t - TAUS; ok = s >= 1.0
            tau_ee = tau_j = None; d_tau_ee = None
            if ok.any():
                de = np.linalg.norm(interp(EE_C, s[ok]) - EE_M[t], axis=1); ib = int(np.argmin(de)); tau_ee = float(TAUS[ok][ib]); d_tau_ee = 1e3 * float(de[ib])
                dj = np.linalg.norm(interp(J_C, s[ok]) - J_M[t], axis=1); tau_j = float(TAUS[ok][int(np.argmin(dj))])
            per.append({
                "t": t, "k": k, "i": i, "phase": phase(t), "cls": cls,
                "PC_rad": None if not np.all(np.isfinite(J_P[t])) else (J_P[t] - J_C[t]).tolist(),
                "CM_rad": (J_C[t] - J_M[t]).tolist(),
                "CMpre_rad": (J_C[t] - np.asarray(r["qpos"], float)[qadr]).tolist(),
                "PC_mm": None if not np.all(np.isfinite(EE_P[t])) else 1e3 * float(np.linalg.norm(EE_P[t] - EE_C[t])),
                "CM_mm": 1e3 * float(np.linalg.norm(EE_C[t] - EE_M[t])),
                "PM_mm": None if not np.all(np.isfinite(EE_P[t])) else 1e3 * float(np.linalg.norm(EE_P[t] - EE_M[t])),
                "RP_mm": None if not np.all(np.isfinite(EE_R[t])) else 1e3 * float(np.linalg.norm(EE_R[t] - EE_P[t])),
                "speed_cmd_mm": None if t == 0 else 1e3 * float(np.linalg.norm(EE_C[t] - EE_C[t - 1])),
                "speed_meas_mm": None if t == 0 else 1e3 * float(np.linalg.norm(EE_M[t] - EE_M[t - 1])),
                "jspeed_cmd": None if t == 0 else float(np.max(np.abs(J_C[t] - J_C[t - 1]))),
                "tau_ee": tau_ee, "tau_j": tau_j, "d_at_tau_ee_mm": d_tau_ee,
                "EE_P": None if not np.all(np.isfinite(EE_P[t])) else EE_P[t].tolist(), "EE_C": EE_C[t].tolist(), "EE_M": EE_M[t].tolist()})
        # run-level lag over exec carry rows
        carry_ex = [p["t"] for p in per if p["cls"] == "exec" and p["phase"] == "carry" and p["t"] >= 7]
        run_lag = None
        if carry_ex:
            med = []
            for tau in TAUS:
                med.append(np.median([np.linalg.norm(interp(EE_C, np.array([t - tau]))[0] - EE_M[t]) for t in carry_ex]))
            run_lag = {"tau_steps": float(TAUS[int(np.argmin(med))]), "median_mm_at_tau": 1e3 * float(np.min(med)),
                       "median_mm_at_0": 1e3 * float(med[0]), "n_rows": len(carry_ex)}
        # q_now check per chunk
        qn = []
        for i, t0 in enumerate(CS):
            if t0 not in ctrl:
                continue
            seq = int(ctrl[t0]["chunk_seq"]); pr = plan.get(seq)
            q_cap = np.asarray(pr["qpos"], float)[qadr] if pr is not None else None
            q_pre = np.asarray(ctrl[t0]["qpos"], float)[qadr]
            e = {"i": i, "t0": int(t0), "seq": seq, "phase": phase(int(t0)),
                 "cap_minus_pre0_rad_max": None if q_cap is None else float(np.max(np.abs(q_cap - q_pre))),
                 "cap_minus_infstate_rad_max": None if q_cap is None else float(np.max(np.abs(q_cap - IS[i, c0:c0 + 7])))}
            if t0 >= 1 and q_cap is not None:
                e["cap_minus_Mprev_rad_max"] = float(np.max(np.abs(q_cap - J_M[t0 - 1])))
                e["qnow_minus_Cprev_rad"] = (q_cap - J_C[t0 - 1]).tolist()
                e["qnow_minus_Cprev_mm"] = 1e3 * float(np.linalg.norm(fk(np.asarray(pr["qpos"], float), q_cap) - EE_C[t0 - 1]))
                e["prev_row_cls"] = per[t0 - 1]["cls"]
                if np.all(np.isfinite(J_P[t0 - 1])):
                    e["qnow_minus_Pprev_rad"] = (q_cap - J_P[t0 - 1]).tolist()
                    e["qnow_minus_Pprev_mm"] = 1e3 * float(np.linalg.norm(fk(np.asarray(pr["qpos"], float), q_cap) - EE_P[t0 - 1]))
            if np.all(np.isfinite(J_P[t0])) and q_cap is not None:
                e["P0_minus_qnow_rad_max"] = float(np.max(np.abs(J_P[t0] - q_cap)))
                if np.all(np.isfinite(J_R[t0])):
                    e["R0_minus_qnow_rad_max"] = float(np.max(np.abs(J_R[t0] - q_cap)))
            if i in summ_by_i:
                fb = summ_by_i[i]["exec_feedback"]; aa = lit(fb.get("applied_arm"))
                if fb.get("available") and isinstance(aa, list) and i >= 1:
                    tp = int(CS[i - 1]); n = len(aa)
                    e["feedback_applied_vs_EA_max"] = float(np.max(np.abs(np.asarray(aa, float) - EA[tp:tp + n, [*range(0, 7), *range(8, 15)]])))
            if i in refd_by_i and arm != "E0":
                n = min(8, T - int(t0))
                exr = [tt for tt in range(int(t0), int(t0) + n) if per[tt]["cls"] != "hold"]
                if exr:
                    e["refined_vs_EA_max"] = float(np.max(np.abs(refd_by_i[i][[tt - int(t0) for tt in exr]][:, [*range(0, 7), *range(8, 15)]]
                                                         - EA[exr][:, [*range(0, 7), *range(8, 15)]])))
                e["refined_vs_traj_predicted_max"] = float(np.max(np.abs(refd_by_i[i] - PC[i])))
            qn.append(e)
        # contacts
        cons = []
        for c in (ct.get(key, {}) or {}).get("contacts", []) if arm != "E0" else []:
            t = int(c["first_t"]); body = c["body"]
            if side == "right" and body.startswith("ee_finger_l"):
                body = body.replace("ee_finger_l", "ee_finger_r")
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
            bg = geoms_of(bid)
            q_pre = np.asarray(ctrl[t]["qpos"], float)
            if t + 1 in ctrl:
                q_post = np.asarray(ctrl[t + 1]["qpos"], float); post_src = "control row t+1 qpos"
            else:
                q_post = q_pre.copy(); q_post[qadr] = J_M[t]; post_src = "row t qpos + measured arm (obstacle stop row)"

            def ee_T(qfull):
                d.qpos[:] = qfull; mujoco.mj_kinematics(m, d)
                Tm = np.eye(4); Tm[:3, :3] = d.xmat[ee].reshape(3, 3); Tm[:3, 3] = d.xpos[ee]; return Tm

            def bar_dist(qfull, arm7, move_apple):
                q = qfull.copy()
                if move_apple:
                    T0 = ee_T(q); q2 = q.copy(); q2[qadr] = arm7; T1 = ee_T(q2)
                    pa = q[apple_q:apple_q + 3]; Ra = np.zeros(9); mujoco.mju_quat2Mat(Ra, q[apple_q + 3:apple_q + 7])
                    Ta = np.eye(4); Ta[:3, :3] = Ra.reshape(3, 3); Ta[:3, 3] = pa
                    Tn = T1 @ np.linalg.inv(T0) @ Ta; qa = np.zeros(4); mujoco.mju_mat2Quat(qa, Tn[:3, :3].reshape(-1))
                    q2[apple_q:apple_q + 3] = Tn[:3, 3]; q2[apple_q + 3:apple_q + 7] = qa; q = q2
                else:
                    q[qadr] = arm7
                d.qpos[:] = q; mujoco.mj_kinematics(m, d)
                best = (np.inf, None); ft = np.zeros(6)
                for g1 in bg:
                    for g2 in hurdle:
                        dist = mujoco.mj_geomDistance(m, d, int(g1), int(g2), 1.0, ft)
                        if dist < best[0]:
                            best = (dist, gname(g2))
                return 1e3 * best[0], best[1]
            mv = c["kind"] == "apple"
            i = int(ci_of[t]); k = t - int(CS[i])
            e = {"key": key, "body": body, "kind": c["kind"], "phase_ct": c["phase"], "t": t, "k": k, "chunk_i": i,
                 "seq": int(ctrl[t]["chunk_seq"]), "row_cls": per[t]["cls"], "ct_max_pen_mm": c.get("max_pen_mm"),
                 "post_src": post_src, "n_body_geoms": len(bg)}
            for nm, base, arm7 in (("P", q_post, J_P[t]), ("C", q_post, J_C[t]), ("M", q_post, q_post[qadr]), ("M_pre", q_pre, q_pre[qadr])):
                if np.all(np.isfinite(arm7)):
                    dd, gg = bar_dist(base, arm7, mv); e[f"bar_{nm}_mm"] = dd; e[f"bar_{nm}_geom"] = gg
            # earlier commands (= what M would be with a pure j-row delay) and the planned rows of the previous + executing chunk up to k
            for j in (1, 2, 3):
                if t - j >= 0:
                    e[f"bar_Cm{j}_mm"] = bar_dist(q_post, J_C[t - j], mv)[0]
            pr_rows = [tt for tt in range(max(0, int(CS[max(i - 1, 0)])), t + 1) if np.all(np.isfinite(J_P[tt]))]
            if pr_rows:
                bp = [bar_dist(q_post, J_P[tt], mv)[0] for tt in pr_rows]
                e["bar_Prows_min_mm"] = float(np.min(bp)); e["bar_Prows_argmin_t"] = int(pr_rows[int(np.argmin(bp))])
            e.update({"PC_mm": per[t]["PC_mm"], "CM_mm": per[t]["CM_mm"], "PM_mm": per[t]["PM_mm"], "tau_ee": per[t]["tau_ee"],
                      "speed_cmd_mm": per[t]["speed_cmd_mm"]})
            if i in summ_by_i:
                S = summ_by_i[i]; to = S["to"]; met = lit(to.get("metrics")) or {}
                e["to_status"] = to.get("status"); e["to_clearance_mm"] = None if met.get("clearance_m") is None else 1e3 * float(met["clearance_m"])
                cs_ = lit(S["clear_step"]); e["clear_row_k_mm"] = None if not cs_ else float(cs_[k]) if k < len(cs_) else None
                links = lit(S["links"]) or []
                if links and k < S["clear"].shape[0]:
                    msk = np.asarray([l == body for l in links])
                    e["clear_row_k_link_mm"] = 1e3 * float(S["clear"][k][msk].min()) if msk.any() else None
                    e["clear_chunk_link_mm"] = 1e3 * float(S["clear"][:, msk].min()) if msk.any() else None
                e["max_violation_pair"] = met.get("max_violation_pair")
            cons.append(e)
        res["runs"][key] = {"run_dir": str(rd), "side": side, "episode": ep, "seed": seed, "stage": a.get("stage"), "grasp_t": g_t,
                            "place_t": p_t, "n_rows": T, "obstacle_stop": stop, "n_hurdle_geoms": len(hurdle),
                            "n_server_chunks": len(refd_by_i), "n_chunks": len(CS), "run_lag": run_lag, "rows": per,
                            "qnow": qn, "contacts": cons}
        print(key, T, "lag", run_lag, "contacts", len(cons), flush=True)
    json.dump(res, open(out_path, "w"))


if __name__ == "__main__":
    main()
