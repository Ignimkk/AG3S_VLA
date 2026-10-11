"""T43 HX — hurdle crossing time line (fingers / palm / held apple vs the bar): P (plan = command) and M (measured)
geometric distance, the TO's ESDF clearance of the same rows (reconstructed), margin applied, binding row; cause of every
moment below the margin; servo model (AG3S/trajopt/servo.py) applied offline to the recorded commands.
(verifier, CPU only, recorded data only, no new runs.)

usage: T43W_SNAP=<client snapshot> PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8 hx_measure.py ARM OUT_JSON [KEY ...]
  ARM = V7g | V8 | V5 ; index = analysis json (V7g CT/raw/analysis_v7g.json, V8 V8/analysis/analysis.json, V5 W/analysis/analysis.json)

Definitions (control row t, executing chunk = server npz with t_step == chunk_seq of row t, k = step_in_chunk):
  C(t)  = executed_actions[t] (value written to d.ctrl, both arms).  P(t) = refined_chunk[k] of the executing chunk.  On executed
          rows P == C (T43TK).  HOLD rows (executed false): C = q_hold, the refined row is not executed.
  M(t)  = post state of row t = control row t+1 qpos (full MuJoCo state incl. apple); last row: row t qpos with the arm = measured_qpos[t].
  Geometric distance to the bar: min over hurdle collision geoms (bar + 2 legs, obstacle manager) of mj_geomDistance (signed, mm,
          distmax 0.5 m) against: fingers = collision geoms of ee_finger_{l1,l2} (ep1834: r1, r2); palm = collision geoms of
          EE_BODY_L|R (= TO link ee_left|ee_right); apple = apple_geom.  P / servo state: arm joints replaced, the apple moved
          rigidly with the EE body from the M state (TK / DX rule).
  TO rows (reconstruction of linearize._esdf_clearance, snapshot work-t43v8 AG3S/trajopt/linearize.py:686-800):
          robot spheres = the record's sphere_centres (base frame, refined row 0 = sphere_states(result.trajectory)[0],
          safe_policy.py:1266-1276) turned into link-frame offsets with FK of (planning-row qpos, arms = refined row 0); link
          ee_left|ee_right = MJCF EE_BODY_L|R, finger links same name.  Row k: FK of (planning qpos, arms = refined row k).
          held spheres = server held query set (DX rule: body_cover lifted/body set, frame = EE body).
          d = composed field FIA.build_field (coarse -> fine), target-free layer for authorized rows only when not attached and
          exclusion mechanism has 'target_free_layer'.  class: held (query >= n_spheres) / target (|d - d_object| <= finest
          voxel; d_object = union of held spheres at the capture pose if attached, else nearest target point) / support (table
          plane z = held.support.plane.offset_m when recorded in the run, |d - s| <= 2 voxel) / obstacle.
          margin = esdf_margin (log, 0) ; obstacle rows max(margin, obstacle_margin 10 mm) ; destination margin (held rows whose
          nearest label is the destination) is NOT reproducible (labels not recorded) -> held rows margin 0 here.
          TO clearance = d - r - margin.  Group value = min over the group's spheres at row k.
          d_true = signed distance of the sphere centre to the hurdle geoms (analytic capsule / cylinder) - r  (sphere model, exact bar).
  Fidelity: (a) my d - r on robot spheres vs the record's clearance (8, S) (full-field diagnostic: main layer d - r - esdf_margin);
          (b) my TO clearance at the recorded worst row (step, query) vs to.metrics.max_violation_pair.clearance_m; (c) my
          min over all rows / queries vs to.metrics.clearance_m.
  Servo offline: ServoParams from the snapshot MJCF (14 arm joints, CTRL_HZ 15), ServoObserver over the run's chunks in order
          (q_now = planning-row qpos, exec_feedback of that planning row), then ServoModel.rollout(q, v, C rows of the chunk, tau):
          predicted end-of-row joints = predicted M(t).  Errors vs M(t): EE body position (mm) and max joint (deg).
          Prediction states: geometric distance per group and TO-sphere ESDF clearance at the predicted joints.
Window: rows from grasp_t; phase carry = grasp_t..place_t, return = after place_t (apple group only in carry).  Crossing window
          = rows of a phase where min over groups of min(C geom, M geom) <= 100 mm.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys, time
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p'))
import numpy as np, mujoco  # noqa: E402
import hold_w as HW  # noqa: E402
import dx_common as C  # noqa: E402
from AG3S.trajopt.experiments import fixed_input_ablation as FIA  # noqa: E402
from AG3S.trajopt.servo import ServoModel, ServoObserver, servo_params_from_xml  # noqa: E402

T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
ANALYSIS = {"V7g": T43 / "CT/raw/analysis_v7g.json", "V8": T43 / "V8/analysis/analysis.json", "V5": T43 / "W/analysis/analysis.json", "V9": T43 / "V9/analysis/analysis.json"}
SNAP = {"V7g": "${AG3S_PROJECT_ROOT}-t43v7", "V8": "${AG3S_PROJECT_ROOT}-t43v8", "V5": "${AG3S_PROJECT_ROOT}-t43w3", "V9": "${AG3S_PROJECT_ROOT}-t43v8"}
WIN_MM = 100.0
OBST_MARGIN = 0.010
LINK2BODY = {"ee_left": "EE_BODY_L", "ee_right": "EE_BODY_R"}


def lit(x):
    return C.lit(x)


def main():
    arm, out_path = sys.argv[1], pathlib.Path(sys.argv[2])
    runs = json.load(open(ANALYSIS[arm]))["runs"]
    keys = sys.argv[3:] or sorted(k for k in runs if f"{arm}/H/" in k)
    cl = HW.load_client()
    res = {"definitions": __doc__, "arm": arm, "client_snapshot": str(HW.HY), "runs": {}}
    servo_model = None
    for key in keys:
        t_start = time.time(); a = runs[key]; rd = pathlib.Path(a["run_dir"])
        man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = int(man["scene"]["episode_index"])
        z = np.load(rd / "traj.npz", allow_pickle=True); seed = int(z["policy_seed"])
        stt = rd.parent / f"{rd.name}.start"
        cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
        L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; mg = L["obstacle_manager"]
        hurdle = sorted(mg.collision_geom_ids)
        side = "right" if ep == 1834 else "left"; SD = side[0].upper(); fl = side[0]
        qL = np.asarray(L["idx"]["left_q"][:7], int); qR = np.asarray(L["idx"]["right_q"][:7], int)
        qadr = qL if side == "left" else qR; c0 = 0 if side == "left" else 8
        ee_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"EE_BODY_{SD}")
        coll = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
        geoms_of = lambda b: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b and coll(g)]
        fingers = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ee_finger_{fl}{i}") for i in (1, 2)]
        G = {"fingers": sum((geoms_of(b) for b in fingers), []), "palm": geoms_of(ee_b),
             "apple": [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom")]}
        apple_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple"); apple_q = int(m.jnt_qposadr[m.body_jntadr[apple_b]])
        TO_LINKS = {"fingers": [f"ee_finger_{fl}1", f"ee_finger_{fl}2"], "palm": [f"ee_{side}"]}
        ft = np.zeros(6)
        if servo_model is None:
            names = [f"left_arm_{j}" for j in range(7)] + [f"right_arm_{j}" for j in range(7)]
            sp = servo_params_from_xml(SNAP[arm] + "/robots/rby1/description/models/rby1a/mujoco/model_transport.xml", names, control_hz=15.0)
            servo_model = ServoModel(sp); res["servo_params"] = sp.summary()
        q14 = np.r_[qL, qR]

        def setq(q):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d)

        def with_arms(qfull, a16, move_apple):
            q = np.asarray(qfull, float).copy()
            if move_apple:
                setq(q); T0 = C.body_T(m, d, f"EE_BODY_{SD}")
            q[qL] = a16[0:7]; q[qR] = a16[8:15]
            if move_apple:
                setq(q); T1 = C.body_T(m, d, f"EE_BODY_{SD}")
                pa = q[apple_q:apple_q + 3]; Ta = np.eye(4); Ta[:3, :3] = C.qmat(q[apple_q + 3:apple_q + 7]); Ta[:3, 3] = pa
                Tn = T1 @ np.linalg.inv(T0) @ Ta; qa = np.zeros(4); mujoco.mju_mat2Quat(qa, Tn[:3, :3].reshape(-1))
                q[apple_q:apple_q + 3] = Tn[:3, 3]; q[apple_q + 3:apple_q + 7] = qa
            return q

        fj_adr = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in
                  ("gripper_finger_l1", "gripper_finger_l2", "gripper_finger_r1", "gripper_finger_r2")}

        def set_fingers(q, vals):
            q = np.asarray(q, float).copy()
            for n, v in vals.items():
                q[fj_adr[n]] = v
            return q

        def geo(q, groups):
            setq(q); out = {}
            for gname in groups:
                best = np.inf
                for g1 in G[gname]:
                    for g2 in hurdle:
                        best = min(best, float(mujoco.mj_geomDistance(m, d, int(g1), int(g2), 0.5, ft)))
                out[gname] = 1e3 * best
            return out

        def ee_pos(q):
            setq(q); return d.xpos[ee_b].copy()

        rows_f = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
        ctrl = {r["t_step"]: r for r in rows_f if r["kind"] == "control"}
        plan = {r["seq"]: r for r in rows_f if r["kind"] == "planning"}
        seg = lit(a.get("server_segment")); chunks = C.segment_chunks(seg) if seg and seg != "None" else {}
        st, outside, om, logp = C.server_settings(seg, rd / "manifest.json"); esdf_margin = float(st.esdf_margin)
        EA = np.asarray(z["executed_actions"], float); MQ = np.asarray(z["measured_qpos"], float); T = len(EA)
        g_t = a.get("grasp_t"); p_t = a.get("place_t")
        g_t = None if g_t in (None, "None") else int(g_t); p_t = None if p_t in (None, "None") else int(p_t)
        runrec = {"key": key, "side": side, "episode": ep, "seed": seed, "grasp_t": g_t, "place_t": p_t, "n_rows": T,
                  "esdf_margin_mm": 1e3 * esdf_margin, "obstacle_margin_mm_log": om, "server_log": logp,
                  "hurdle_geoms": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in hurdle],
                  "bar": None, "rows": [], "chunks": {}, "servo": {}}
        mujoco.mj_forward(m, d)
        gb = next(g for g in hurdle if "bar" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))
        runrec["bar"] = {"centre_world": d.geom_xpos[gb].tolist(), "axis_world": d.geom_xmat[gb].reshape(3, 3)[:, 2].tolist(),
                         "radius_mm": 1e3 * float(m.geom_size[gb][0]), "half_length_mm": 1e3 * float(m.geom_size[gb][1])}
        if g_t is None:
            res["runs"][key] = runrec; print(key, "no grasp", flush=True); continue

        # ---------------- servo observer over all chunks in order ----------------
        obs = ServoObserver(servo_model); pred_q = {}; servo_src = {}
        for seq in sorted(plan):
            pr = plan[seq]; t0 = int(pr["t_step"])
            qn = np.asarray(pr["qpos"], float)[q14]
            fb = pr.get("exec_feedback") or {}
            if not (isinstance(fb, dict) and fb.get("applied_arm") is not None) and seq in chunks:
                with np.load(chunks[seq], allow_pickle=True) as zz:
                    fb = (json.loads(str(zz["summary_json"])).get("exec_feedback") or {})
            if isinstance(fb, dict) and isinstance(fb.get("applied_arm"), str):
                fb = dict(fb); fb["applied_arm"] = lit(fb["applied_arm"])
            stt_ = obs.update(qn, fb, seq)
            rows_c = sorted(t for t, r in ctrl.items() if r["chunk_seq"] == seq)
            if not rows_c:
                continue
            U = np.stack([np.r_[EA[t, 0:7], EA[t, 8:15]] for t in rows_c], axis=1)
            ro = servo_model.rollout(stt_.q, stt_.v, U, stt_.tau)
            for j, t in enumerate(rows_c):
                pred_q[t] = ro["q"][:, j]; servo_src[t] = stt_.source

        # ---------------- per-chunk TO reconstruction cache ----------------
        zc = {}

        def chunk(seq):
            if seq in zc:
                return zc[seq]
            if seq not in chunks:
                zc[seq] = None; return None
            zz = np.load(chunks[seq], allow_pickle=True); s = json.loads(str(zz["summary_json"]))
            field = FIA.build_field(zz, outside)
            S = int(zz["sphere_radii"].shape[0]); links = list(lit(s.get("sphere_link_names")) or [])
            ref = np.asarray(zz["refined_chunk"], float); rfc = np.asarray(zz["reference_chunk"], float)
            qcap = np.asarray(plan[seq]["qpos"], float)
            # T29 finger path (gripper_state.opening_envelope): open[0] = measured, open[k>=1] = max(measured, cummax(clip(cmd[0..k])))
            fj = s.get("finger_joints") or {}; on = lit(fj.get("opening_norm")) or {}
            def fpath(chunk_arr):
                out = []
                for k_ in range(min(8, chunk_arr.shape[0])):
                    vals = {}
                    for hand, col, a1, a2 in (("left", 7, "gripper_finger_l1", "gripper_finger_l2"), ("right", 15, "gripper_finger_r1", "gripper_finger_r2")):
                        if on.get(hand) is None:
                            continue
                        meas = float(on[hand]); cmd = np.clip(chunk_arr[:8, col], 0.0, 1.0)
                        o = meas if k_ == 0 else max(meas, float(np.max(cmd[:k_ + 1])))
                        vals[a1] = -0.045 * o; vals[a2] = 0.045 * o
                    out.append(vals)
                return out
            fp_ref, fp_rfc = fpath(ref), fpath(rfc)
            # link-frame offsets from refined row 0
            q0 = with_arms(qcap, ref[0], False); q0 = set_fingers(q0, fp_ref[0] if fp_ref else {}); setq(q0); Tb = C.T_world_base(m, d)
            cw = (Tb[:3, :3] @ np.asarray(zz["sphere_centres"], float)[:S].T).T + Tb[:3, 3]
            local = np.zeros((S, 3)); body_ids = np.zeros(S, int)
            for i_, ln in enumerate(links):
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, LINK2BODY.get(ln, ln)); body_ids[i_] = b
                local[i_] = d.xmat[b].reshape(3, 3).T @ (cw[i_] - d.xpos[b])
            used, sets, flk, bc = C.held_sets(s)
            held = sets.get(used) if used else None
            g = s.get("grasp") or {}
            attached = held is not None and bool(held)
            exc = s.get("exclusion") or {}
            mask = None
            if (not attached and st.target_field_policy != "relax" and s.get("has_target") and "esdf_free_distance" in zz.files
                    and "target_free_layer" in (exc.get("mechanism") or [])):
                auth = set(exc.get("authorized_links") or [])
                mask = np.asarray([ln in auth for ln in links], bool)
                if not mask.any():
                    mask = None
            # d_object
            d_obj_fn = None
            if attached:
                setq(qcap); Te = C.body_T(m, d, f"EE_BODY_{SD}"); Hc, Hr = C.place(Te, held); Hb = C.to_base(C.T_world_base(m, d), Hc)
                d_obj_fn = lambda P, Hb=Hb, Hr=Hr: (np.linalg.norm(P[:, None, :] - Hb[None], axis=2) - Hr[None]).min(1)
            elif s.get("has_target") and "target_points" in zz.files and len(zz["target_points"]):
                from scipy.spatial import cKDTree
                tree = cKDTree(np.asarray(zz["target_points"], float)); d_obj_fn = lambda P, tree=tree: tree.query(P)[0]
            plane_z = None
            hs = ((((g.get("attached") or {}) if isinstance(g.get("attached"), dict) else {}).get("held") or {}).get("support") or {})
            if isinstance(hs, dict) and isinstance(hs.get("plane"), dict) and hs["plane"].get("offset_m") is not None:
                plane_z = float(hs["plane"]["offset_m"])
            to = s.get("to") or {}; met = to.get("metrics") or {}; mv = lit(met.get("max_violation_pair")) or {}
            rec = {"seq": seq, "t0": int(plan[seq]["t_step"]), "status": to.get("status"), "iterations": to.get("iterations"),
                   "sqp_iterations": to.get("sqp_iterations"), "max_iterations": to.get("max_iterations"),
                   "max_iterations_hit": to.get("max_iterations_hit"), "time_budget_hit": to.get("time_budget_hit"),
                   "time_budget_ms": to.get("time_budget_ms"), "solve_ms": to.get("solve_ms"), "qp_iterations": to.get("qp_iterations"),
                   "notes": to.get("notes"), "clearance_mm": None if met.get("clearance_m") is None else 1e3 * float(met["clearance_m"]),
                   "initial_violation_mm": None if met.get("initial_violation_m") is None else 1e3 * float(met["initial_violation_m"]),
                   "returned": met.get("returned"), "returned_index": met.get("returned_index"), "n_accepted": met.get("n_accepted"),
                   "n_candidates": met.get("n_candidates"), "qp_failures": met.get("qp_failures"), "candidates": lit(met.get("candidates")),
                   "geometry_certified": met.get("geometry_certified"), "planned_steps": met.get("planned_steps"),
                   "limit_overshoot": lit(met.get("limit_overshoot")), "projection": met.get("projection"),
                   "worst": {"link": mv.get("link"), "query": mv.get("query"), "step": mv.get("step"), "class": mv.get("obstacle_margin_class"),
                             "tier": mv.get("tier"), "clearance_mm": None if mv.get("clearance_m") is None else 1e3 * float(mv["clearance_m"]),
                             "point_m": mv.get("point_m")},
                   "verdict": [(x.get("kind"), x.get("action")) for x in ((s.get("verdict") or {}).get("reasons") or [])],
                   "gate_action": (plan[seq].get("gate") or {}).get("action") if isinstance(plan[seq].get("gate"), dict) else plan[seq].get("gate"),
                   "attached": attached, "held_set": used, "n_held": 0 if not held else len(held), "mask_rows": 0 if mask is None else int(mask.sum()),
                   "plane_z": plane_z, "n_spheres": S, "grasp_state": g.get("state")}
            lm = FIA.contact_policy(st, links, has_target=True, has_free=False, attached=False)[0]
            rec["link_margin_mm_by_link"] = {ln: 1e3 * float(v) for ln, v in zip(links, lm)} if lm is not None else None
            zc[seq] = dict(z=zz, s=s, field=field, S=S, links=np.asarray(links), local=local, body_ids=body_ids, held=held, link_margin=lm, fp_ref=fp_ref, fp_rfc=fp_rfc,
                           mask=mask, d_obj=d_obj_fn, plane_z=plane_z, ref=ref, rfc=rfc, qcap=qcap, rec=rec, rows={})
            return zc[seq]

        voxel_fine = None

        def to_rows(cc, a16, qbase=None, fingers=None):
            """TO rows at the configuration (qbase or planning qpos, arms = a16, finger joints override). -> per-query arrays."""
            nonlocal voxel_fine
            q = with_arms(cc["qcap"] if qbase is None else qbase, a16, False); q = set_fingers(q, fingers or {}); setq(q); Tb = C.T_world_base(m, d)
            S = cc["S"]; R = d.xmat[cc["body_ids"]].reshape(-1, 3, 3); p = d.xpos[cc["body_ids"]]
            cw = np.einsum("nij,nj->ni", R, cc["local"]) + p
            radii = np.asarray(cc["z"]["sphere_radii"], float)[:S]
            if cc["held"]:
                Te = C.body_T(m, d, f"EE_BODY_{SD}"); Hc, Hr = C.place(Te, cc["held"]); cw = np.r_[cw, Hc]; radii = np.r_[radii, Hr]
            Pb = C.to_base(Tb, cw); fld = cc["field"]
            dd = np.asarray(fld.distance(Pb), float); dd_main = dd.copy()
            if cc["mask"] is not None:
                msk = np.r_[cc["mask"], np.zeros(len(Pb) - S, bool)]
                if msk.any():
                    dd[msk] = np.asarray(fld.target_free_distance(Pb[msk]), float)
            if voxel_fine is None:
                voxel_fine = float(fld.grid.voxel_size)
            cls = np.array(["obstacle"] * len(Pb), dtype=object)
            is_obj = np.zeros(len(Pb), bool)
            if cc["d_obj"] is not None:
                is_obj = np.abs(dd - cc["d_obj"](Pb)) <= voxel_fine; is_obj[S:] = False
                cls[is_obj] = "target"
            if cc["plane_z"] is not None:
                sz = Pb[:, 2] - cc["plane_z"]; cls[(np.abs(sz - dd) <= 2 * voxel_fine) & (cls == "obstacle")] = "support"
            cls[S:] = "held"
            margin = np.full(len(Pb), esdf_margin)
            # E1 rule (linearize.py manipulated_link_margin): target rows of non-exempt robot spheres take the per-link margin
            exempt = np.zeros(len(Pb), bool)
            if cc["mask"] is not None:
                exempt[:S] = cc["mask"]
            sel = is_obj & ~exempt
            if cc["d_obj"] is not None and sel.any():
                margin[:S][sel[:S]] = np.asarray(cc["link_margin"], float)[sel[:S]]
            margin[cls == "obstacle"] = np.maximum(margin[cls == "obstacle"], OBST_MARGIN)
            clear = dd - radii - margin
            dtrue = C.hurdle_sd(m, d, hurdle, cw) - radii
            return {"clear": clear, "d_minus_r": dd - radii, "d_main_minus_r": dd_main - radii, "margin": margin, "cls": cls, "dtrue": dtrue, "S": S}

        def grp(tr, cc, gname):
            if gname == "apple":
                idx = np.arange(tr["S"], len(tr["clear"]))
            else:
                idx = np.flatnonzero(np.isin(cc["links"], TO_LINKS[gname]))
            if not len(idx):
                return None
            j = idx[int(np.argmin(tr["clear"][idx]))]; jt = idx[int(np.argmin(tr["dtrue"][idx]))]
            return {"to_clear_mm": 1e3 * float(tr["clear"][j]), "margin_mm": 1e3 * float(tr["margin"][j]), "cls": str(tr["cls"][j]),
                    "d_minus_r_mm": 1e3 * float(tr["d_minus_r"][j]), "query": int(j),
                    "true_sphere_mm": 1e3 * float(tr["dtrue"][jt]), "esdf_at_true_nearest_mm": 1e3 * float(tr["d_minus_r"][jt]),
                    "true_nearest_query": int(jt), "true_nearest_margin_mm": 1e3 * float(tr["margin"][jt]),
                    "true_nearest_cls": str(tr["cls"][jt])}

        def chunk_rows(seq):
            cc = chunk(seq)
            if cc is None:
                return None
            if cc["rows"]:
                return cc
            n_steps = min(8, cc["ref"].shape[0]); allc = []
            fid_a = []
            for k in range(n_steps):
                tr = to_rows(cc, cc["ref"][k], fingers=cc["fp_ref"][k]); trr = to_rows(cc, cc["rfc"][k], fingers=cc["fp_rfc"][k])
                cc["rows"][k] = {"refined": {g_: grp(tr, cc, g_) for g_ in ("fingers", "palm", "apple")},
                                 "reference": {g_: grp(trr, cc, g_) for g_ in ("fingers", "palm", "apple")}}
                allc.append(tr["clear"])
                w0 = cc["rec"]["worst"]
                if w0.get("step") is not None and int(w0["step"]) == k and w0.get("query") is not None and int(w0["query"]) < len(tr["clear"]):
                    qq = int(w0["query"])
                    cc["rec"]["mine_at_rec_worst_detail"] = {"d_minus_r_mm": 1e3 * float(tr["d_minus_r"][qq]), "cls": str(tr["cls"][qq]),
                                                            "margin_mm": 1e3 * float(tr["margin"][qq]), "true_bar_mm": 1e3 * float(tr["dtrue"][qq]),
                                                            "implied_margin_mm": None if w0.get("clearance_mm") is None else 1e3 * float(tr["d_minus_r"][qq]) - w0["clearance_mm"]}
                rec_cl = np.asarray(cc["z"]["clearance"], float)
                if rec_cl.ndim == 2 and k < rec_cl.shape[0] and rec_cl.shape[1] >= tr["S"]:
                    dif_ = np.abs((tr["d_main_minus_r"][:tr["S"]] - esdf_margin) - rec_cl[k, :tr["S"]]); fid_a.append(float(dif_.max()))
                    jb = int(dif_.argmax()); cc["rec"].setdefault("fid_worst_by_row", []).append(
                        [k, jb, str(cc["links"][jb]), 1e3 * float(dif_[jb]), 1e3 * float(rec_cl[k, jb]), 1e3 * float(tr["d_main_minus_r"][jb])])
            allc = np.asarray(allc); rec = cc["rec"]; w = rec["worst"]
            rec["fid_record_clearance_max_abs_mm"] = 1e3 * max(fid_a) if fid_a else None
            rec["mine_min_clear_mm"] = 1e3 * float(allc.min())
            ks, qs = np.unravel_index(int(allc.argmin()), allc.shape); rec["mine_min_step_query"] = [int(ks), int(qs)]
            if w.get("step") is not None and w.get("query") is not None and int(w["step"]) < allc.shape[0] and int(w["query"]) < allc.shape[1]:
                rec["mine_at_rec_worst_mm"] = 1e3 * float(allc[int(w["step"]), int(w["query"])])
            # max_step saturation over rows 0..7 (anchor row 0 vs q_now, then step differences), rad
            qn = np.r_[cc["qcap"][qL], cc["qcap"][qR]]; Q = np.stack([np.r_[cc["ref"][k, 0:7], cc["ref"][k, 8:15]] for k in range(n_steps)])
            steps = np.abs(np.diff(np.r_[qn[None], Q], axis=0)); rec["max_abs_step_rad_by_row"] = steps.max(1).tolist()
            Qr = np.stack([np.r_[cc["rfc"][k, 0:7], cc["rfc"][k, 8:15]] for k in range(n_steps)])
            rec["refined_minus_reference_rad_max_by_row"] = np.abs(Q - Qr).max(1).tolist()
            return cc

        # ---------------- per row ----------------
        rows = []
        for t in range(g_t, T):
            r = ctrl[t]; seq = int(r["chunk_seq"]); k = int(r["step_in_chunk"]); ex = bool(r.get("executed", True))
            phase = "carry" if (p_t is None or t <= p_t) else "return"
            groups = ("fingers", "palm", "apple") if phase == "carry" else ("fingers", "palm")
            q_pre = np.asarray(r["qpos"], float)
            if t + 1 in ctrl:
                qM = np.asarray(ctrl[t + 1]["qpos"], float); msrc = "row t+1 qpos"
            else:
                qM = q_pre.copy(); qM[qL] = MQ[t, 0:7]; qM[qR] = MQ[t, 8:15]; msrc = "row t qpos + measured arms"
            qC = with_arms(qM, EA[t], True)
            row = {"t": t, "seq": seq, "k": k, "executed": ex, "phase": phase, "M_src": msrc,
                   "geo_C": geo(qC, groups), "geo_M": geo(qM, groups),
                   "ee_C": ee_pos(qC).tolist(), "ee_M": ee_pos(qM).tolist()}
            cc = chunk(seq)
            if cc is not None and k < cc["ref"].shape[0]:
                row["P_minus_C_rad"] = float(np.abs(np.r_[cc["ref"][k, 0:7] - EA[t, 0:7], cc["ref"][k, 8:15] - EA[t, 8:15]]).max())
            if t in pred_q:
                a16 = EA[t].copy(); a16[0:7] = pred_q[t][:7]; a16[8:15] = pred_q[t][7:]
                qS = with_arms(qM, a16, True)
                row["servo_src"] = servo_src[t]; row["pred_q14"] = pred_q[t].tolist()
                row["ee_S"] = ee_pos(qS).tolist()
                row["servo_err_mm"] = 1e3 * float(np.linalg.norm(np.asarray(row["ee_S"]) - np.asarray(row["ee_M"])))
                row["cmd_err_mm"] = 1e3 * float(np.linalg.norm(np.asarray(row["ee_C"]) - np.asarray(row["ee_M"])))
                jm = np.r_[qM[qL], qM[qR]]; row["servo_err_joint_deg_max"] = float(np.degrees(np.abs(pred_q[t] - jm)[0:7] if side == "left" else np.abs(pred_q[t] - jm)[7:]).max())
                row["cmd_err_joint_deg_max"] = float(np.degrees(np.abs(np.r_[EA[t, 0:7], EA[t, 8:15]] - jm)[0:7] if side == "left" else np.abs(np.r_[EA[t, 0:7], EA[t, 8:15]] - jm)[7:]).max())
                row["_qS"] = qS
            rows.append(row)
        # window rows and TO view
        for row in rows:
            dm = min(min(row["geo_C"].values()), min(row["geo_M"].values()))
            row["in_window"] = bool(dm <= WIN_MM)
        need = sorted({row["seq"] for row in rows if row["in_window"]})
        for seq in need:
            chunk_rows(seq)
        for row in rows:
            qS = row.pop("_qS", None)
            if not row["in_window"]:
                continue
            cc = zc.get(row["seq"])
            if cc is None or not cc["rows"] or row["k"] not in cc["rows"]:
                row["to"] = None; continue
            row["to"] = cc["rows"][row["k"]]["refined"]; row["to_reference"] = cc["rows"][row["k"]]["reference"]
            # TO spheres at the measured and the servo-predicted configuration (same chunk field)
            qMf = np.asarray(ctrl[row["t"] + 1]["qpos"], float) if row["t"] + 1 in ctrl else None
            if qMf is not None:
                a16 = EA[row["t"]].copy(); a16[0:7] = qMf[qL]; a16[8:15] = qMf[qR]
                trm = to_rows(cc, a16, qbase=qMf); row["to_at_M"] = {g_: grp(trm, cc, g_) for g_ in ("fingers", "palm", "apple")}
            if qS is not None:
                a16 = EA[row["t"]].copy(); a16[0:7] = qS[qL]; a16[8:15] = qS[qR]
                trs = to_rows(cc, a16, qbase=qMf if qMf is not None else cc["qcap"]); row["to_at_S"] = {g_: grp(trs, cc, g_) for g_ in ("fingers", "palm", "apple")}
                row["geo_S"] = geo(qS, ("fingers", "palm", "apple") if row["phase"] == "carry" else ("fingers", "palm"))
            row["binding"] = cc["rec"]["worst"]
        runrec["rows"] = rows
        runrec["chunks"] = {str(sq): zc[sq]["rec"] for sq in need if zc.get(sq) is not None}
        runrec["voxel_fine_m"] = voxel_fine
        runrec["sec"] = round(time.time() - t_start, 1)
        res["runs"][key] = runrec
        nwin = sum(r["in_window"] for r in rows)
        print(key, "rows", len(rows), "window", nwin, "chunks", len(need), "sec", runrec["sec"], flush=True)
        out_path.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    out_path.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
