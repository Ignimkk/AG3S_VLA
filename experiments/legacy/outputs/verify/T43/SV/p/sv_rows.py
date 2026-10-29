"""T43 SV — per-chunk TO row reconstruction at three states (plan P, servo prediction S, measured M) for the approach, carry and
return phases. (verifier, CPU only, recorded data only, no new runs.)  Machinery = T43 HX hx_measure.py (same field, spheres, classes).

usage: T43W_SNAP=<client snapshot> PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8 sv_rows.py ARM OUT_JSON [KEY_SUBSTR ...]
  ARM = V7g (CT/raw/analysis_v7g.json, snapshot work-t43v7) | V8 (V8/analysis) | V9 (V9/analysis) | V3, V4 (W/analysis, snapshot work-t43w)

Phases (server grasp.state of the chunk; observable on the server):
  approach = chunks before the first attached chunk with t <= first closing t + 8 (or <= 200 without closing)
  carry    = attached chunks (state held); sub-phase crate_entry = attached and the apple (planning-row object_poses) over the crate
             footprint (agg_v9 over_crate rule: |x| < 77 mm, |y| < 137 mm in the crate frame), else carry
  return   = chunks from the first chunk whose grasp.state == 'placed' (post-PLACED) to the end of the run
States per chunk row k = 0..7 (control row t0+k):
  P = refined_chunk[k] (the plan = the command on executed rows), finger joints = T29 opening envelope (hx rule)
  S = servo prediction of the end of row k: ServoObserver over the run's chunks in order (planning-row qpos + exec_feedback), then
      ServoModel.rollout(q, v, refined rows 0..7, tau) (AG3S/trajopt/servo.py of the snapshot) -- what the TO's servo rows evaluate
      for this plan; fingers as P
  M = measured post state of row t0+k = control row t0+k+1 qpos (arms and fingers measured)
TO rows (linearize._esdf_clearance reconstruction, hx_measure to_rows): robot spheres from the record (link-frame offsets), held spheres
  (server held query set) rigid with the EE body; d = composed field (FIA.build_field) or the target-free layer for authorized rows
  (mask: not attached, policy exclude-authorized, has_target, target_free_layer); class held | target | support | obstacle; margin =
  esdf_margin (0) | E1 per-link | obstacle max(., 10 mm).  Stored: d - r (no margin) and class per sphere, reduced below.
Truth label of a row: nearest MuJoCo collision surface to the sphere centre among hurdle (obstacle manager collision geoms), crate,
  table_top, apple (ellipsoid, radial approx.), other fruit; label = that set if |d_esdf - d_true| <= 10 mm, else 'unexplained'
  (the field sees a surface no scene geom explains: residual / ghost voxels).
Reduction per (chunk, state, row k): for every (group, class, label) cell the min of d - r (mm) over its spheres, with
  group = finger (links ee_finger_*) | palm (ee_left / ee_right) | robot_other | held.  Also per (state,row): min true hurdle distance
  minus r over hand spheres (finger + palm) and over held spheres, and finger-sphere table clearance (z - table_top_z - r, truth).
Chunk record: seq, t0, phase, grasp.state, executed rows, to.status, recorded to.metrics.clearance_m, recorded worst row, follow_to.
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
ANALYSIS = {"V7g": T43 / "CT/raw/analysis_v7g.json", "V8": T43 / "V8/analysis/analysis.json", "V9": T43 / "V9/analysis/analysis.json",
            "V3": T43 / "W/analysis/analysis.json", "V4": T43 / "W/analysis/analysis.json"}
SNAP = {"V7g": "${AG3S_PROJECT_ROOT}-t43v7", "V8": "${AG3S_PROJECT_ROOT}-t43v8", "V9": "${AG3S_PROJECT_ROOT}-t43v8", "V3": "${AG3S_PROJECT_ROOT}-t43w", "V4": "${AG3S_PROJECT_ROOT}-t43w"}
OBST = 0.010
LABEL_TOL = 0.010
LINK2BODY = {"ee_left": "EE_BODY_L", "ee_right": "EE_BODY_R"}
FRUITS = ("banana", "orange", "pear")
LABELS = ("hurdle", "crate", "table", "apple", "fruit_other", "unexplained")


def lit(x):
    return C.lit(x)


def ellipsoid_sd(m, d, g, P):
    c = d.geom_xpos[g]; R = d.geom_xmat[g].reshape(3, 3); s = m.geom_size[g][:3]
    q = (np.atleast_2d(P) - c) @ R; f = np.linalg.norm(q / s, axis=1); r = np.linalg.norm(q, axis=1)
    return r * (1.0 - 1.0 / np.maximum(f, 1e-9))


def over_crate(p):
    op = p.get("object_poses") or {}
    if "crate" not in op or "apple" not in op:
        return False
    loc = C.qmat(op["crate"]["quat"]).T @ (np.asarray(op["apple"]["pos"], float) - np.asarray(op["crate"]["pos"], float))
    return bool(abs(loc[0]) < 0.077 and abs(loc[1]) < 0.137)


def main():
    arm, out_path = sys.argv[1], pathlib.Path(sys.argv[2])
    runs = json.load(open(ANALYSIS[arm]))["runs"]
    keys = sorted(k for k in runs if k.startswith(arm + "/"))
    if len(sys.argv) > 3:
        keys = [k for k in keys if any(x in k for x in sys.argv[3:])]
    cl = HW.load_client()
    res = {"definitions": __doc__, "arm": arm, "client_snapshot": str(HW.HY), "runs": {}}
    names = [f"left_arm_{j}" for j in range(7)] + [f"right_arm_{j}" for j in range(7)]
    sp = servo_params_from_xml(SNAP[arm] + "/robots/rby1/description/models/rby1a/mujoco/model_transport.xml", names, control_hz=15.0)
    servo_model = ServoModel(sp)
    for key in keys:
        t_start = time.time()
        try:
            rr = one_run(cl, key, runs[key], servo_model)
        except Exception as ex:  # keep going, record the failure
            import traceback
            rr = {"error": repr(ex), "trace": traceback.format_exc()[-2000:]}
        rr["sec"] = round(time.time() - t_start, 1)
        res["runs"][key] = rr
        print(key, "chunks", len(rr.get("chunks", [])), "sec", rr["sec"], rr.get("error", ""), flush=True)
        out_path.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    out_path.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def one_run(cl, key, a, servo_model):
    rd = pathlib.Path(a["run_dir"]); man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile")
    ep = int(man["scene"]["episode_index"]); z = np.load(rd / "traj.npz", allow_pickle=True); seed = int(z["policy_seed"])
    stt = rd.parent / f"{rd.name}.start"
    cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
    L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; mg = L.get("obstacle_manager")
    hurdle = sorted(mg.collision_geom_ids) if mg is not None else []
    side = "right" if ep == 1834 else "left"; SD = side[0].upper()
    qL = np.asarray(L["idx"]["left_q"][:7], int); qR = np.asarray(L["idx"]["right_q"][:7], int); q14 = np.r_[qL, qR]
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    coll = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
    crate = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid("crate") and coll(g)]
    table_g = gid("table_top"); apple_g = gid("apple_geom")
    fruit_g = [g for g in range(m.ngeom) if coll(g) and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "") in FRUITS]
    apple_b = bid("apple"); apple_q = int(m.jnt_qposadr[m.body_jntadr[apple_b]])
    fj_adr = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in
              ("gripper_finger_l1", "gripper_finger_l2", "gripper_finger_r1", "gripper_finger_r2")}

    def setq(q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)

    def with_arms(qfull, a14):
        q = np.asarray(qfull, float).copy(); q[qL] = a14[0:7]; q[qR] = a14[7:14]; return q

    def set_fingers(q, vals):
        q = np.asarray(q, float).copy()
        for n_, v in vals.items():
            q[fj_adr[n_]] = v
        return q

    rows_f = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
    ctrl = {r["t_step"]: r for r in rows_f if r["kind"] == "control"}
    plan = {r["seq"]: r for r in rows_f if r["kind"] == "planning"}
    seg = lit(a.get("server_segment")); chunks = C.segment_chunks(seg) if seg and seg != "None" else {}
    if not chunks:
        return {"error": "no server segment", "chunks": []}
    st, outside, om, logp = C.server_settings(seg, rd / "manifest.json"); esdf_margin = float(st.esdf_margin)
    sc = {c["seq"]: c for c in (a.get("server_chunks") or [])}
    seqs = sorted(s_ for s_ in plan if s_ in chunks)
    att_seq = next((s_ for s_ in seqs if (sc.get(s_) or {}).get("attached")), None)
    close_t = next((sc[s_]["t"] for s_ in seqs if (sc.get(s_) or {}).get("grasp_state") == "closing"), None)
    placed_seq = next((s_ for s_ in seqs if (sc.get(s_) or {}).get("grasp_state") == "placed"), None)

    def phase(s_):
        c = sc.get(s_) or {}; t = int(plan[s_]["t_step"])
        if placed_seq is not None and s_ >= placed_seq:
            return "return"
        if c.get("attached"):
            return "crate_entry" if over_crate(plan[s_]) else "carry"
        if att_seq is None or s_ < att_seq:
            lim = (close_t + 8) if close_t is not None else 200
            return "approach" if t <= lim else None
        return None

    # servo observer in order over all chunks (the server's own observer sees every chunk)
    obs = ServoObserver(servo_model); sstate = {}
    for s_ in seqs:
        pr = plan[s_]; qn = np.asarray(pr["qpos"], float)[q14]; fb = pr.get("exec_feedback") or {}
        if not (isinstance(fb, dict) and fb.get("applied_arm") is not None):
            with np.load(chunks[s_], allow_pickle=True) as zz:
                fb = (json.loads(str(zz["summary_json"])).get("exec_feedback") or {})
        if isinstance(fb, dict) and isinstance(fb.get("applied_arm"), str):
            fb = dict(fb); fb["applied_arm"] = lit(fb["applied_arm"])
        stt_ = obs.update(qn, fb, s_); sstate[s_] = (stt_.q.copy(), stt_.v.copy(), np.asarray(stt_.tau).copy(), stt_.source)

    mujoco.mj_forward(m, d)
    table_top_z = float(d.geom_xpos[table_g][2] + m.geom_size[table_g][2])
    out_chunks = []
    voxel_fine = None
    for s_ in seqs:
        ph = phase(s_)
        if ph is None:
            continue
        zz = np.load(chunks[s_], allow_pickle=True); s = json.loads(str(zz["summary_json"]))
        field = FIA.build_field(zz, outside)
        S = int(zz["sphere_radii"].shape[0]); links = list(lit(s.get("sphere_link_names")) or [])
        ref = np.asarray(zz["refined_chunk"], float); qcap = np.asarray(plan[s_]["qpos"], float)
        fj = s.get("finger_joints") or {}; on = lit(fj.get("opening_norm")) or {}

        def fpath(arr):
            out = []
            for k_ in range(min(8, arr.shape[0])):
                vals = {}
                for hand, col, a1, a2 in (("left", 7, "gripper_finger_l1", "gripper_finger_l2"), ("right", 15, "gripper_finger_r1", "gripper_finger_r2")):
                    if on.get(hand) is None:
                        continue
                    meas = float(on[hand]); cmd = np.clip(arr[:8, col], 0.0, 1.0)
                    o = meas if k_ == 0 else max(meas, float(np.max(cmd[:k_ + 1])))
                    vals[a1] = -0.045 * o; vals[a2] = 0.045 * o
                out.append(vals)
            return out
        fp = fpath(ref)
        a0 = np.r_[ref[0, 0:7], ref[0, 8:15]]
        q0 = set_fingers(with_arms(qcap, a0), fp[0] if fp else {}); setq(q0); Tb = C.T_world_base(m, d)
        cw = (Tb[:3, :3] @ np.asarray(zz["sphere_centres"], float)[:S].T).T + Tb[:3, 3]
        local = np.zeros((S, 3)); body_ids = np.zeros(S, int)
        for i_, ln in enumerate(links):
            b = bid(LINK2BODY.get(ln, ln)); body_ids[i_] = b
            local[i_] = d.xmat[b].reshape(3, 3).T @ (cw[i_] - d.xpos[b])
        used, sets, flk, bc = C.held_sets(s); held = sets.get(used) if used else None
        attached = held is not None and bool(held)
        exc = s.get("exclusion") or {}
        mask = None
        if (not attached and st.target_field_policy != "relax" and s.get("has_target") and "esdf_free_distance" in zz.files
                and "target_free_layer" in (exc.get("mechanism") or [])):
            auth = set(exc.get("authorized_links") or [])
            mask = np.asarray([ln in auth for ln in links], bool)
            if not mask.any():
                mask = None
        d_obj_fn = None
        if attached:
            setq(qcap); Te = C.body_T(m, d, f"EE_BODY_{SD}"); Hc, Hr = C.place(Te, held); Hb = C.to_base(C.T_world_base(m, d), Hc)
            d_obj_fn = lambda P, Hb=Hb, Hr=Hr: (np.linalg.norm(P[:, None, :] - Hb[None], axis=2) - Hr[None]).min(1)
        elif s.get("has_target") and "target_points" in zz.files and len(zz["target_points"]):
            from scipy.spatial import cKDTree
            tree = cKDTree(np.asarray(zz["target_points"], float)); d_obj_fn = lambda P, tree=tree: tree.query(P)[0]
        g = s.get("grasp") or {}
        plane_z = None
        hs = ((((g.get("attached") or {}) if isinstance(g.get("attached"), dict) else {}).get("held") or {}).get("support") or {})
        if isinstance(hs, dict) and isinstance(hs.get("plane"), dict) and hs["plane"].get("offset_m") is not None:
            plane_z = float(hs["plane"]["offset_m"])
        if plane_z is None:
            est = lit((s.get("esdf_stats") or {}).get("target_free")) or {}
            ball = est.get("ball") if isinstance(est, dict) else None
            if isinstance(ball, dict) and ball.get("support_plane_z_m") is not None:
                plane_z = float(ball["support_plane_z_m"])
        lm = FIA.contact_policy(st, links, has_target=True, has_free=False, attached=False)[0]
        radii_r = np.asarray(zz["sphere_radii"], float)[:S]
        lk = np.asarray(links)
        grp_r = np.where(np.char.startswith(lk.astype(str), "ee_finger"), "finger", np.where(np.isin(lk, ["ee_left", "ee_right"]), "palm", "robot_other"))
        if voxel_fine is None:
            voxel_fine = float(field.grid.voxel_size)

        def rows_at(q):
            setq(q); Tb_ = C.T_world_base(m, d)
            R = d.xmat[body_ids].reshape(-1, 3, 3); p = d.xpos[body_ids]
            cwl = np.einsum("nij,nj->ni", R, local) + p; radii = radii_r.copy(); grp = grp_r.copy()
            if held:
                Te_ = C.body_T(m, d, f"EE_BODY_{SD}"); Hc_, Hr_ = C.place(Te_, held); cwl = np.r_[cwl, Hc_]; radii = np.r_[radii, Hr_]
                grp = np.r_[grp, np.array(["held"] * len(Hr_))]
            Pb = C.to_base(Tb_, cwl)
            dd = np.asarray(field.distance(Pb), float)
            if mask is not None:
                msk = np.r_[mask, np.zeros(len(Pb) - S, bool)]
                if msk.any():
                    dd[msk] = np.asarray(field.target_free_distance(Pb[msk]), float)
            cls = np.array(["obstacle"] * len(Pb), dtype=object)
            if d_obj_fn is not None:
                io = np.abs(dd - d_obj_fn(Pb)) <= voxel_fine; io[S:] = False; cls[io] = "target"
            if plane_z is not None:
                sz = Pb[:, 2] - plane_z; cls[(np.abs(sz - dd) <= 2 * voxel_fine) & (cls == "obstacle")] = "support"
            cls[S:] = "held"
            tv = {"crate": np.min([C.geom_point_sd(m, d, g_, cwl) for g_ in crate], axis=0),
                  "table": C.geom_point_sd(m, d, table_g, cwl), "apple": ellipsoid_sd(m, d, apple_g, cwl)}
            if fruit_g:
                tv["fruit_other"] = np.min([ellipsoid_sd(m, d, g_, cwl) for g_ in fruit_g], axis=0)
            if hurdle:
                tv["hurdle"] = C.hurdle_sd(m, d, hurdle, cwl)
            names_ = list(tv); T_ = np.stack([tv[n_] for n_ in names_], 0)
            j = np.argmin(T_, axis=0); dtmin = T_[j, np.arange(T_.shape[1])]
            lab = np.array([names_[i] for i in j], dtype=object)
            lab[np.abs(dd - dtmin) > LABEL_TOL] = "unexplained"
            dmr = dd - radii
            red = {}
            for gname in ("finger", "palm", "robot_other", "held"):
                gs = grp == gname
                if not gs.any():
                    continue
                for cname in ("target", "support", "obstacle", "held"):
                    sel = gs & (cls == cname)
                    if not sel.any():
                        continue
                    for lname in LABELS:
                        sl = sel & (lab == lname)
                        if sl.any():
                            red[f"{gname}|{cname}|{lname}"] = round(1e3 * float(dmr[sl].min()), 2)
            hand = (grp == "finger") | (grp == "palm")
            extra = {}
            if hurdle:
                hb = tv["hurdle"] - radii
                extra["hand_true_hurdle_mm"] = round(1e3 * float(hb[hand].min()), 2) if hand.any() else None
                extra["held_true_hurdle_mm"] = round(1e3 * float(hb[grp == "held"].min()), 2) if (grp == "held").any() else None
            fs = grp == "finger"
            extra["finger_table_mm"] = round(1e3 * float((cwl[fs, 2] - table_top_z - radii[fs]).min()), 2) if fs.any() else None
            return red, extra

        c = sc.get(s_) or {}
        to = s.get("to") or {}; met = to.get("metrics") or {}; mv = lit(met.get("max_violation_pair")) or {}
        t0 = int(plan[s_]["t_step"]); n_steps = min(8, ref.shape[0])
        qs, vs, taus, ssrc = sstate[s_]
        U = np.stack([np.r_[ref[k_, 0:7], ref[k_, 8:15]] for k_ in range(n_steps)], axis=1)
        ro = servo_model.rollout(qs, vs, U, taus)
        rec = {"seq": s_, "t0": t0, "phase": ph, "grasp_state": c.get("grasp_state"), "to_status": to.get("status"),
               "clearance_mm": None if met.get("clearance_m") is None else 1e3 * float(met["clearance_m"]),
               "worst": {"link": mv.get("link"), "query": mv.get("query"), "class": mv.get("obstacle_margin_class"), "step": mv.get("step"),
                         "obstacle": mv.get("obstacle")},
               "follow_to": (c.get("follow_to") or {}).get("decision") if isinstance(c.get("follow_to"), dict) else None,
               "hold": bool(any(not bool((ctrl.get(t0 + k_) or {}).get("executed", True)) for k_ in range(n_steps))),
               "executed": [bool((ctrl.get(t0 + k_) or {}).get("executed", True)) for k_ in range(n_steps)],
               "attached": attached, "mask": None if mask is None else int(mask.sum()), "plane_z": plane_z, "servo_source": ssrc,
               "P": [], "S": [], "M": []}
        for k_ in range(n_steps):
            qP = set_fingers(with_arms(qcap, U[:, k_]), fp[k_] if k_ < len(fp) else {})
            rec["P"].append(rows_at(qP))
            qS = set_fingers(with_arms(qcap, ro["q"][:, k_]), fp[k_] if k_ < len(fp) else {})
            rec["S"].append(rows_at(qS))
            if (t0 + k_ + 1) in ctrl:
                qM = np.asarray(ctrl[t0 + k_ + 1]["qpos"], float)
                if held:   # held spheres ride the EE body; the measured qpos already has the measured fingers
                    pass
                rec["M"].append(rows_at(qM))
            else:
                rec["M"].append(None)
        out_chunks.append(rec)
    return {"side": side, "episode": ep, "seed": seed, "stage": a.get("stage"), "grasp": a.get("grasp"), "success": a.get("success"),
            "grasp_t": a.get("grasp_t"), "place_t": a.get("place_t"), "attach_seq": att_seq, "placed_seq": placed_seq,
            "first_closing_t": close_t, "table_top_z_world": table_top_z, "voxel_fine_m": voxel_fine, "esdf_margin_mm": 1e3 * esdf_margin,
            "obstacle_margin_mm_log": om, "n_hurdle_geoms": len(hurdle), "chunks": out_chunks}


if __name__ == "__main__":
    main()
