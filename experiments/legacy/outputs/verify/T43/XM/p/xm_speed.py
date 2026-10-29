"""T43 XM part 2 — speed-proportional obstacle margin, offline, on the V7g / V8 records (no servo).
(verifier, CPU only, recorded data only, no new runs.)  Machinery = T43 SV sv_rows.py (field, spheres, classes, truth labels) + the
HM held-row classification (outputs/impl/T43HM/p/hm_survey.py features; linearize._held_rows_class rule of T43HM.impl.md).

usage: T43W_SNAP=<client snapshot> PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8 xm_speed.py ARM OUT_JSON [KEY_SUBSTR ...]
  ARM = V7g (CT/raw/analysis_v7g.json, snapshot work-t43v7) | V8 (V8/analysis, work-t43v8); stages H and B.

Phase of a chunk (server grasp.state / attached of the executing chunk; analysis server_chunks):
  return      = chunks from the first chunk whose grasp.state == 'placed' to the end of the run
  carry       = attached (state held) and the apple (planning-row object_poses) not over the crate footprint
  crate_entry = attached and over the crate footprint (SV / agg_v9 rule: |x| < 77 mm, |y| < 137 mm in the crate frame)
  closing     = not attached, grasp.state == 'closing'
  approach    = every other chunk before the first 'placed' chunk (states searching / latched, incl. after a revoke)
Plan rows: k = 0..7 of refined_chunk (the plan = the command on executed rows), fingers = T29 opening envelope (HX rule).
Spheres: robot spheres from the record (link-frame offsets, HX rule) + held query spheres (server held set) rigid with the EE body.
Per-sphere commanded speed: disp_i(k) = |c_i(k) - c_i(k-1)| (m per control row, 15 Hz) between consecutive commanded rows;
  c_i(-1) = sphere at the previous command executed_actions[t0 - 1] (arms; HOLD rows = q_hold) on the planning-row qpos (t0 = 0: q_now).
  (q_now is not used for k = 0: the measured state lags the command by about 2 rows, T43TK.)  speed * tau, tau = 2 rows -> 2 * disp_i(k).
Classes: robot rows (linearize._obstacle_rows): target if |d - d_object| <= fine voxel (d_object = union of held spheres at the
  capture pose if attached else nearest target point), support if |z - plane_z - d| <= 2 fine voxel, else obstacle.
  held rows (HM rule, T43HM.impl.md 'rule' table): support if |n.P - o - d| <= 2 voxel (held.support.plane of the record);
  held (own remnant) if |grad d| > 0 and the nearest field surface point P - d grad/|grad| lies within 2 voxel (10 mm) of the
  attach-time fit spheres (record held.fit_spheres, base frame); else obstacle.
Margins (mm) per variant, applied to obstacle-class rows only (robot obstacle rows and HM-obstacle held rows); every other row keeps its
  recorded-rule margin (esdf_margin 0, E1 per-link on non-exempt target rows; held support / remnant 0):
  cur   : robot obstacle 10, held rows 0 (the recorded V7g / V8 rule)
  HM    : robot obstacle 10, held obstacle 10 (k = 0)
  k0.5  : 10 + 0.5 * 2 * disp_i(k)          k1.0 : 10 + 1.0 * 2 * disp_i(k)     (robot obstacle and held obstacle rows)
  clearance_i = d_i - r_i - margin_i.
Truth label of a sphere (SV rule): nearest MuJoCo collision surface among hurdle / crate / table / apple / fruit_other, label = that set if
  |d_esdf - d_true| <= 10 mm else 'unexplained'.
Stored per (chunk, row k, variant): min clearance over cells group|label for obstacle-class rows, group = finger | palm | robot_other |
  held; plus the argmin (group, label, disp) and the row's max sphere disp (mm/row) over hand spheres and held spheres.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys, time
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p'))
import numpy as np, mujoco  # noqa: E402
import hold_w as HW  # noqa: E402
import dx_common as C  # noqa: E402
from AG3S.trajopt.experiments import fixed_input_ablation as FIA  # noqa: E402

T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
ANALYSIS = {"V7g": T43 / "CT/raw/analysis_v7g.json", "V8": T43 / "V8/analysis/analysis.json"}
OBST = 0.010
LABEL_TOL = 0.010
TAU_ROWS = 2.0
VARIANTS = {"cur": None, "HM": 0.0, "k0.5": 0.5, "k1.0": 1.0}
LINK2BODY = {"ee_left": "EE_BODY_L", "ee_right": "EE_BODY_R"}
FRUITS = ("banana", "orange", "pear")


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
    for key in keys:
        t_start = time.time()
        try:
            rr = one_run(cl, key, runs[key])
        except Exception as ex:
            import traceback
            rr = {"error": repr(ex), "trace": traceback.format_exc()[-2000:]}
        rr["sec"] = round(time.time() - t_start, 1)
        res["runs"][key] = rr
        print(key, "chunks", len(rr.get("chunks", [])), "sec", rr["sec"], rr.get("error", ""), flush=True)
        out_path.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    out_path.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def one_run(cl, key, a):
    rd = pathlib.Path(a["run_dir"]); man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile")
    ep = int(man["scene"]["episode_index"]); z = np.load(rd / "traj.npz", allow_pickle=True); seed = int(z["policy_seed"])
    stt = rd.parent / f"{rd.name}.start"
    cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
    L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; mg = L.get("obstacle_manager")
    hurdle = sorted(mg.collision_geom_ids) if mg is not None else []
    side = "right" if ep == 1834 else "left"; SD = side[0].upper()
    qL = np.asarray(L["idx"]["left_q"][:7], int); qR = np.asarray(L["idx"]["right_q"][:7], int)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    coll = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
    crate = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid("crate") and coll(g)]
    table_g = gid("table_top"); apple_g = gid("apple_geom")
    fruit_g = [g for g in range(m.ngeom) if coll(g) and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "") in FRUITS]
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
    EA = np.asarray(z["executed_actions"], float)
    sc = {int(lit(c["seq"])): c for c in (lit(a.get("server_chunks")) or [])}
    seqs = sorted(s_ for s_ in plan if s_ in chunks)
    placed_seq = next((s_ for s_ in seqs if lit((sc.get(s_) or {}).get("grasp_state")) == "placed"), None)

    def phase(s_):
        c = sc.get(s_) or {}
        if placed_seq is not None and s_ >= placed_seq:
            return "return"
        if lit(c.get("attached")) in (True, "True"):
            return "crate_entry" if over_crate(plan[s_]) else "carry"
        if lit(c.get("grasp_state")) == "closing":
            return "closing"
        return "approach"

    mujoco.mj_forward(m, d)
    out_chunks = []
    voxel_fine = None
    for s_ in seqs:
        ph = phase(s_)
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
        att = lit(g.get("attached")) if isinstance(lit(g.get("attached")), dict) else {}
        hrec = lit(att.get("held")) or {}
        sup = (hrec.get("support") or {}).get("plane") or {}
        plane = None; plane_z = None
        if sup.get("normal") is not None and sup.get("offset_m") is not None:
            plane = (np.asarray(sup["normal"], float), float(sup["offset_m"])); plane_z = float(sup["offset_m"])
        if plane_z is None:
            est = lit((s.get("esdf_stats") or {}).get("target_free")) or {}
            ball = est.get("ball") if isinstance(est, dict) else None
            if isinstance(ball, dict) and ball.get("support_plane_z_m") is not None:
                plane_z = float(ball["support_plane_z_m"])
        fit = [(np.asarray(x["centre_m"], float), 1e-3 * float(x["radius_mm"])) for x in (hrec.get("fit_spheres") or [])]
        Fc = np.array([c_ for c_, _ in fit]) if fit else None; Fr = np.array([r_ for _, r_ in fit]) if fit else None
        lm = FIA.contact_policy(st, links, has_target=True, has_free=False, attached=False)[0]
        radii_r = np.asarray(zz["sphere_radii"], float)[:S]
        lk = np.asarray(links).astype(str)
        grp_r = np.where(np.char.startswith(lk, "ee_finger"), "finger", np.where(np.isin(lk, ["ee_left", "ee_right"]), "palm", "robot_other"))
        if voxel_fine is None:
            voxel_fine = float(field.grid.voxel_size)
        vf = voxel_fine

        def centres(q):
            setq(q); Tb_ = C.T_world_base(m, d)
            R = d.xmat[body_ids].reshape(-1, 3, 3); p = d.xpos[body_ids]
            cwl = np.einsum("nij,nj->ni", R, local) + p; radii = radii_r.copy(); grp = grp_r.copy()
            if held:
                Te_ = C.body_T(m, d, f"EE_BODY_{SD}"); Hc_, Hr_ = C.place(Te_, held); cwl = np.r_[cwl, Hc_]; radii = np.r_[radii, Hr_]
                grp = np.r_[grp, np.array(["held"] * len(Hr_))]
            return cwl, radii, grp, Tb_

        def rows_at(q, cprev):
            cwl, radii, grp, Tb_ = centres(q)
            Pb = C.to_base(Tb_, cwl)
            dd = np.asarray(field.distance(Pb), float)
            if mask is not None:
                msk = np.r_[mask, np.zeros(len(Pb) - S, bool)]
                if msk.any():
                    dd[msk] = np.asarray(field.target_free_distance(Pb[msk]), float)
            cls = np.array(["obstacle"] * len(Pb), dtype=object); is_obj = np.zeros(len(Pb), bool)
            if d_obj_fn is not None:
                is_obj = np.abs(dd - d_obj_fn(Pb)) <= vf; is_obj[S:] = False; cls[is_obj] = "target"
            if plane_z is not None:
                sz = Pb[:S, 2] - plane_z; sel = (np.abs(sz - dd[:S]) <= 2 * vf) & (cls[:S] == "obstacle"); cls[:S][sel] = "support"
            if len(Pb) > S:   # HM held-row classification
                Ph = Pb[S:]; dh = dd[S:]
                hcls = np.array(["obstacle"] * len(Ph), dtype=object)
                gg = np.asarray(field.gradient(Ph), float); nrm = np.linalg.norm(gg, axis=1, keepdims=True)
                ghat = np.where(nrm > 1e-12, gg / np.maximum(nrm, 1e-12), 0.0); Sb = Ph - dh[:, None] * ghat
                if Fc is not None:
                    selfm = (nrm[:, 0] > 1e-12) & (C.union_sd(Sb, Fc, Fr) <= 2 * vf); hcls[selfm] = "held"
                if plane is not None:
                    spl = Ph @ plane[0] - plane[1]; hcls[np.abs(spl - dh) <= 2 * vf] = "support"
                cls[S:] = hcls
            base_margin = np.full(len(Pb), esdf_margin)
            exempt = np.zeros(len(Pb), bool)
            if mask is not None:
                exempt[:S] = mask
            selm = is_obj & ~exempt
            if d_obj_fn is not None and selm.any():
                base_margin[:S][selm[:S]] = np.asarray(lm, float)[selm[:S]]
            disp = np.linalg.norm(cwl - cprev, axis=1) if cprev is not None else np.zeros(len(cwl))
            tv = {"crate": np.min([C.geom_point_sd(m, d, g_, cwl) for g_ in crate], axis=0),
                  "table": C.geom_point_sd(m, d, table_g, cwl), "apple": ellipsoid_sd(m, d, apple_g, cwl)}
            if fruit_g:
                tv["fruit_other"] = np.min([ellipsoid_sd(m, d, g_, cwl) for g_ in fruit_g], axis=0)
            if hurdle:
                tv["hurdle"] = C.hurdle_sd(m, d, hurdle, cwl)
            names_ = list(tv); T_ = np.stack([tv[n_] for n_ in names_], 0)
            j = np.argmin(T_, axis=0); dtmin = T_[j, np.arange(T_.shape[1])]
            lab = np.array([names_[i] for i in j], dtype=object); lab[np.abs(dd - dtmin) > LABEL_TOL] = "unexplained"
            obst = cls == "obstacle"; is_held = np.zeros(len(Pb), bool); is_held[S:] = True
            out = {}
            for vname, kk in VARIANTS.items():
                mg_ = base_margin.copy()
                if kk is None:
                    mg_[obst & ~is_held] = np.maximum(mg_[obst & ~is_held], OBST)
                else:
                    extra = kk * TAU_ROWS * disp
                    mg_[obst] = np.maximum(mg_[obst], OBST + extra[obst])
                clr = dd - radii - mg_
                red = {}
                sel_all = obst if kk is not None else (obst & ~is_held)
                for gname in ("finger", "palm", "robot_other", "held"):
                    gs = (grp == gname) & sel_all
                    if not gs.any():
                        continue
                    for lname in set(lab[gs]):
                        sl = gs & (lab == lname)
                        jj = np.flatnonzero(sl)[np.argmin(clr[sl])]
                        red[f"{gname}|{lname}"] = [round(1e3 * float(clr[jj]), 2), round(1e3 * float(disp[jj]), 2), round(1e3 * float(mg_[jj]), 2)]
                out[vname] = red
            hand = (grp == "finger") | (grp == "palm")
            info = {"disp_hand_max_mm": round(1e3 * float(disp[hand].max()), 2) if hand.any() else None,
                    "disp_held_max_mm": round(1e3 * float(disp[is_held].max()), 2) if is_held.any() else None,
                    "n_held_cls": {c_: int(np.sum(cls[S:] == c_)) for c_ in ("obstacle", "support", "held")} if len(Pb) > S else None}
            return out, info, cwl

        c = sc.get(s_) or {}
        to = s.get("to") or {}; met = to.get("metrics") or {}
        t0 = int(plan[s_]["t_step"]); n_steps = min(8, ref.shape[0])
        # previous command: spheres at executed_actions[t0 - 1] (arms) on the planning-row qpos (t0 = 0: q_now)
        if t0 >= 1:
            cprev = centres(with_arms(qcap, np.r_[EA[t0 - 1, 0:7], EA[t0 - 1, 8:15]]))[0]
        else:
            cprev = centres(qcap)[0]
        rec = {"seq": s_, "t0": t0, "phase": ph, "grasp_state": lit(c.get("grasp_state")), "to_status": to.get("status"),
               "clearance_mm": None if met.get("clearance_m") is None else 1e3 * float(met["clearance_m"]),
               "hold": bool(any(not bool((ctrl.get(t0 + k_) or {}).get("executed", True)) for k_ in range(n_steps))),
               "attached": attached, "P": []}
        for k_ in range(n_steps):
            qP = set_fingers(with_arms(qcap, np.r_[ref[k_, 0:7], ref[k_, 8:15]]), fp[k_] if k_ < len(fp) else {})
            red, info, cwl = rows_at(qP, cprev)
            rec["P"].append([red, info]); cprev = cwl
        out_chunks.append(rec)
        zz.close()
    return {"side": side, "episode": ep, "seed": seed, "stage": a.get("stage"), "grasp": a.get("grasp"), "success": a.get("success"),
            "grasp_t": a.get("grasp_t"), "place_t": a.get("place_t"), "placed_seq": placed_seq, "voxel_fine_m": voxel_fine,
            "esdf_margin_mm": 1e3 * esdf_margin, "obstacle_margin_mm_log": om, "n_hurdle_geoms": len(hurdle), "chunks": out_chunks}


if __name__ == "__main__":
    main()
