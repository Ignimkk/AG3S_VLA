"""T43 XM part 1 — TO intervention at the hurdle crossing (V5 / V7g / V8 / V9 hurdle runs).
(verifier, CPU only, recorded data only, no new runs.)  Machinery = T43 HX hx_measure.py (scene reset, FK, TO-row reconstruction).

usage: T43W_SNAP=<client snapshot> PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8 xm_cross.py ARM OUT_JSON [KEY_SUBSTR ...]
  ARM = V5 (W/analysis, snap work-t43w3) | V7g (CT/raw/analysis_v7g.json, work-t43v7) | V8 (V8/analysis, work-t43v8) | V9 (V9/analysis, work-t43v8)

Frame (world): bar = hurdle geom whose name contains 'bar'; a = bar axis projected horizontal (unit); n = z x a, sign chosen so that
  n . (crate body xpos - bar centre) > 0  (= travel direction pick -> crate); z = world up.  s(p) = n . (p - bar centre) (signed
  horizontal distance across the bar; < 0 pick side, > 0 crate side).  bar top z = bar centre z + bar radius.
Points: palm = MJCF body EE_BODY_<S> origin (= TO link ee_left|ee_right); tip = mean of MJCF bodies ee_finger_<s>1, ee_finger_<s>2
  (both fingers -> the l1/l2 MJCF/URDF swap does not enter); apple = body 'apple' centre.
Control rows t (all rows): C(t) = executed_actions[t] on the M state (apple moved rigidly with EE_BODY, HX rule); M(t) = post state of
  row t = control row t+1 qpos (last row: row t qpos + measured arms).  Video frame t of third_person_front.mp4 = post state of row t
  (pi05_infer.py:2025-2035, appended after mj_step of row t; 15 fps -> time t/15 s).
  geo_* = mj_geomDistance (signed mm, distmax 0.5 m) of fingers (collision geoms of both finger bodies) / palm (EE body) / apple geom
  to the hurdle (bar + legs) and to the bar alone.  apple_bottom_z = lowest point of the rotated apple ellipsoid.
  apple_in_hand(M) = |apple - tip| <= 60 mm and apple z - apple z at t=0 >= 20 mm.  geoH apple is computed on every row (in hand or
  not; C moves the apple only when in hand).  grip_cmd = executed_actions gripper column (7 left / 15 right); finger_open_M = measured
  finger joint gap (gripper_finger_<s>2 - <s>1, rad-or-m as in the MJCF).
  drop_t = first row after the first in-hand row from which apple_in_hand stays false for >= 8 rows (None if never in hand / never left).
Crossings (per point, per state): carry crossing = first row t >= t_from with s(t-1) < 0 <= s(t) (t_from = first in-hand row, else the
  first 'closing' chunk row, else 0); return crossing = first row after the carry crossing with s(t-1) > 0 >= s(t).
  At a crossing row: apple_bottom - bar_top (vertical, mm), geo to bar (apple, fingers, palm), and min geo to the hurdle over rows
  crossing-8 .. crossing+8.
Per chunk (planning seq), k = 0..7: P_ref(k) / P_rfc(k) = FK of (planning-row qpos, arms = refined_chunk[k] / reference_chunk[k]); apple
  rigid with EE_BODY from the planning-row pose.  D(k) = P_ref(k) - P_rfc(k) (refined - reference), components:
  dz = D.z (up +), dn = D.n (along the travel direction, + forward toward the crate), da = D.a (along the bar),
  away = dn * sign(s(point at the planning-row state q_now))  (+ = horizontal push away from the bar, toward the side the robot is
  on when the chunk is planned); away_refside = dn * sign(s(P_rfc(k))) (side of the reference point),
  du = D.u, u = unit 3D velocity of the reference point (central difference over reference rows k-1..k+1)  (+ forward along the
  commanded motion, - backward).
  joint space: max_k max_j |refined - reference| (rad, 14 arm joints); anchor ratio = max_j |refined[0] - q_now| / max_step_j;
  step ratio_k = max_j |refined[k] - refined[k-1]| / max_step_j; accel ratio_k = max_j |second difference| / max_step_change_j
  (max_step = v_urdf * 0.9 / 15 Hz = 10.8 deg (j0-3), 21.6 deg (j4-5), 7.2 deg (j6); max_step_change = 10 * 0.9 / 15^2 rad = 2.29 deg).
  continuity: previous chunk (seq-1) refined rows 8..15 vs this chunk's refined / reference rows 0..7, palm position (mm).
  record: to.status, initial_source, initial_violation_m, clearance_m, path_check.clearance_m, max_violation_pair, path_check.worst,
  sqp_iterations, max_iterations_hit, candidates (step_size_rad, trust_radius_rad, violation_m), returned, continuity_active,
  plan_join_discontinuity_rad, gate action, executed rows, grasp state / attached / attach_revoked.
  binding = the smaller of waypoint clearance_m and path_check.clearance_m (analyze_t43w rule).
TO-row reconstruction (window chunks only, HX to_rows rule incl. the 10 mm obstacle margin, held rows margin 0, no destination
  margin, no path rows): min over all queries of d - r - margin for refined rows and reference rows k = 0..7 (+ argmin link / class /
  true hurdle distance of that sphere).
Window = rows with min over groups of min(geo C, geo M to the hurdle) <= 100 mm (HX rule; apple group only while in hand).
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
ANALYSIS = {"V7g": T43 / "CT/raw/analysis_v7g.json", "V8": T43 / "V8/analysis/analysis.json", "V5": T43 / "W/analysis/analysis.json",
            "V9": T43 / "V9/analysis/analysis.json", "V11A": T43 / "V11/analysis/analysis_V11A.json", "V11AB": T43 / "V11/analysis/analysis_V11AB.json",
            "V11ABC": T43 / "V11/analysis/analysis_V11ABC.json"}   # T43 V11: copy of T43/XM/p/xm_cross.py + the V11 arms (method unchanged)
WIN_MM = 100.0
OBST_MARGIN = 0.010
LINK2BODY = {"ee_left": "EE_BODY_L", "ee_right": "EE_BODY_R"}
MAX_STEP = np.array([np.pi, np.pi, np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2.094395102] * 2) * 0.9 / 15.0
MAX_STEP_CHANGE = np.full(14, 10.0 * 0.9 / 225.0)
URDF_OF_MJCF_FINGER = {"ee_finger_l1": "ee_finger_l2", "ee_finger_l2": "ee_finger_l1", "ee_finger_r1": "ee_finger_r1", "ee_finger_r2": "ee_finger_r2"}


def lit(x):
    return C.lit(x)


def f_or_none(x, s=1.0):
    try:
        return None if x is None else s * float(x)
    except (TypeError, ValueError):
        return None


def main():
    arm, out_path = sys.argv[1], pathlib.Path(sys.argv[2])
    runs = json.load(open(ANALYSIS[arm]))["runs"]
    keys = sorted(k for k in runs if k.startswith(f"{arm}/H/"))
    if len(sys.argv) > 3:
        keys = [k for k in keys if any(x in k for x in sys.argv[3:])]
    cl = HW.load_client()
    res = {"definitions": __doc__, "arm": arm, "client_snapshot": str(HW.HY), "max_step_rad": MAX_STEP.tolist(),
           "max_step_change_rad": MAX_STEP_CHANGE.tolist(), "runs": {}}
    for key in keys:
        t0w = time.time()
        try:
            rr = one_run(cl, key, runs[key])
        except Exception as ex:
            import traceback
            rr = {"error": repr(ex), "trace": traceback.format_exc()[-3000:]}
        rr["sec"] = round(time.time() - t0w, 1)
        res["runs"][key] = rr
        print(key, "sec", rr["sec"], rr.get("error", ""), flush=True)
        out_path.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    out_path.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def one_run(cl, key, a):
    rd = pathlib.Path(a["run_dir"]); man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile")
    ep = int(man["scene"]["episode_index"]); z = np.load(rd / "traj.npz", allow_pickle=True); seed = int(z["policy_seed"])
    stt = rd.parent / f"{rd.name}.start"
    cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
    L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; mg = L["obstacle_manager"]
    hurdle = sorted(int(g) for g in mg.collision_geom_ids)
    side = "right" if ep == 1834 else "left"; SD = side[0].upper(); fl = side[0]
    qL = np.asarray(L["idx"]["left_q"][:7], int); qR = np.asarray(L["idx"]["right_q"][:7], int)
    J = slice(0, 7) if side == "left" else slice(7, 14)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    ee_b = bid(f"EE_BODY_{SD}"); fb = [bid(f"ee_finger_{fl}1"), bid(f"ee_finger_{fl}2")]
    coll = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
    geoms_of = lambda b: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b and coll(g)]
    apple_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom")
    G = {"fingers": sum((geoms_of(b) for b in fb), []), "palm": geoms_of(ee_b), "apple": [apple_g]}
    apple_b = bid("apple"); apple_q = int(m.jnt_qposadr[m.body_jntadr[apple_b]])
    fj_adr = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in
              ("gripper_finger_l1", "gripper_finger_l2", "gripper_finger_r1", "gripper_finger_r2")}
    ft = np.zeros(6)
    mujoco.mj_forward(m, d)
    gbar = next(g for g in hurdle if "bar" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))
    bar_c = d.geom_xpos[gbar].copy(); ax = d.geom_xmat[gbar].reshape(3, 3)[:, 2].copy(); ax[2] = 0.0; ax /= np.linalg.norm(ax)
    nh = np.cross([0.0, 0.0, 1.0], ax); nh /= np.linalg.norm(nh)
    crate_p = d.xpos[bid("crate")].copy()
    if float(nh @ (crate_p - bar_c)) < 0:
        nh = -nh
    bar_r = float(m.geom_size[gbar][0]); bar_top = float(bar_c[2] + bar_r)
    apple_size = m.geom_size[apple_g][:3].copy()
    table_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    table_top_z = float(d.geom_xpos[table_g][2] + m.geom_size[table_g][2])

    def setq(q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)

    def with_arms(qfull, a14, move_apple):
        q = np.asarray(qfull, float).copy()
        if move_apple:
            setq(q); T0 = C.body_T(m, d, f"EE_BODY_{SD}")
        q[qL] = a14[0:7]; q[qR] = a14[7:14]
        if move_apple:
            setq(q); T1 = C.body_T(m, d, f"EE_BODY_{SD}")
            Ta = np.eye(4); Ta[:3, :3] = C.qmat(q[apple_q + 3:apple_q + 7]); Ta[:3, 3] = q[apple_q:apple_q + 3]
            Tn = T1 @ np.linalg.inv(T0) @ Ta; qa = np.zeros(4); mujoco.mju_mat2Quat(qa, Tn[:3, :3].reshape(-1))
            q[apple_q:apple_q + 3] = Tn[:3, 3]; q[apple_q + 3:apple_q + 7] = qa
        return q

    def set_fingers(q, vals):
        q = np.asarray(q, float).copy()
        for n_, v in vals.items():
            q[fj_adr[n_]] = v
        return q

    def pts(q):
        setq(q)
        return {"palm": d.xpos[ee_b].copy(), "tip": 0.5 * (d.xpos[fb[0]] + d.xpos[fb[1]]), "apple": d.xpos[apple_b].copy()}

    def apple_bottom(q):
        setq(q); R = d.xmat[apple_b].reshape(3, 3)
        gR = d.geom_xmat[apple_g].reshape(3, 3)
        return float(d.geom_xpos[apple_g][2] - np.sqrt(np.sum((gR[2, :] * apple_size) ** 2)))

    def geo(q, groups, targets):
        setq(q); out = {}
        for gname in groups:
            best = np.inf
            for g1 in G[gname]:
                for g2 in targets:
                    best = min(best, float(mujoco.mj_geomDistance(m, d, int(g1), int(g2), 0.5, ft)))
            out[gname] = 1e3 * best
        return out

    sfun = lambda p: float(nh @ (np.asarray(p) - bar_c))
    rows_f = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
    ctrl = {r["t_step"]: r for r in rows_f if r["kind"] == "control"}
    plan = {r["seq"]: r for r in rows_f if r["kind"] == "planning"}
    seg = lit(a.get("server_segment")); chunks = C.segment_chunks(seg) if seg and seg != "None" else {}
    st, outside, om, logp = C.server_settings(seg, rd / "manifest.json"); esdf_margin = float(st.esdf_margin)
    EA = np.asarray(z["executed_actions"], float); MQ = np.asarray(z["measured_qpos"], float); T = len(EA)
    sc = {int(lit(c["seq"])): c for c in (lit(a.get("server_chunks")) or [])}
    q0 = np.asarray(ctrl[0]["qpos"], float); apple_z0 = float(q0[apple_q + 2])
    first_close_t = min((int(lit(c["t"])) for c in sc.values() if lit(c.get("grasp_state")) == "closing"), default=None)
    first_att_t = min((int(lit(c["t"])) for c in sc.values() if lit(c.get("attached")) in (True, "True")), default=None)

    # ---------------- per control row ----------------
    rows = []
    for t in range(T):
        r = ctrl[t]
        if t + 1 in ctrl:
            qM = np.asarray(ctrl[t + 1]["qpos"], float)
        else:
            qM = np.asarray(r["qpos"], float).copy(); qM[qL] = MQ[t, 0:7]; qM[qR] = MQ[t, 8:15]
        pM = pts(qM); ab_M = apple_bottom(qM)
        inhand = bool(np.linalg.norm(pM["apple"] - pM["tip"]) <= 0.060 and pM["apple"][2] - apple_z0 >= 0.020)
        qC = with_arms(qM, np.r_[EA[t, 0:7], EA[t, 8:15]], inhand)
        pC = pts(qC); ab_C = apple_bottom(qC)
        groups = ("fingers", "palm", "apple")
        row = {"t": t, "seq": int(r["chunk_seq"]), "k": int(r["step_in_chunk"]), "executed": bool(r.get("executed", True)),
               "inhand": inhand, "apple_tip_mm": 1e3 * float(np.linalg.norm(pM["apple"] - pM["tip"])),
               "apple_dz_mm": 1e3 * float(pM["apple"][2] - apple_z0),
               "s_M": {k_: 1e3 * sfun(v) for k_, v in pM.items()}, "s_C": {k_: 1e3 * sfun(v) for k_, v in pC.items()},
               "z_M": {k_: 1e3 * float(v[2] - bar_top) for k_, v in pM.items()}, "z_C": {k_: 1e3 * float(v[2] - bar_top) for k_, v in pC.items()},
               "apple_bottom_minus_bar_top_M": 1e3 * (ab_M - bar_top), "apple_bottom_minus_bar_top_C": 1e3 * (ab_C - bar_top),
               "geoH_M": geo(qM, groups, hurdle), "geoH_C": geo(qC, groups, hurdle),
               "geoB_M": geo(qM, ("fingers", "palm", "apple"), [gbar]), "geoB_C": geo(qC, ("fingers", "palm", "apple"), [gbar]),
               "palm_M": pM["palm"].tolist(), "palm_C": pC["palm"].tolist(), "tip_M": pM["tip"].tolist(), "apple_M": pM["apple"].tolist()}
        row["grip_cmd"] = float(EA[t, 7] if side == "left" else EA[t, 15])
        row["finger_open_M"] = float(qM[fj_adr[f"gripper_finger_{fl}2"]] - qM[fj_adr[f"gripper_finger_{fl}1"]])
        gw = ("fingers", "palm", "apple") if inhand else ("fingers", "palm")
        dm = min(min(row["geoH_M"][g_] for g_ in gw), min(row["geoH_C"][g_] for g_ in gw))
        row["in_window"] = bool(dm <= WIN_MM)
        rows.append(row)
    # in-hand / drop
    ih = [r_["inhand"] for r_ in rows]
    first_in = next((t for t in range(T) if ih[t]), None)
    drop_t = None
    if first_in is not None:
        for t in range(first_in + 1, T):
            if not ih[t] and all(not ih[u] for u in range(t, min(T, t + 8))):
                drop_t = t; break
    # crossings
    t_from = first_in if first_in is not None else (first_close_t if first_close_t is not None else 0)
    cross = {}
    for st_ in ("M", "C"):
        for p_ in ("palm", "tip", "apple"):
            s_ = [r_[f"s_{st_}"][p_] for r_ in rows]
            tc = next((t for t in range(max(1, t_from), T) if s_[t - 1] < 0 <= s_[t]), None)
            tr = next((t for t in range((tc or T) + 1, T) if s_[t - 1] > 0 >= s_[t]), None) if tc is not None else None
            cross[f"{p_}_{st_}"] = {"carry": tc, "return": tr}

    def at_cross(t, st_):
        if t is None:
            return None
        r_ = rows[t]; lo, hi = max(0, t - 8), min(T, t + 9)
        win = rows[lo:hi]
        out = {"t": t, "inhand": r_["inhand"], "seq": r_["seq"], "k": r_["k"], "executed": r_["executed"],
               "apple_bottom_minus_bar_top_mm": r_[f"apple_bottom_minus_bar_top_{st_}"],
               "geo_bar_mm": r_[f"geoB_{st_}"], "z_minus_bar_top_mm": r_[f"z_{st_}"],
               "min_geoH_pm8_mm": {g_: min(w[f"geoH_{st_}"].get(g_, np.inf) for w in win) for g_ in ("fingers", "palm", "apple")},
               "min_geoH_pm8_t": {g_: win[int(np.argmin([w[f"geoH_{st_}"].get(g_, np.inf) for w in win]))]["t"] for g_ in ("fingers", "palm", "apple")}}
        return out
    crossing = {}
    for st_ in ("M", "C"):
        for which in ("carry", "return"):
            for p_ in ("palm", "tip", "apple"):
                crossing[f"{which}_{p_}_{st_}"] = at_cross(cross[f"{p_}_{st_}"][which], st_)

    # ---------------- per chunk ----------------
    zc = {}
    voxel_fine = [None]

    def chunk_full(seq):
        """field + sphere machinery for TO rows (HX rule)."""
        if seq in zc:
            return zc[seq]
        zz = np.load(chunks[seq], allow_pickle=True); s = json.loads(str(zz["summary_json"]))
        field = FIA.build_field(zz, outside)
        S = int(zz["sphere_radii"].shape[0]); links = list(lit(s.get("sphere_link_names")) or [])
        ref = np.asarray(zz["refined_chunk"], float); rfc = np.asarray(zz["reference_chunk"], float)
        qcap = np.asarray(plan[seq]["qpos"], float)
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
        fp_ref, fp_rfc = fpath(ref), fpath(rfc)
        qq = set_fingers(with_arms(qcap, np.r_[ref[0, 0:7], ref[0, 8:15]], False), fp_ref[0] if fp_ref else {}); setq(qq); Tb = C.T_world_base(m, d)
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
        lm = FIA.contact_policy(st, links, has_target=True, has_free=False, attached=False)[0]
        zc[seq] = dict(z=zz, s=s, field=field, S=S, links=np.asarray(links), local=local, body_ids=body_ids, held=held, mask=mask,
                       d_obj=d_obj_fn, plane_z=plane_z, link_margin=lm, ref=ref, rfc=rfc, qcap=qcap, fp_ref=fp_ref, fp_rfc=fp_rfc)
        return zc[seq]

    def to_min(cc, a14, fingers):
        q = set_fingers(with_arms(cc["qcap"], a14, False), fingers or {}); setq(q); Tb = C.T_world_base(m, d)
        S = cc["S"]; R = d.xmat[cc["body_ids"]].reshape(-1, 3, 3); p = d.xpos[cc["body_ids"]]
        cw = np.einsum("nij,nj->ni", R, cc["local"]) + p
        radii = np.asarray(cc["z"]["sphere_radii"], float)[:S]
        if cc["held"]:
            Te = C.body_T(m, d, f"EE_BODY_{SD}"); Hc, Hr = C.place(Te, cc["held"]); cw = np.r_[cw, Hc]; radii = np.r_[radii, Hr]
        Pb = C.to_base(Tb, cw); fld = cc["field"]
        dd = np.asarray(fld.distance(Pb), float)
        if cc["mask"] is not None:
            msk = np.r_[cc["mask"], np.zeros(len(Pb) - S, bool)]
            if msk.any():
                dd[msk] = np.asarray(fld.target_free_distance(Pb[msk]), float)
        if voxel_fine[0] is None:
            voxel_fine[0] = float(fld.grid.voxel_size)
        vf = voxel_fine[0]
        cls = np.array(["obstacle"] * len(Pb), dtype=object); is_obj = np.zeros(len(Pb), bool)
        if cc["d_obj"] is not None:
            is_obj = np.abs(dd - cc["d_obj"](Pb)) <= vf; is_obj[S:] = False; cls[is_obj] = "target"
        if cc["plane_z"] is not None:
            sz = Pb[:, 2] - cc["plane_z"]; cls[(np.abs(sz - dd) <= 2 * vf) & (cls == "obstacle")] = "support"
        cls[S:] = "held"
        margin = np.full(len(Pb), esdf_margin)
        exempt = np.zeros(len(Pb), bool)
        if cc["mask"] is not None:
            exempt[:S] = cc["mask"]
        sel = is_obj & ~exempt
        if cc["d_obj"] is not None and sel.any():
            margin[:S][sel[:S]] = np.asarray(cc["link_margin"], float)[sel[:S]]
        margin[cls == "obstacle"] = np.maximum(margin[cls == "obstacle"], OBST_MARGIN)
        clear = dd - radii - margin
        dtrue = C.hurdle_sd(m, d, hurdle, cw) - radii
        j = int(np.argmin(clear))
        lk = str(cc["links"][j]) if j < S else "held"
        hand = np.r_[np.char.startswith(cc["links"].astype(str), "ee_"), np.ones(len(Pb) - S, bool)]
        jh = int(np.flatnonzero(hand)[np.argmin(clear[hand])])
        return {"min_mm": 1e3 * float(clear[j]), "link": lk, "query": j, "cls": str(cls[j]), "true_hurdle_mm": 1e3 * float(dtrue[j]),
                "hand_min_mm": 1e3 * float(clear[jh]), "hand_query": jh, "hand_cls": str(cls[jh]),
                "hand_true_hurdle_mm": 1e3 * float(dtrue[jh])}

    win_seqs = sorted({r_["seq"] for r_ in rows if r_["in_window"]})
    out_chunks = []
    prev_ref = {}
    for seq in sorted(plan):
        if seq not in chunks:
            continue
        with np.load(chunks[seq], allow_pickle=True) as zz:
            s = json.loads(str(zz["summary_json"])); ref = np.asarray(zz["refined_chunk"], float); rfc = np.asarray(zz["reference_chunk"], float)
        prev_ref[seq] = ref
        qcap = np.asarray(plan[seq]["qpos"], float); t0 = int(plan[seq]["t_step"])
        c = sc.get(seq) or {}
        att = lit(c.get("attached")) in (True, "True")
        pc = pts(qcap)
        inhand_cap = bool(np.linalg.norm(pc["apple"] - pc["tip"]) <= 0.060 and pc["apple"][2] - apple_z0 >= 0.020)
        n_steps = min(8, ref.shape[0])
        P = {"ref": [], "rfc": []}
        for k in range(n_steps):
            for lab, arr in (("ref", ref), ("rfc", rfc)):
                P[lab].append(pts(with_arms(qcap, np.r_[arr[k, 0:7], arr[k, 8:15]], inhand_cap)))
        dfl = {}
        for p_ in ("palm", "tip", "apple"):
            comp = []
            for k in range(n_steps):
                D = P["ref"][k][p_] - P["rfc"][k][p_]
                prv = P["rfc"][max(0, k - 1)][p_]; nxt = P["rfc"][min(n_steps - 1, k + 1)][p_]
                v = nxt - prv; vn = float(np.linalg.norm(v)); u = v / vn if vn > 1e-9 else np.zeros(3)
                srf = sfun(P["rfc"][k][p_]); snow = sfun(pc[p_])
                comp.append({"mag": 1e3 * float(np.linalg.norm(D)), "dz": 1e3 * float(D[2]), "dn": 1e3 * float(D @ nh), "da": 1e3 * float(D @ ax),
                             "away": 1e3 * float(D @ nh) * (1.0 if snow >= 0 else -1.0), "away_refside": 1e3 * float(D @ nh) * (1.0 if srf >= 0 else -1.0),
                             "s_now_mm": 1e3 * snow, "du": 1e3 * float(D @ u),
                             "ref_speed_mm_per_row": 1e3 * vn / max(1, min(n_steps - 1, k + 1) - max(0, k - 1)),
                             "s_rfc_mm": 1e3 * srf, "z_rfc_minus_bar_top_mm": 1e3 * float(P["rfc"][k][p_][2] - bar_top)})
            dfl[p_] = comp
        Q = np.stack([np.r_[ref[k, 0:7], ref[k, 8:15]] for k in range(n_steps)]); Qr = np.stack([np.r_[rfc[k, 0:7], rfc[k, 8:15]] for k in range(n_steps)])
        qn = np.r_[qcap[qL], qcap[qR]]
        dQ = np.diff(np.r_[qn[None], Q], axis=0)
        step_ratio = (np.abs(dQ) / MAX_STEP[None]).max(1)
        d2 = np.diff(np.r_[qn[None], Q], n=2, axis=0) if n_steps >= 2 else np.zeros((0, 14))
        acc_ratio = (np.abs(d2) / MAX_STEP_CHANGE[None]).max(1) if len(d2) else np.zeros(0)
        cont = None
        if (seq - 1) in prev_ref and prev_ref[seq - 1].shape[0] >= 16:
            pr = prev_ref[seq - 1]; cont = {"ref": [], "rfc": []}
            for k in range(n_steps):
                pp = pts(with_arms(qcap, np.r_[pr[k + 8, 0:7], pr[k + 8, 8:15]], False))["palm"]
                cont["ref"].append(1e3 * float(np.linalg.norm(P["ref"][k]["palm"] - pp)))
                cont["rfc"].append(1e3 * float(np.linalg.norm(P["rfc"][k]["palm"] - pp)))
        to = s.get("to") or {}; met = to.get("metrics") or {}
        mv = lit(met.get("max_violation_pair")) or {}; pch = lit(met.get("path_check")) or {}
        pw = (pch.get("worst") or {}) if isinstance(pch, dict) else {}
        isrc = lit(met.get("initial_source")) or {}
        cands = lit(met.get("candidates")) or []
        g = s.get("grasp") or {}
        wpc = f_or_none(met.get("clearance_m"), 1e3); ptc = f_or_none(pch.get("clearance_m"), 1e3) if isinstance(pch, dict) else None
        if ptc is not None and (wpc is None or ptc < wpc):
            bind = {"rows": "path", "clearance_mm": ptc, "link": pw.get("link"), "query": pw.get("query"), "step": pw.get("step")}
        else:
            bind = {"rows": "waypoint", "clearance_mm": wpc, "link": mv.get("link"), "query": mv.get("query"), "step": mv.get("step"),
                    "class": mv.get("obstacle_margin_class"), "obstacle": mv.get("obstacle")}
        S_ = None
        try:
            with np.load(chunks[seq], allow_pickle=True) as zz:
                S_ = int(zz["sphere_radii"].shape[0])
        except Exception:
            pass
        q_ = bind.get("query")
        bind["category"] = None if q_ is None or S_ is None else ("base" if int(q_) < 122 else ("gripper_cover" if int(q_) < S_ else "held"))
        bind["link_mjcf"] = {v: k for k, v in URDF_OF_MJCF_FINGER.items()}.get(bind.get("link"), bind.get("link"))
        executed = [bool((ctrl.get(t0 + k) or {}).get("executed", True)) for k in range(n_steps)]
        rec = {"seq": seq, "t0": t0, "grasp_state": g.get("state"), "attached": att, "attach_revoked": lit(c.get("attach_revoked")),
               "grasp_detach": lit(c.get("grasp_detach")), "held_slip": lit(c.get("held_slip")), "inhand_at_plan": inhand_cap,
               "in_window": seq in win_seqs, "executed": executed,
               "to_status": to.get("status"), "sqp_iterations": f_or_none(to.get("sqp_iterations")), "max_iterations_hit": to.get("max_iterations_hit"),
               "time_budget_hit": to.get("time_budget_hit"), "initial_source": isrc.get("source") if isinstance(isrc, dict) else isrc,
               "stay_violation_mm": f_or_none(isrc.get("stay_violation_m"), 1e3) if isinstance(isrc, dict) else None,
               "initial_violation_mm": f_or_none(met.get("initial_violation_m"), 1e3), "wp_clearance_mm": wpc, "path_clearance_mm": ptc,
               "returned": met.get("returned"), "returned_index": met.get("returned_index"), "n_accepted": met.get("n_accepted"),
               "candidates": [{"it": c_.get("iteration"), "step_rad": c_.get("step_size_rad"), "trust": c_.get("trust_radius_rad"),
                               "viol_mm": f_or_none(c_.get("violation_m"), 1e3), "acc": c_.get("accepted")} for c_ in cands if isinstance(c_, dict)],
               "continuity_active": met.get("continuity_active"), "plan_join_discontinuity_rad": f_or_none(met.get("plan_join_discontinuity_rad")),
               "binding": bind, "wp_worst": {"link": mv.get("link"), "query": mv.get("query"), "step": mv.get("step"), "class": mv.get("obstacle_margin_class"),
                                              "obstacle": mv.get("obstacle"), "clearance_mm": f_or_none(mv.get("clearance_m"), 1e3)},
               "path_worst": {k_: pw.get(k_) for k_ in ("step", "at", "query", "link", "floor_m")} if pw else None,
               "gate_action": (lit(plan[seq].get("gate")) or {}).get("action") if isinstance(lit(plan[seq].get("gate")), dict) else plan[seq].get("gate"),
               "verdict": [(x.get("kind"), x.get("action")) for x in ((s.get("verdict") or {}).get("reasons") or [])],
               "deflection": dfl, "joint_max_abs_rad": float(np.abs(Q - Qr).max()), "joint_max_abs_rad_active_arm": float(np.abs(Q - Qr)[:, J].max()),
               "anchor_ratio": float(step_ratio[0]), "step_ratio": step_ratio.tolist(), "accel_ratio": acc_ratio.tolist(),
               "ref_anchor_ratio": float((np.abs(Qr[0] - qn) / MAX_STEP).max()), "continuity_palm_mm": cont}
        if seq in win_seqs:
            cc = chunk_full(seq)
            rec["to_ref"] = [to_min(cc, np.r_[cc["ref"][k, 0:7], cc["ref"][k, 8:15]], cc["fp_ref"][k] if k < len(cc["fp_ref"]) else {}) for k in range(n_steps)]
            rec["to_rfc"] = [to_min(cc, np.r_[cc["rfc"][k, 0:7], cc["rfc"][k, 8:15]], cc["fp_rfc"][k] if k < len(cc["fp_rfc"]) else {}) for k in range(n_steps)]
            zc.pop(seq, None)
        out_chunks.append(rec)
    return {"key": key, "side": side, "episode": ep, "seed": seed, "grasp": a.get("grasp"), "success": a.get("success"), "place": a.get("place"),
            "grasp_t": a.get("grasp_t"), "place_t": a.get("place_t"), "n_rows": T, "first_closing_t": first_close_t, "first_attached_t": first_att_t,
            "first_inhand_t": first_in, "drop_t": drop_t, "obstacle_margin_mm_log": om, "esdf_margin_mm": 1e3 * esdf_margin,
            "bar": {"centre": bar_c.tolist(), "axis_h": ax.tolist(), "n_travel": nh.tolist(), "radius_mm": 1e3 * bar_r, "top_z": bar_top,
                    "half_length_mm": 1e3 * float(m.geom_size[gbar][1])},
            "crate_xpos": crate_p.tolist(), "table_top_z": table_top_z, "apple_z0": apple_z0, "voxel_fine_m": voxel_fine[0],
            "crossing_rows": cross, "crossing": crossing, "rows": rows, "chunks": out_chunks,
            "mp4": str(rd / "third_person_front.mp4"), "n_mp4_frames_expected": T}


if __name__ == "__main__":
    main()
