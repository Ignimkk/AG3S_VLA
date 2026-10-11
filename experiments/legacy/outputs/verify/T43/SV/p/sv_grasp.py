"""T43 SV part 1/2 — why grasp collapses with the servo model: per-pair (same episode:seed) chunk table, servo arm A vs no-servo arm B.
(verifier, CPU only, recorded data only, no new runs.)

usage: T43W_SNAP=<client snapshot> PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8 sv_grasp.py PAIR OUT_JSON
  PAIR = V9V8 (A = V9 servo, B = V8; analyses V9/analysis, V8/analysis; snapshot work-t43v8)
       | V3V4 (A = V3 servo+sweep, B = V4; W/analysis; snapshot work-t43w)

Definitions
  keys            : every (stage, episode, seed) present in both arms. grasp / success = analysis rule (analyze_t39 / analyze_e0).
  measured pose   : control row qpos (MuJoCo FK, model_transport_pick_place_obstacles.xml of the client snapshot): TCP = site <side>_ee,
                    fingertip midpoint = mean of bodies ee_finger_<s>1/2 (MJCF names), apple = body 'apple'.
  divergence      : first control row t with |TCP_A(t) - TCP_B(t)| > 10 mm (div10) and > 30 mm (div30); phase at that row = server grasp.state
                    of the executing chunk in each arm; first planning seq whose reference_chunk (policy output) differs between arms
                    (max |diff| > 1e-4 rad) and the refined-vs-refined difference at seq 1 (identical inputs).
  chunk table     : seq, t, grasp.state, attach / detach, to.status, initial_source (sqp._stay_start record), solve_ms, sqp iterations,
                    measured fingertip-apple distance at the planning row (mm), outward_mm (analyze_t43w focus rule, every chunk before attach:
                    fingertip midpoint at control row 7, refined vs reference arm columns, planning-row finger joints, distance to the apple centre
                    of the planning row; > 0 = refined further from the apple), shift_mm, tcp_deflection_mm (analysis), tip_alignment.now
                    (server record: d3 / dxy / dz of fingertip vs AG3S centroid).
  binding row     : the smaller of waypoint clearance (to.metrics.max_violation_pair) and path clearance (to.metrics.path_check.worst)
                    (analyze_t43w rule). Waypoint class = recorded obstacle_margin_class. Path class is not recorded -> reconstructed at the
                    recorded worst point (path_check.worst.point_m, base frame) with the chunk's own field (FIA.build_field):
                      authorized = link in exclusion.authorized_links; mask = exclusion.mechanism has target_free_layer and not attached and
                      has_target and the record has esdf_free_*  (linearize._target_free_rows, linearize.py:1044-1052);
                      d = target_free_distance if (mask and authorized) else composite distance (linearize.py:714-728);
                      d_object = nearest recorded target point (not attached) (linearize.py:788-802);
                      class (linearize._obstacle_rows, linearize.py:804-862): target if |d - d_object| <= fine voxel; support if
                      |d - (z - plane_z)| <= 2 fine voxel (plane_z = esdf_stats.target_free.ball.support_plane_z_m or held support plane);
                      else obstacle.  E1 margin (manipulated_link_margin, linearize.py:764-778) applies to target rows of non-exempt spheres;
                      exempt (T30 F1, linearize.py:731-737, 1054-1072) = mask rows; path rows drop robot rows whose composite d is the target
                      and which are not mask rows (sweep._target_rows, sweep.py:427-460).
                      margin = esdf_margin (0) | E1 per-link margin | max(margin, obstacle_margin 10 mm) for obstacle rows.
                      reconstructed clearance = d - r - margin; fidelity = |recon - recorded path clearance_m| (floor 0 rows).
                    The same reconstruction is run on the waypoint worst row (recorded step / query / point_m) to check the class rule against
                    the recorded obstacle_margin_class.
                    Geometric truth at the binding point: nearest MuJoCo collision surface among apple (target), other fruit, table, crate,
                    hurdle (obstacle manager geoms; ellipsoids approximated radially), signed distance minus the sphere radius.
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
PAIRS = {"V9V8": ("V9", "V8", T43 / "V9/analysis/analysis.json", T43 / "V8/analysis/analysis.json"),
         "V3V4": ("V3", "V4", T43 / "W/analysis/analysis.json", T43 / "W/analysis/analysis.json")}
OBST = 0.010
LINK2BODY = {"ee_left": "EE_BODY_L", "ee_right": "EE_BODY_R"}
LEFT_Q = list(range(20, 27)); RIGHT_Q = list(range(11, 18))
FRUITS = ("banana", "orange", "pear")


def lit(x):
    return C.lit(x)


def ellipsoid_sd(m, d, g, P):
    c = d.geom_xpos[g]; R = d.geom_xmat[g].reshape(3, 3); s = m.geom_size[g][:3]
    q = (np.atleast_2d(P) - c) @ R; f = np.linalg.norm(q / s, axis=1); r = np.linalg.norm(q, axis=1)
    return r * (1.0 - 1.0 / np.maximum(f, 1e-9))


def main():
    pair, out_path = sys.argv[1], pathlib.Path(sys.argv[2])
    A, B, pa, pb = PAIRS[pair]
    RA = json.load(open(pa))["runs"]; RB = RA if pb == pa else json.load(open(pb))["runs"]
    cl = HW.load_client()
    res = {"definitions": __doc__, "pair": pair, "A": A, "B": B, "client_snapshot": str(HW.HY), "runs": {}}
    keys = sorted(k for k in RA if k.startswith(A + "/"))
    if len(sys.argv) > 3:
        keys = [k for k in keys if any(x in k for x in sys.argv[3:])]
    for ka in keys:
        kb = ka.replace(f"{A}/", f"{B}/", 1).replace(f"/{A}_", f"/{B}_")
        if kb not in RB:
            continue
        t0w = time.time()
        rec = {"key_A": ka, "key_B": kb, "stage": RA[ka]["stage"], "episode": RA[ka]["episode"], "seed": int(RA[ka]["key"]),
               "grasp_A": RA[ka]["grasp"], "grasp_B": RB[kb]["grasp"], "success_A": RA[ka]["success"], "success_B": RB[kb]["success"],
               "grasp_t_A": RA[ka]["grasp_t"], "grasp_t_B": RB[kb]["grasp_t"], "arms": {}}
        series = {}
        for arm, key, R in ((A, ka, RA), (B, kb, RB)):
            series[arm] = one_arm(cl, key, R[key])
            rec["arms"][arm] = series[arm]["rec"]
        # divergence (control rows)
        ta, tb = series[A]["tcp"], series[B]["tcp"]; n = min(len(ta), len(tb))
        dt = 1e3 * np.linalg.norm(ta[:n] - tb[:n], axis=1)
        def first(th):
            w = np.flatnonzero(dt > th); return int(w[0]) if len(w) else None
        div = {"tcp_diff_mm_by_row_head": [round(float(x), 2) for x in dt[:200]]}
        for th in (10, 30):
            t = first(th); div[f"div{th}_t"] = t
            if t is not None:
                div[f"div{th}_state_A"] = series[A]["state_at"](t); div[f"div{th}_state_B"] = series[B]["state_at"](t)
        # policy-output divergence
        pa_, pb_ = series[A]["plan"], series[B]["plan"]
        fr = None
        for seq in sorted(set(pa_) & set(pb_)):
            ra = np.asarray(pa_[seq].get("actions_reference") or pa_[seq]["actions"], float)
            rb = np.asarray(pb_[seq].get("actions_reference") or pb_[seq]["actions"], float)
            if ra.shape == rb.shape and np.abs(ra - rb).max() > 1e-4:
                fr = {"seq": seq, "t": pa_[seq]["t_step"], "max_abs_rad": float(np.abs(ra - rb).max())}; break
        div["first_reference_diff"] = fr
        if 1 in pa_ and 1 in pb_:
            ra = np.asarray(pa_[1]["actions"], float)[:8]; rb = np.asarray(pb_[1]["actions"], float)[:8]
            r0a = np.asarray(pa_[1].get("actions_reference") or pa_[1]["actions"], float)[:8]
            r0b = np.asarray(pb_[1].get("actions_reference") or pb_[1]["actions"], float)[:8]
            div["seq1_reference_equal_max_abs_rad"] = float(np.abs(r0a - r0b).max())
            div["seq1_refined_A_minus_B_max_abs_rad"] = float(np.abs(ra - rb).max())
        rec["divergence"] = div
        rec["sec"] = round(time.time() - t0w, 1)
        res["runs"][ka] = rec
        print(ka, "gA", rec["grasp_A"], "gB", rec["grasp_B"], "div10", div.get("div10_t"), "div30", div.get("div30_t"), "sec", rec["sec"], flush=True)
        out_path.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    out_path.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def one_arm(cl, key, a):
    rd = pathlib.Path(a["run_dir"]); man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile")
    ep = int(man["scene"]["episode_index"]); z = np.load(rd / "traj.npz", allow_pickle=True); seed = int(z["policy_seed"])
    stt = rd.parent / f"{rd.name}.start"
    cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
    L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; mg = L.get("obstacle_manager")
    hurdle = sorted(mg.collision_geom_ids) if mg is not None else []
    side = "right" if ep == 1834 else "left"; fl = side[0]; SD = side[0].upper()
    qadr = LEFT_Q if side == "left" else RIGHT_Q; c0 = 0 if side == "left" else 8
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    b1, b2, ba = bid(f"ee_finger_{fl}1"), bid(f"ee_finger_{fl}2"), bid("apple")
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee")
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    coll = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
    crate = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid("crate") and coll(g)]
    table = [gid("table_top")]
    apple_g = gid("apple_geom")
    fruit_g = [g for g in range(m.ngeom) if coll(g) and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "") in FRUITS]

    rows = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
    ctrl = sorted([r for r in rows if r["kind"] == "control"], key=lambda r: r["t_step"])
    plan = {r["seq"]: r for r in rows if r["kind"] == "planning"}

    def setq(q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    tcp = []
    for r in ctrl:
        setq(r["qpos"]); tcp.append(d.site_xpos[site].copy())
    tcp = np.asarray(tcp)
    sc = {c["seq"]: c for c in (a.get("server_chunks") or [])}
    seq_of_t = {r["t_step"]: r["chunk_seq"] for r in ctrl}

    def state_at(t):
        c = sc.get(seq_of_t.get(t)); return None if c is None else c.get("grasp_state")

    seg = lit(a.get("server_segment")); chunks = C.segment_chunks(seg) if seg and seg != "None" else {}
    st, outside, om, logp = C.server_settings(seg, rd / "manifest.json") if chunks else (None, None, None, None)
    esdf_margin = float(st.esdf_margin) if st is not None else 0.0

    def truth(pw, r):
        """nearest MuJoCo surface to world point pw for a sphere of radius r."""
        out = {}
        out["apple"] = float(ellipsoid_sd(m, d, apple_g, pw)[0]) - r
        if fruit_g:
            out["fruit_other"] = float(min(ellipsoid_sd(m, d, g, pw)[0] for g in fruit_g)) - r
        out["table"] = float(C.geom_point_sd(m, d, table[0], pw)[0]) - r
        out["crate"] = float(min(C.geom_point_sd(m, d, g, pw)[0] for g in crate)) - r
        if hurdle:
            out["hurdle"] = float(C.hurdle_sd(m, d, hurdle, pw)[0]) - r
        lab = min(out, key=out.get)
        return lab, {k: round(1e3 * v, 2) for k, v in out.items()}

    att_t = next((c["t"] for c in (a.get("server_chunks") or []) if c.get("attached")), None)
    first_close = next((c["t"] for c in (a.get("server_chunks") or []) if c.get("grasp_state") == "closing"), None)
    t_end = max([x for x in (att_t, first_close) if x is not None], default=160) + 24
    t_end = min(max(t_end, 120), 216)
    table_rows = []
    for seq in sorted(plan):
        p = plan[seq]; t = int(p["t_step"])
        if t > t_end:
            break
        c = sc.get(seq) or {}
        e = {"seq": seq, "t": t, "grasp_state": c.get("grasp_state"), "attached": c.get("attached"), "attach_now": c.get("attach_now"),
             "detach": c.get("grasp_detach"), "to_status": c.get("to_status"), "initial_source": c.get("initial_source"),
             "solve_ms": c.get("solve_ms"), "sqp_iterations": c.get("sqp_iterations"), "time_budget_hit": c.get("time_budget_hit"),
             "tcp_deflection_mm": c.get("tcp_deflection_mm"), "binding": c.get("binding"),
             "wp": c.get("wp"), "path": c.get("path"), "follow_to": (c.get("follow_to") or {}).get("decision") if isinstance(c.get("follow_to"), dict) else None}
        q = np.asarray(p["qpos"], float); setq(q)
        tip = 0.5 * (d.xpos[b1] + d.xpos[b2]); ap = d.xpos[ba].copy()
        e["tip_apple_mm"] = 1e3 * float(np.linalg.norm(tip - ap))
        # outward (focus rule) on every chunk
        acts = np.asarray(p["actions"], float); ref = np.asarray(p.get("actions_reference") or p["actions"], float)
        k = min(7, len(acts) - 1); Tt = []
        for row in (acts[k], ref[k]):
            qq = q.copy(); qq[qadr] = row[c0:c0 + 7]; setq(qq); Tt.append(0.5 * (d.xpos[b1] + d.xpos[b2]))
        e["outward_mm"] = 1e3 * float(np.linalg.norm(Tt[0] - ap) - np.linalg.norm(Tt[1] - ap))
        e["shift_mm"] = 1e3 * float(np.linalg.norm(Tt[0] - Tt[1]))
        setq(q)
        if seq in chunks:
            zz = np.load(chunks[seq], allow_pickle=True); s = json.loads(str(zz["summary_json"]))
            g = s.get("grasp") or {}; ta = g.get("tip_alignment") or {}
            if isinstance(ta, dict) and ta.get("now"):
                e["tip_alignment_now"] = {k_: ta["now"].get(k_) for k_ in ("d3_mm", "dxy_mm", "dz_mm")}
            e["recon"] = recon(zz, s, st, outside, esdf_margin, m, d, q, truth)
        table_rows.append(e)
    rec = {"side": side, "attach_t": att_t, "first_closing_t": first_close, "grasp_t": a.get("grasp_t"), "chunks": table_rows,
           "esdf_margin_mm": 1e3 * esdf_margin, "obstacle_margin_mm_log": om, "server_log": logp}
    return {"rec": rec, "tcp": tcp, "plan": plan, "state_at": state_at}


def recon(zz, s, st, outside, esdf_margin, m, d, qplan, truth):
    """class / exemptions / clearance of the recorded waypoint and path worst rows (see module doc)."""
    field = FIA.build_field(zz, outside)
    vox = float(field.grid.voxel_size)
    links = list(lit(s.get("sphere_link_names")) or [])
    radii = np.asarray(zz["sphere_radii"], float); S = len(radii)
    g = s.get("grasp") or {}; att = lit(g.get("attached")); attached = isinstance(att, dict) and bool(att)
    exc = s.get("exclusion") or {}; auth = set(exc.get("authorized_links") or [])
    mask_on = bool((not attached) and st.target_field_policy != "relax" and s.get("has_target") and "esdf_free_distance" in zz.files
                   and "target_free_layer" in (exc.get("mechanism") or []))
    tp = np.asarray(zz["target_points"], float) if ("target_points" in zz.files and s.get("has_target") and not attached) else None
    tree = None
    if tp is not None and len(tp):
        from scipy.spatial import cKDTree
        tree = cKDTree(tp)
    est = lit((s.get("esdf_stats") or {}).get("target_free")) or {}
    ball = est.get("ball") if isinstance(est, dict) else None
    plane_z = (ball or {}).get("support_plane_z_m") if isinstance(ball, dict) else None
    if plane_z is None and attached:
        hs = (((att or {}).get("held") or {}).get("support") or {}) if isinstance(att, dict) else {}
        if isinstance(hs, dict) and isinstance(hs.get("plane"), dict):
            plane_z = hs["plane"].get("offset_m")
    lm = FIA.contact_policy(st, links, has_target=True, has_free=False, attached=False)[0]
    to = s.get("to") or {}; met = to.get("metrics") or {}
    mv = lit(met.get("max_violation_pair")) or {}; pc = lit(met.get("path_check")) or {}
    setq_ = lambda q: (d.qpos.__setitem__(slice(None), q), mujoco.mj_kinematics(m, d))
    setq_(qplan); Tb = C.T_world_base(m, d)

    def one(q_idx, point, recorded_mm, recorded_cls=None):
        if q_idx is None or point is None:
            return None
        q_idx = int(q_idx); pb = np.asarray(point, float)[None]
        if q_idx >= S:
            return {"query": q_idx, "held_row": True, "recorded_clearance_mm": recorded_mm, "recorded_class": recorded_cls}
        ln = links[q_idx] if q_idx < len(links) else None; r = float(radii[q_idx])
        dmain = float(np.asarray(field.distance(pb))[0])
        is_mask = bool(mask_on and ln in auth)
        dused = float(np.asarray(field.target_free_distance(pb))[0]) if is_mask else dmain
        dobj = float(tree.query(pb)[0][0]) if tree is not None else None
        cls = "obstacle"
        if dobj is not None and abs(dused - dobj) <= vox:
            cls = "target"
        if cls == "obstacle" and plane_z is not None and abs(dused - (pb[0, 2] - float(plane_z))) <= 2 * vox:
            cls = "support"
        e1_applies = bool(cls == "target" and not is_mask and lm is not None)
        margin = esdf_margin
        if e1_applies:
            margin = float(lm[q_idx])
        if cls == "obstacle":
            margin = max(margin, OBST)
        path_dropped = bool(dobj is not None and abs(dmain - dobj) <= vox and not is_mask)
        pw = (Tb[:3, :3] @ pb[0]) + Tb[:3, 3]
        lab, tv = truth(pw[None], r)
        clear = dused - r - margin
        return {"query": q_idx, "link": ln, "radius_mm": 1e3 * r, "authorized": ln in auth, "mask_on": mask_on, "reads_target_free": is_mask,
                "d_main_mm": 1e3 * dmain, "d_used_mm": 1e3 * dused, "d_object_mm": None if dobj is None else 1e3 * dobj,
                "class": cls, "e1_margin_applies": e1_applies, "f1_exempt": is_mask, "margin_mm": 1e3 * margin,
                "path_target_row_dropped": path_dropped, "recon_clearance_mm": 1e3 * clear, "recorded_clearance_mm": recorded_mm,
                "fidelity_mm": None if recorded_mm is None else abs(1e3 * clear - recorded_mm), "recorded_class": recorded_cls,
                "truth_nearest": lab, "truth_mm": tv, "plane_z": plane_z}
    out = {"attached": attached, "mask_on": mask_on, "authorized_links": sorted(auth), "voxel_mm": 1e3 * vox}
    out["wp"] = one(mv.get("query") if mv.get("block") == "esdf" else None, mv.get("point_m"),
                    None if mv.get("clearance_m") is None else 1e3 * float(mv["clearance_m"]), mv.get("obstacle_margin_class"))
    w = pc.get("worst") or {}
    if w:
        rec_mm = None if pc.get("clearance_m") is None else 1e3 * float(pc["clearance_m"])
        if rec_mm is not None and w.get("floor_m") is not None:
            rec_mm = rec_mm + 1e3 * float(w["floor_m"])        # path clearance is reported above the floor
        out["path"] = one(w.get("query"), w.get("point_m"), rec_mm)
        out["path_target_rows_excluded"] = pc.get("target_rows_excluded"); out["path_floor_rows"] = pc.get("floor_rows")
        out["path_servo_source"] = (pc.get("servo_state") or {}).get("source")
    return out


if __name__ == "__main__":
    main()
