"""T43 S (4) — right-arm vs left-arm pick, from T42-1 records only (CPU, no server, no GPU). Verifier 2026-10-05.

Inputs (read-only): outputs/verify/T42/1/runs/{E0,E3b}/noobs/<run>/frames.jsonl, <run>.robust.json (T42-1 verifier's
grasp_robust output, frozen list raw/t42_runs_frozen.json), server npz via raw/chunk_index_t42.json.
Model: ${AG3S_PROJECT_ROOT}-t42 model_transport.xml (the T42-1 client model). Arm/fruit from the run manifest.

Rows measured per run (control rows, recorded qpos = pose when that row was commanded):
  close_t   = first applied arm-gripper transition open(>=0.9) -> <0.9 (robust.json close_cmd_t[0])
  rows      = close_t ("onset"), close_t + 16 ("settled", 2 chunks later), lift_t (robust.json) when present
Geometry at a row (world frame, MuJoCo kinematics + mj_collision at that qpos, qvel 0):
  finger_mid       = mean of the two finger bodies (ee_finger_<s>1/2) xpos
  approach         = unit(finger_mid - EE_BODY_<S> xpos); tilt_deg = angle(approach, -z)
  approach_yaw_deg = atan2 of the approach's horizontal part (right arm reported mirrored: y -> -y, so left/right compare)
  closing_axis     = unit(finger1 - finger2); closing_elev_deg = angle to the horizontal plane;
                     closing_yaw_deg (mod 180, mirrored for right)
  grasp_dz_mm      = finger_mid z - fruit body z ; grasp_dxy_mm = horizontal |finger_mid - fruit|
  contacts         = MuJoCo contacts fruit geom <-> (finger / EE_BODY) geoms of the arm: count per body, contact point
                     height above the fruit centre (mm), contact dist (mm, <0 = penetration)
  finger_gap_mm    = per finger body min mj_geomDistance(fruit geom, finger geom) (cap 50 mm)
  tip_mid          = mean of the two fingertips (per finger body: collision-mesh vertex farthest along the approach);
                     tip_dz_mm / tip_dxy_mm = tip_mid relative to the fruit body centre; tip_spread_mm = tip-to-tip distance
  opening_meas     = |qpos[gripper_finger_<s>1]| / 0.045
E3b grasp chunks (planning chunks whose 8 control rows intersect [close_t - 16, close_t + 15]):
  server summary: TO status / returned / max_violation_pair / initial violation / limit overshoot, grasp record, exclusion,
  finger_joints, window links; reference vs refined over the 8 executed steps: used-arm joint max |d| (deg), gripper column
  max |d|, finger_mid position difference (mm, MuJoCo FK with the chunk's arm joints substituted into the planning qpos).
Whole-run audit (every E3b chunk): grasp.hand, grasp.parent_link/hold_link, attached.parent_link, exclusion.authorized_links,
  finger_joints.applied.constraint joints vs opening_norm, exec_latch_signal.hand, window.links set, worst-row link names.
usage: measure_pick.py [N_WORKERS]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, math, pathlib, ast
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, mujoco
from multiprocessing import Pool

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S/raw'))
RUNS = json.loads((OUT / "t42_runs_frozen.json").read_text())
IDX = json.loads((OUT / "chunk_index_t42.json").read_text())
XML = "${AG3S_PROJECT_ROOT}-t42/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
QADR = {"apple": 38, "banana": 45, "orange": 52, "pear": 59, "crate": 31, "grip_left": 27, "grip_right": 18}
OPEN_M = 0.045
_M = {}


def model():
    if not _M:
        m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        coll = lambda b: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == B(b) and (m.geom_contype[g] or m.geom_conaffinity[g])]
        J = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): int(m.jnt_qposadr[i]) for i in range(m.njnt)}
        _M.update(m=m, d=d, B=B, coll=coll, J=J)
    return _M


def jl(o):
    if isinstance(o, dict): return {str(k): jl(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jl(v) for v in o]
    if isinstance(o, np.ndarray): return jl(o.tolist())
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


def lit(x):
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except Exception: return x
    return x


def unit(v): v = np.asarray(v, float); n = np.linalg.norm(v); return v / n if n > 0 else v


def geom_vertices_world(g):
    M = model(); m, d = M["m"], M["d"]
    did = int(m.geom_dataid[g]); va, nv = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
    return d.geom_xpos[g] + m.mesh_vert[va:va + nv].astype(float) @ d.geom_xmat[g].reshape(3, 3).T


def geom_at(q, arm, fruit):
    M = model(); m, d = M["m"], M["d"]; s = "L" if arm == "left" else "R"; sl = s.lower()
    d.qpos[:] = q; d.qvel[:] = 0
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_collision(m, d)
    ee = M["B"](f"EE_BODY_{s}"); f1 = M["B"](f"ee_finger_{sl}1"); f2 = M["B"](f"ee_finger_{sl}2"); fb = M["B"](fruit)
    pe, p1, p2, pf = d.xpos[ee].copy(), d.xpos[f1].copy(), d.xpos[f2].copy(), d.xpos[fb].copy()
    mid = 0.5 * (p1 + p2)
    a = unit(mid - pe); c = unit(p1 - p2)
    mir = np.array([1.0, -1.0, 1.0]) if arm == "right" else np.ones(3)
    am, cm = a * mir, c * mir
    out = {
        "fruit_pos": pf, "ee_pos": pe, "finger_mid": mid, "approach": a, "closing_axis": c,
        "tilt_deg": float(np.degrees(np.arccos(np.clip(-a[2], -1, 1)))),
        "approach_yaw_deg_mirrored": float(np.degrees(np.arctan2(am[1], am[0]))),
        "closing_elev_deg": float(np.degrees(np.arcsin(np.clip(abs(c[2]), 0, 1)))),
        "closing_yaw_deg_mirrored_mod180": float(np.degrees(np.arctan2(cm[1], cm[0])) % 180.0),
        "grasp_dz_mm": float(1e3 * (mid[2] - pf[2])),
        "grasp_dxy_mm": float(1e3 * np.linalg.norm(mid[:2] - pf[:2])),
        "fruit_offset_in_ee_frame_mm": (d.xmat[ee].reshape(3, 3).T @ (pf - pe) * 1e3),
        "ee_xmat": d.xmat[ee].reshape(3, 3).copy(),
        "opening_meas": float(abs(q[QADR["grip_" + arm]]) / OPEN_M),
    }
    fg = set(M["coll"](fruit)); bodies = {f"EE_BODY_{s}": set(M["coll"](f"EE_BODY_{s}")), f"ee_finger_{sl}1": set(M["coll"](f"ee_finger_{sl}1")),
                                          f"ee_finger_{sl}2": set(M["coll"](f"ee_finger_{sl}2"))}
    cons = []
    for i in range(d.ncon):
        con = d.contact[i]; g1, g2 = int(con.geom1), int(con.geom2)
        for bn, gs in bodies.items():
            if (g1 in fg and g2 in gs) or (g2 in fg and g1 in gs):
                p = np.asarray(con.pos, float)
                cons.append({"body": bn, "dz_from_fruit_centre_mm": float(1e3 * (p[2] - pf[2])),
                             "dxy_from_fruit_centre_mm": float(1e3 * np.linalg.norm(p[:2] - pf[:2])),
                             "dist_mm": float(1e3 * con.dist), "pos": p})
    # fingertips: per finger body, the collision-mesh vertex farthest along the approach direction
    tips = []
    for bn in (f"ee_finger_{sl}1", f"ee_finger_{sl}2"):
        V = np.concatenate([geom_vertices_world(g) for g in bodies[bn]])
        tips.append(V[np.argmax(V @ a)])
    tip = 0.5 * (tips[0] + tips[1])
    out["tip_mid"] = tip
    out["tip_dz_mm"] = float(1e3 * (tip[2] - pf[2])); out["tip_dxy_mm"] = float(1e3 * np.linalg.norm(tip[:2] - pf[:2]))
    out["tip_spread_mm"] = float(1e3 * np.linalg.norm(tips[0] - tips[1]))
    out["contacts"] = cons
    out["n_contacts_by_body"] = {bn: sum(1 for c_ in cons if c_["body"] == bn) for bn in bodies}
    ft = np.zeros(6); gap = {}
    for bn in (f"ee_finger_{sl}1", f"ee_finger_{sl}2", f"EE_BODY_{s}"):
        gap[bn] = float(1e3 * min(mujoco.mj_geomDistance(m, d, g1, g2, 0.05, ft) for g1 in fg for g2 in bodies[bn]))
    out["finger_gap_mm"] = gap
    return out


ARM_COLS = {"left": list(range(0, 7)), "right": list(range(8, 15))}
GRIP_COL = {"left": 7, "right": 15}


def q_with_chunk_row(q, row, arm):
    M = model(); q = np.array(q, float).copy()
    for i, col in enumerate(ARM_COLS[arm]):
        q[M["J"][f"{arm}_arm_{i}"]] = row[col]
    return q


def finger_mid_at(q, arm):
    M = model(); m, d = M["m"], M["d"]; sl = "l" if arm == "left" else "r"
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    return 0.5 * (d.xpos[M["B"](f"ee_finger_{sl}1")] + d.xpos[M["B"](f"ee_finger_{sl}2")])


def chunk_record(npz, plan, ctrl_rows, arm):
    with np.load(npz) as z:
        s = json.loads(str(z["summary_json"])); ref = np.asarray(z["reference_chunk"], float); fin = np.asarray(z["refined_chunk"], float)
    to = s.get("to") or {}; met = to.get("metrics") or {}
    mvp = lit(met.get("max_violation_pair")); g = s.get("grasp") or {}
    fj = s.get("finger_joints") or {}
    q0 = np.asarray(plan["qpos"], float)
    tcp_d = [float(1e3 * np.linalg.norm(finger_mid_at(q_with_chunk_row(q0, ref[k], arm), arm) - finger_mid_at(q_with_chunk_row(q0, fin[k], arm), arm)))
             for k in range(8)]
    cols = ARM_COLS[arm]
    att = (g.get("attached") or {})
    win = lit(((s.get("esdf_stats") or {}).get("window")))
    rec = {
        "seq": plan["seq"], "t_step": plan["t_step"],
        "gate": (plan.get("gate") or {}).get("action"), "hold_kinds": (plan.get("gate") or {}).get("hold_kinds"),
        "trajopt_status": (plan.get("verdict") or {}).get("trajopt_status"), "max_violation_m": (plan.get("verdict") or {}).get("max_violation_m"),
        "to_returned": met.get("returned"), "to_clearance_mm": None if met.get("clearance_m") is None else 1e3 * met["clearance_m"],
        "to_initial_violation_mm": None if met.get("initial_violation_m") is None else 1e3 * met["initial_violation_m"],
        "initial_limit_overshoot": lit(met.get("initial_limit_overshoot")),
        "worst_row": None if not isinstance(mvp, dict) else {k: mvp.get(k) for k in ("link", "obstacle", "tier", "clearance_m", "step", "query")},
        "executed": [bool(c.get("executed", True)) for c in ctrl_rows],
        "ref_vs_refined_arm_max_deg": float(np.degrees(np.abs(ref[:8, cols] - fin[:8, cols]).max())),
        "ref_vs_refined_other_arm_max_deg": float(np.degrees(np.abs(ref[:8, ARM_COLS["right" if arm == "left" else "left"]] - fin[:8, ARM_COLS["right" if arm == "left" else "left"]]).max())),
        "ref_vs_refined_gripper_max_abs": float(np.abs(ref[:8, GRIP_COL[arm]] - fin[:8, GRIP_COL[arm]]).max()),
        "ref_vs_refined_tcp_mm_per_step": tcp_d,
        "ref_gripper_cmd": ref[:8, GRIP_COL[arm]].tolist(),
        "grasp": {k: g.get(k) for k in ("mode", "hand", "parent_link", "hold_link", "state", "transition", "attach", "detach", "rejected", "attach_revoked", "note")},
        "attached_parent_link": att.get("parent_link"),
        "authorized_links": (s.get("exclusion") or {}).get("authorized_links"),
        "exclusion_source": (s.get("exclusion") or {}).get("source"),
        "finger_opening_norm": fj.get("opening_norm"), "finger_joints_constraint": ((fj.get("applied") or {}).get("constraint")),
        "exec_latch_signal": s.get("exec_latch_signal"),
        "window_links": sorted(set(win.get("links") or [])) if isinstance(win, dict) else None,
        "has_target": s.get("has_target"), "manipulated_state": (s.get("manipulated") or {}).get("state"),
        "target_centroid": None,
    }
    return rec


def audit_run(chunks, arm):
    """every chunk of an E3b run: is the server treating the used hand as the hand?"""
    sl = "l" if arm == "left" else "r"; other = "r" if sl == "l" else "l"
    hand_links = {f"ee_{arm}", f"ee_finger_{sl}1", f"ee_finger_{sl}2"}
    c = {"n_chunks": 0, "grasp_hand": {}, "grasp_parent_link": {}, "grasp_hold_link": {}, "attached_parent_link": {},
         "authorized_links": {}, "latch_hand": {}, "window_links": {}, "worst_row_link_kind": {}, "finger_joint_mismatch": 0,
         "finger_joint_checked": 0, "grasp_state": {}, "attach_events": 0, "detach_events": 0, "attach_revoked": 0,
         "worst_row_other_hand": 0, "initial_position_overshoot_nonzero": 0}
    def inc(dd, k): k = json.dumps(k) if not isinstance(k, str) else k; dd[k] = dd.get(k, 0) + 1
    for seq, ch in sorted(chunks.items(), key=lambda kv: int(kv[0])):
        with np.load(ch["npz"]) as z:
            s = json.loads(str(z["summary_json"]))
        c["n_chunks"] += 1
        g = s.get("grasp") or {}
        inc(c["grasp_hand"], str(g.get("hand"))); inc(c["grasp_parent_link"], str(g.get("parent_link"))); inc(c["grasp_hold_link"], str(g.get("hold_link")))
        inc(c["grasp_state"], str(g.get("state")))
        if g.get("attach"): c["attach_events"] += 1
        if g.get("detach"): c["detach_events"] += 1
        if g.get("attach_revoked"): c["attach_revoked"] += 1
        inc(c["attached_parent_link"], str((g.get("attached") or {}).get("parent_link")))
        inc(c["authorized_links"], sorted((s.get("exclusion") or {}).get("authorized_links") or []))
        inc(c["latch_hand"], str((s.get("exec_latch_signal") or {}).get("hand")))
        win = lit((s.get("esdf_stats") or {}).get("window"))
        inc(c["window_links"], sorted(set(win.get("links") or [])) if isinstance(win, dict) else None)
        mvp = lit(((s.get("to") or {}).get("metrics") or {}).get("max_violation_pair"))
        if isinstance(mvp, dict):
            ln = str(mvp.get("link"))
            kind = ("attached" if ln.startswith("attached:") else "used_hand" if ln in hand_links else
                    "other_hand" if ln in {f"ee_{'right' if arm == 'left' else 'left'}", f"ee_finger_{other}1", f"ee_finger_{other}2"} else ln)
            inc(c["worst_row_link_kind"], kind)
            if kind == "other_hand": c["worst_row_other_hand"] += 1
        ov = lit(((s.get("to") or {}).get("metrics") or {}).get("initial_limit_overshoot"))
        if isinstance(ov, dict) and float(ov.get("position") or 0) > 0: c["initial_position_overshoot_nonzero"] += 1
        fj = s.get("finger_joints") or {}; on = (fj.get("opening_norm") or {}).get(arm); cj = ((fj.get("applied") or {}).get("constraint") or {})
        if on is not None and f"gripper_finger_{sl}1" in cj:
            c["finger_joint_checked"] += 1
            if abs(abs(cj[f"gripper_finger_{sl}1"]) - on * OPEN_M) > 1e-6 or abs(cj[f"gripper_finger_{sl}1"] + cj[f"gripper_finger_{sl}2"]) > 1e-9:
                c["finger_joint_mismatch"] += 1
    return c


def one(key):
    r = RUNS[key]; cond, name = key.split("/")
    rd = pathlib.Path(r["run_dir"]); arm, fruit = r["used_arm"], r["target_fruit"]
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = {c["t_step"]: c for c in rows if c["kind"] == "control"}; plan = [p for p in rows if p["kind"] == "planning"]
    out = {"key": key, "cond": cond, "arm": arm, "fruit": fruit, "episode": int(name.split("_ep")[1].split("_")[0]),
           "seed": int(name.split("_s")[-1]), "grasp": r["grasp"], "place": r["place"], "success": r["success"],
           "grasp_t": r["grasp_t"], "lift_t": r.get("lift_t"), "close_cmd_t": r.get("close_cmd_t"), "rows": {}}
    q_plan0 = np.asarray(plan[0]["qpos"], float)
    out["fruit_z0"] = float(q_plan0[QADR[fruit] + 2])
    ct = (r.get("close_cmd_t") or [None])[0]
    out["close_t"] = ct
    tsel = {}
    if ct is not None:
        tsel["onset"] = ct; tsel["settled"] = ct + 16
    if r.get("lift_t") is not None:
        tsel["lift"] = r["lift_t"]
    for nm, t in tsel.items():
        if t in ctrl:
            g = geom_at(np.asarray(ctrl[t]["qpos"], float), arm, fruit)
            g.pop("ee_xmat", None)
            out["rows"][nm] = {"t": t, **g}
    # pre-close approach path: finger_mid relative to the fruit, every 4th row in [close_t-48, close_t]
    if ct is not None:
        path = []
        for t in range(max(0, ct - 48), ct + 1, 4):
            if t in ctrl:
                g = geom_at(np.asarray(ctrl[t]["qpos"], float), arm, fruit)
                path.append({"t": t, "rel_mm": (g["finger_mid"] - g["fruit_pos"]) * 1e3, "tilt_deg": g["tilt_deg"]})
        out["approach_path"] = path
    if cond == "E3b":
        ix = IDX.get(f"${AG3S_PROJECT_ROOT}/outputs/verify/T42/1/runs/E3b/noobs/{name}")
        if ix is None: out["server"] = None; return jl(out)
        out["pairing_refined_vs_actions_max_abs"] = ix["refined_vs_actions_max_abs"]
        out["audit"] = audit_run(ix["chunks"], arm)
        if ct is not None:
            recs = []
            for p in plan:
                t0 = p["t_step"]
                if t0 + 7 >= ct - 16 and t0 <= ct + 15:
                    ch = ix["chunks"].get(str(p["seq"]))
                    if ch is None: continue
                    rows8 = [ctrl[t] for t in range(t0, t0 + 8) if t in ctrl]
                    recs.append(chunk_record(ch["npz"], p, rows8, arm))
            out["grasp_chunks"] = recs
    return jl(out)


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    keys = sorted(RUNS)
    with Pool(nw) as pool:
        res = pool.map(one, keys, chunksize=1)
    (OUT / "pick_runs.json").write_text(json.dumps({"definitions": __doc__, "runs": {r["key"]: r for r in res}}, indent=0))
    print("done", len(res))
