"""T43 Y — Y1 time line and Y2 place metrics per run (verifier, CPU only). Inputs: replay/<ARM>_<STAGE>_<name>.npz (replay_y.py,
physics replay, qpos_mismatch_rows 0), raw/server_chunks.json (server_y.py), T43/W/analysis/analysis.json (analyze_t43w.py).

usage: y_metrics.py OUT_JSON

Row convention: control row i has t = t_step; its post-state = replay state index i + 1 (state after its 33 sub-steps).
Crate frame = crate body pose (replay). Apple distances use the apple collision-mesh vertices (geom apple_geom, world pose from the
apple body pose) against the crate collision boxes (point-to-OBB signed distance, min over vertices) — mj_geomDistance on the mesh
was not repeatable (same pose gave 0.0 and 63.4 mm), so it is not used.
  apple_floor_gap_mm  = min over vertices of the signed distance to box crate_floor (< 0 = below its top face).
  apple_crate_gap_mm  = min over vertices and all 13 crate boxes.
Time line (all t = control-row t_step):
  grasp_t, place_t           analysis.json (analyze_e0 rule: grasp = planning-row apple z >= z0 + 50 mm; place = apple centre inside the
                             crate box at the LAST planning row; place_t = first planning row from which the rule holds — analysis value).
  lift_t                     analysis hurdle.lift_t.   attach_t = first server chunk with grasp.attach true (its t).
  over_crate_t               first row whose post-state apple centre has crate-local |x| < 0.077 and |y| < 0.137 (any z).
  in_crate_t                 first row whose post-state apple centre is inside the full place box (also -0.052 < z < 0.075).
  hold_intervals             analysis hold_t chunks, merged when consecutive (t .. t + 7).
  policy_open_t              first chunk t >= attach_t whose REFERENCE (policy) gripper rows 0-7 reach >= 0.9 (1 = open).
  applied_open_t             first control row t >= attach_t with applied gripper command >= 0.9.
  measured_open_t            first row t >= attach_t whose post-state finger opening (finger joint / open value) >= 0.9.
  last_apple_hand_contact_t  last row with >= 1 apple-hand MuJoCo contact sub-step (hand = palm + both fingers of the active arm).
  release_mode               'open_cmd' if applied_open_t <= release_t, else 'closed_cmd' (contact lost while the applied command was closed).
  apple_obstacle_substeps_10rows_before_release  apple-obstacle (hurdle bar) contact sub-steps on rows release_t - 10 .. release_t.
  release_t                  last_apple_hand_contact_t + 1 if that row exists before the run's end (no apple-hand contact afterwards),
                             else None. release_with_cmd = applied gripper command at release_t; release_open_norm = measured opening there.
  apple_left_hand_t          analysis hurdle rule (apple to finger-midpoint distance > its value at grasp_t + 30 mm).
  server_detach_t            first chunk t with grasp.detach true; server_state_after_attach = ordered distinct grasp.state values.
  first contact rows         apple-crate, hand-crate, robot_other-crate (physics replay, any sub-step).
Y2 place metrics (rows up to release_t; only when release_t exists):
  release_apple_floor_gap_mm   apple_floor_gap_mm at the post-state of row release_t - 1 (last state with hand contact).
  release_apple_centre_above_floor_mm  crate-local apple centre z - (-0.052) at that state.
  release_v_horiz_mm_s / release_v_vert_mm_s   apple free-joint linear velocity (world) at that state, |v_xy| and v_z.
  post_release_max_speed_mm_s  max |v| over the 15 rows after release_t.  post_release_horiz_travel_mm = |xy_crate(end) - xy_crate(release)|,
                             end = min(release_t + 30, last row) (2 s).  post_release_drop_mm = z drop over the same window.
  in_hand_slip_max_mm / in_hand_slip_at_release_mm  |apple_tcp(t) - apple_tcp(attach_t)|, apple_tcp = R_tcp^T (apple - tcp), t in [attach_t, release_t).
  pre_release_apple_crate_substeps / pre_release_hand_crate_substeps   contact sub-steps on rows < release_t.
  tcp_speed_at_open_mm_s       |tcp(post row) - tcp(pre row)| * 15 Hz at applied_open_t; tcp_speed_4rows_before_open_mm_s = mean over the 4 rows before.
  tcp_height_above_floor_at_open_mm  crate-local TCP z - (-0.052) at applied_open_t.
  open_minus_release_rows      release_t - applied_open_t.
  place_window_deflection_mm   analysis server_chunks tcp_deflection_mm (max over rows 0-7 of |TCP(refined) - TCP(reference)|) over chunks with
                             over_crate_t - 8 < t <= release_t: median and max; n chunks.
Flag-linked quantities (per run, from analysis.json): attach held query (body-cover radii, fit radius, n_points), closing carve-out ball radius,
  decision_reason_counts['closing_kept_geometry'], latched_hold_counts['True'], chunks_with_not_supported_cluster, attached_chunks, n_hold_chunks."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco

Y = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/Y'))
AN = _ag3s_workspace_path('outputs/verify/T43/W/analysis/analysis.json')
XML = "${AG3S_PROJECT_ROOT}-t43w/robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml"
FLOOR_TOP = -0.052
HZ = 15.0


def qmat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


class Geo:
    def __init__(self):
        m = mujoco.MjModel.from_xml_path(XML)
        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        ga = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom"); mid = int(m.geom_dataid[ga])
        self.av = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        self.a_pos = m.geom_pos[ga].copy(); self.a_R = qmat(m.geom_quat[ga])
        cb = B("crate")
        self.boxes = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) == cb and (m.geom_contype[g] or m.geom_conaffinity[g]):
                self.boxes.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), m.geom_pos[g].copy(), qmat(m.geom_quat[g]), m.geom_size[g].copy()))

    def apple_gaps(self, ap, aq, cp, cq):
        Ra, Rc = qmat(aq), qmat(cq)
        V = ap + Ra @ self.a_pos + (Ra @ self.a_R @ self.av.T).T          # world vertices
        Vc = (Rc.T @ (V - cp).T).T                                          # crate frame
        out = {}
        for name, gp, gR, h in self.boxes:
            q = (gR.T @ (Vc - gp).T).T; dv = np.abs(q) - h
            sd = np.linalg.norm(np.maximum(dv, 0), axis=1) + np.minimum(dv.max(axis=1), 0)
            out[name] = float(sd.min())
        return out["crate_floor"], min(out.values())


def first(it):
    return next(iter(it), None)


def merge_holds(ts):
    out = []
    for t in ts:
        if out and t == out[-1][1] + 1:
            out[-1][1] = t + 7
        else:
            out.append([t, t + 7])
    return out


def run(key, a, S, G):
    arm, stage, name = key.split("/")
    rz = Y / "replay" / f"{arm}_{stage}_{name}.npz"
    r = a["runs"][key]; sc = (S.get(key) or {}).get("chunks", [])
    z = np.load(rz); t = z["t"]; N = len(t)
    post = lambda k, i: z[k][i + 1]
    Rc = [qmat(q) for q in z["crate_quat"]]
    apc = np.array([Rc[i].T @ (z["apple_pos"][i] - z["crate_pos"][i]) for i in range(N + 1)])
    tcpc = np.array([Rc[i].T @ (z["tcp"][i] - z["crate_pos"][i]) for i in range(N + 1)])
    gaps = np.array([G.apple_gaps(z["apple_pos"][i], z["apple_quat"][i], z["crate_pos"][i], z["crate_quat"][i]) for i in range(N + 1)])
    a_tcp = np.array([z["tcp_R"][i].T @ (z["apple_pos"][i] - z["tcp"][i]) for i in range(N + 1)])
    inxy = (np.abs(apc[1:, 0]) < 0.077) & (np.abs(apc[1:, 1]) < 0.137)
    inbox = inxy & (apc[1:, 2] > -0.052) & (apc[1:, 2] < 0.075)
    o = {"key": key, "success": r["success"], "grasp": r["grasp"], "place": r["place"], "grasp_t": r["grasp_t"], "place_t": r["place_t"],
         "n_rows": int(N), "lift_t": (r.get("hurdle") or {}).get("lift_t"), "apple_left_hand_t": (r.get("hurdle") or {}).get("apple_left_hand_t")}
    o["attach_t"] = first(c["t"] for c in sc if c.get("attach"))
    o["server_detach_t"] = first(c["t"] for c in sc if c.get("detach"))
    seqst = []
    for c in sc:
        if o["attach_t"] is not None and c["t"] >= o["attach_t"] and (not seqst or seqst[-1][0] != c["grasp_state"]):
            seqst.append([c["grasp_state"], c["t"]])
    o["server_states_after_attach"] = seqst
    o["over_crate_t"] = int(t[np.argmax(inxy)]) if inxy.any() else None
    o["in_crate_t"] = int(t[np.argmax(inbox)]) if inbox.any() else None
    o["hold_intervals"] = merge_holds(r["hold_t"]); o["n_hold_chunks"] = r["n_hold_chunks"]; o["trailing_hold_chunks"] = r["trailing_hold_chunks"]
    at = o["attach_t"] if o["attach_t"] is not None else (r["grasp_t"] or 0)
    o["policy_open_t"] = first(c["t"] for c in sc if c["t"] >= at and c["grip_reference_rows0_7"][1] >= 0.9)
    o["applied_open_t"] = first(int(t[i]) for i in range(N) if t[i] >= at and z["grip_cmd"][i] >= 0.9)
    o["measured_open_t"] = first(int(t[i]) for i in range(N) if t[i] >= at and z["finger_open_norm"][i + 1] >= 0.9)
    if not r["grasp"]:
        o["policy_open_t"] = o["applied_open_t"] = o["measured_open_t"] = None
    ah = np.where(z["cnt_apple-hand"] > 0)[0]
    o["last_apple_hand_contact_t"] = int(t[ah[-1]]) if len(ah) else None
    rel_i = (int(ah[-1]) + 1) if (len(ah) and ah[-1] + 1 < N and r["grasp"]) else None
    o["release_t"] = int(t[rel_i]) if rel_i is not None else None
    for c in ("apple-crate", "hand-crate", "robot_other-crate"):
        w = np.where(z["cnt_" + c] > 0)[0]
        o[f"first_{c}_contact_t"] = int(t[w[0]]) if len(w) else None
        o[f"n_{c}_contact_rows"] = int(len(w)); o[f"{c}_max_force_N"] = float(z["maxf_" + c].max()); o[f"{c}_min_dist_mm"] = float(np.nanmin(z["mind_" + c]) * 1e3) if len(w) else None
    o["apple_in_crate_end_mm"] = (1e3 * apc[-1]).tolist(); o["apple_floor_gap_end_mm"] = 1e3 * gaps[-1, 0]
    o["finger_open_end"] = float(z["finger_open_norm"][-1]); o["apple_finger_mid_end_mm"] = 1e3 * float(np.linalg.norm(z["apple_pos"][-1] - z["finger_mid"][-1]))
    # key-time snapshots (post-state of that row)
    def snap(tt):
        if tt is None:
            return None
        i = int(np.searchsorted(t, tt)); i = min(i, N - 1)
        return {"t": int(t[i]), "apple_in_crate_mm": (1e3 * apc[i + 1]).tolist(), "apple_floor_gap_mm": 1e3 * gaps[i + 1, 0], "apple_crate_gap_mm": 1e3 * gaps[i + 1, 1],
                "grip_cmd": float(z["grip_cmd"][i]), "finger_open": float(z["finger_open_norm"][i + 1]), "hold": bool(z["hold"][i])}
    o["snap"] = {k: snap(o[k]) for k in ("grasp_t", "attach_t", "over_crate_t", "in_crate_t", "place_t", "policy_open_t", "applied_open_t", "release_t")}
    if o["hold_intervals"]:
        o["snap"]["first_hold"] = snap(o["hold_intervals"][0][0])
    if rel_i is not None:
        j = rel_i  # state index of the post-state of row rel_i - 1
        v = z["apple_linvel"][j]
        o["release_apple_floor_gap_mm"] = 1e3 * gaps[j, 0]; o["release_apple_centre_above_floor_mm"] = 1e3 * (apc[j, 2] - FLOOR_TOP)
        o["release_v_horiz_mm_s"] = 1e3 * float(np.linalg.norm(v[:2])); o["release_v_vert_mm_s"] = 1e3 * float(v[2])
        o["release_with_cmd"] = float(z["grip_cmd"][rel_i]); o["release_open_norm"] = float(z["finger_open_norm"][j])
        o["release_in_hold"] = bool(z["hold"][rel_i])
        o["release_mode"] = "open_cmd" if (o["applied_open_t"] is not None and o["applied_open_t"] <= o["release_t"]) else "closed_cmd"
        o["apple_obstacle_substeps_10rows_before_release"] = int(z["cnt_apple-obstacle"][max(0, rel_i - 10):rel_i + 1].sum())
        k2 = min(rel_i + 15, N)
        o["post_release_max_speed_mm_s"] = 1e3 * float(np.linalg.norm(z["apple_linvel"][j:k2 + 1], axis=1).max())
        e = min(rel_i + 30, N)
        o["post_release_horiz_travel_mm"] = 1e3 * float(np.linalg.norm(apc[e, :2] - apc[j, :2])); o["post_release_drop_mm"] = 1e3 * float(apc[j, 2] - apc[e, 2])
        ia = int(np.searchsorted(t, at)) + 1
        sl = np.linalg.norm(a_tcp[ia:j + 1] - a_tcp[ia], axis=1)
        o["in_hand_slip_max_mm"] = 1e3 * float(sl.max()) if len(sl) else None; o["in_hand_slip_at_release_mm"] = 1e3 * float(sl[-1]) if len(sl) else None
        o["pre_release_apple_crate_substeps"] = int(z["cnt_apple-crate"][:rel_i].sum()); o["pre_release_hand_crate_substeps"] = int(z["cnt_hand-crate"][:rel_i].sum())
        if o["applied_open_t"] is not None:
            io = int(np.searchsorted(t, o["applied_open_t"]))
            sp = lambda i: 1e3 * HZ * float(np.linalg.norm(z["tcp"][i + 1] - z["tcp"][i]))
            o["tcp_speed_at_open_mm_s"] = sp(io); o["tcp_speed_4rows_before_open_mm_s"] = float(np.mean([sp(i) for i in range(max(0, io - 4), io)])) if io > 0 else None
            o["tcp_height_above_floor_at_open_mm"] = 1e3 * float(tcpc[io, 2] - FLOOR_TOP)
            o["apple_floor_gap_at_open_mm"] = 1e3 * gaps[io, 0]
            o["open_minus_release_rows"] = o["release_t"] - o["applied_open_t"]
        lo = (o["over_crate_t"] if o["over_crate_t"] is not None else at) - 8
        dfl = [c["tcp_deflection_mm"] for c in r.get("server_chunks", []) if c.get("t") is not None and lo < c["t"] <= o["release_t"] and c.get("tcp_deflection_mm") is not None]
        o["place_window_deflection_mm"] = {"n": len(dfl), "median": float(np.median(dfl)) if dfl else None, "max": float(max(dfl)) if dfl else None}
    ah_ = r.get("attach_held") or []
    o["flag_linked"] = {"attach_held": ah_[0] if ah_ else None, "closing_carve_ball_radius_mm": (r.get("closing") or {}).get("carve_ball_radius_mm"),
                        "n_closing_kept_geometry": (r.get("decision_reason_counts") or {}).get("closing_kept_geometry", 0),
                        "n_latched_hold_true": (r.get("latched_hold_counts") or {}).get("True", 0),
                        "chunks_with_not_supported_cluster": r.get("chunks_with_not_supported_cluster"), "attached_chunks": r.get("attached_chunks"),
                        "manipulated_id_switches": r.get("manipulated_id_switches"), "decision_reason_counts": r.get("decision_reason_counts")}
    o["series"] = {"t": t.tolist(), "apple_in_crate_z_mm": (1e3 * apc[1:, 2]).tolist(), "apple_floor_gap_mm": (1e3 * gaps[1:, 0]).tolist(),
                   "apple_crate_gap_mm": (1e3 * gaps[1:, 1]).tolist(), "finger_open": z["finger_open_norm"][1:].tolist(), "grip_cmd": z["grip_cmd"].tolist(),
                   "apple_hand_substeps": z["cnt_apple-hand"].tolist(), "apple_crate_substeps": z["cnt_apple-crate"].tolist(), "hand_crate_substeps": z["cnt_hand-crate"].tolist(),
                   "hold": z["hold"].astype(int).tolist(), "apple_tcp_slip_mm": (1e3 * np.linalg.norm(a_tcp[1:] - a_tcp[int(np.searchsorted(t, at)) + 1], axis=1)).tolist(),
                   "apple_speed_mm_s": (1e3 * np.linalg.norm(z["apple_linvel"][1:], axis=1)).tolist(), "apple_in_crate_xy_mm": (1e3 * apc[1:, :2]).tolist()}
    return o


def divergence(a, k4, k5):
    """first control step where max_j |executed_actions V4 - V5| > 1e-3 (analyze_t39 DIV_TOL), and its chunk t."""
    e4 = np.load(pathlib.Path(a["runs"][k4]["run_dir"]) / "traj.npz")["executed_actions"]; e5 = np.load(pathlib.Path(a["runs"][k5]["run_dir"]) / "traj.npz")["executed_actions"]
    n = min(len(e4), len(e5)); dif = np.abs(e4[:n] - e5[:n]).max(axis=1)
    w = np.where(dif > 1e-3)[0]
    return None if not len(w) else {"step": int(w[0]), "chunk_t": int(8 * (w[0] // 8)), "max_abs_rad": float(dif[w[0]])}


if __name__ == "__main__":
    a = json.load(open(AN)); S = json.load(open(Y / "raw/server_chunks.json"))["runs"]; G = Geo()
    keys = sorted(k for k in a["runs"] if k.startswith(("V4/", "V5/")))
    res = {"definitions": __doc__, "runs": {}, "divergence": {}}
    for k in keys:
        res["runs"][k] = run(k, a, S, G)
        o = res["runs"][k]
        print(k, "rel", o["release_t"], "cmd", o.get("release_with_cmd"), "aopen", o["applied_open_t"], "popen", o["policy_open_t"], "ac", o["first_apple-crate_contact_t"], "hc", o["first_hand-crate_contact_t"], flush=True)
    for k in keys:
        if k.startswith("V4/"):
            k5 = k.replace("V4", "V5")
            if k5 in a["runs"]:
                res["divergence"][k.split("/", 1)[1].replace("V4_", "")] = divergence(a, k, k5)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))
