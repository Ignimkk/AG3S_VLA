"""T43Z3V (copy of T43/W/p/analyze_t43w.py; arm V5D only + slip fields) — per-run analysis of runs/<ARM>/<STAGE>/<ARM>_ep<EP>_s<SEED> (verifier, CPU only, MUJOCO_GL=osmesa).
ARM = E0 | V1 | V3 | V4 | V5 (T43W), STAGE = H (T43 P hurdle) | B (no obstacle). Only runs with NAME.done are read.

usage: analyze_t43w.py OUT_JSON           (env T43W_ROOT overrides the runs root)

Reused (unchanged rules, copies in this p/ dir):
  analyze_t39.run_metrics : success / grasp / place (analyze_e0 rules), HOLD chunks (planning row with not all control rows executed,
                            kinds = gate.hold_kinds), verdicts, chunk times (server verdict.timing_ms, client round trip), first chunk excluded,
                            share of round trips > 533.3 ms. do_contact = False (grasp-contact height is not measured here).
  hurdle_metrics.run_hurdle : per control row, hurdle distance (client pair_distance, GJK; capped 0.30 m) of
                            Af = fingers ee_finger_l1/l2, Ae = palm EE_BODY_L, B = apple, C = every other robot collision geom;
                            phases before_lift / carry_lift_to_place_or_end / after / all (lift_t, place_t rules in its doc).
  server pairing          : V arms — the server segment (t_step restarts at 1) whose every refined_chunk equals the run's planning actions
                            (<= 1e-6) with as many chunks as planning rows (analyze_t41 rule).
Added here:
  contact rows            : per set, control rows with distance <= 0 and the phase of the first one.
  trailing HOLD           : number of consecutive HOLD chunks that end at the run's last planning row ("hold to the end"), and the
                            longest run of consecutive HOLD chunks anywhere.
  deflection (V arms)     : per planning row, max over control rows 0-7 of |TCP(actions) - TCP(actions_reference)| (MuJoCo FK of the
                            left-arm columns 0-6 on that row's qpos, site left_ee) — the same quantity the avoidance arrow draws.
                            deflected chunk := that value > 5 mm (the arrow's default threshold).
  worst TO row (V arms)   : server summary to.metrics.max_violation_pair (waypoint rows) and, V2, to.metrics.path_check.worst (servo / sweep
                            path rows). Category of a row by query index q and n_robot = len(sphere_radii) of the chunk record:
                            q < 122 'base' (the E3b sphere chain), 122 <= q < n_robot 'gripper_cover' (T43 R), q >= n_robot 'held' (held object
                            query spheres; with --held-body-cover these are the body-cover spheres). The binding row of a chunk = the smaller of
                            waypoint clearance_m and path_check.clearance_m ('path' wins ties only if strictly smaller).
  focus (T43W, V arms)    : closing chunks (grasp.state closing, not attached): fingertip midpoint (active-arm finger bodies) at control
                            row 7, actions vs actions_reference, distance to the apple centre -> outward_mm = d(refined) - d(reference)
                            (> 0 = pushed away from the apple), shift_mm = |tip(refined) - tip(reference)|; carve-out ball radius
                            (esdf_stats.target_free.ball.radius_mm) on those chunks; attach chunks: held query spheres (grasp.attached.held:
                            mode, max_radius_mm, radii, body_cover radii); held-row violations: attached chunks whose waypoint worst row is
                            a held query sphere with clearance < 0 (and, V3 / V6, the path worst row).
  TO timing               : to.solve_ms, to.sqp_iterations, time_budget_hit, metrics.timing_ms (linearize / qp / check / path), initial_source."""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, json, math, os, pathlib, re, sys, datetime
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import analyze_t39 as A39  # noqa: E402
import hurdle_metrics as HM  # noqa: E402
import mujoco  # noqa: E402

ROOT = pathlib.Path(os.environ.get("T43W_ROOT", _ag3s_workspace_path('outputs/verify/T43/Z3V/runs')))
NAME = re.compile(r"^(V5D)_ep(\d+)_s(\d+)$")
N_BASE = 122
DEFLECT_MM = 5.0
LEFT_Q = list(range(20, 27))
RIGHT_Q = list(range(11, 18))
SIDE = {1834: "right"}   # T43W: active arm per episode (T43 P2); default left


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return {"_unparsed": x[:200]}
    return x


def iso(p):
    try:
        return datetime.datetime.fromisoformat(p.read_text().splitlines()[0].strip()).timestamp()
    except Exception:
        return None


def obstacle_of(start_file):
    for l in start_file.read_text().splitlines():
        if l.startswith("obstacle="):
            v = l.split("=", 1)[1]
            if v and v != ":":
                c, p = v.split(":", 1)
                return c, p
    return None


def segments(arm_dir):
    segs = []
    for rec in sorted(arm_dir.glob("server_*/server_constraints/run_*")):
        cur = []
        for f in sorted(rec.glob("chunk_*.npz")):
            ts = int(np.load(f, allow_pickle=True)["t_step"])
            if ts == 1 and cur:
                segs.append(cur); cur = []
            cur.append(f)
        if cur:
            segs.append(cur)
    return segs


def pair(segs, plan):
    acts = [np.asarray(p["actions"], np.float64) for p in plan]
    hits = []
    for sg in segs:
        if len(sg) != len(acts):
            continue
        ok = True
        for f, a in zip(sg, acts):
            r = np.asarray(np.load(f, allow_pickle=True)["refined_chunk"], np.float64)
            if r.shape != a.shape or np.abs(r - a).max() > 1e-6:
                ok = False; break
        if ok:
            hits.append(sg)
    return hits


def category(q, n_robot):
    if q is None:
        return None
    q = int(q)
    return "base" if q < N_BASE else ("gripper_cover" if q < n_robot else "held")


def phase_of(t, lift_t, end_carry):
    if lift_t is None or t < lift_t:
        return "before_lift"
    if t <= end_carry:
        return "carry"
    return "after"


def tcp_dev(plan_row, side="left"):
    M = HM.model(); m, d = M["m"], M["d"]
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee")
    qi, c0 = (LEFT_Q, 0) if side == "left" else (RIGHT_Q, 8)
    a = np.asarray(plan_row["actions"], float); ar = np.asarray(plan_row.get("actions_reference") or plan_row["actions"], float)
    out = 0.0
    for k in range(min(8, len(a))):
        P = []
        for row in (a[k], ar[k]):
            d.qpos[:] = plan_row["qpos"]; d.qpos[qi] = row[c0:c0 + 7]; mujoco.mj_kinematics(m, d); P.append(d.site_xpos[site].copy())
        out = max(out, float(np.linalg.norm(P[0] - P[1])))
    return 1e3 * out


def tip_radial(plan_row, side):
    """T43W focus metric: fingertip midpoint (finger bodies of the active arm) at the last executed control row (row 7) for
    actions vs actions_reference (arm joints only; finger joints = the planning-row qpos), distance to the apple centre
    (planning-row object_poses). Returns (d_refined_mm, d_reference_mm, |tip shift| mm)."""
    M = HM.model(); m, d = M["m"], M["d"]
    fl = "l" if side == "left" else "r"
    b1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ee_finger_{fl}1"); b2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ee_finger_{fl}2")
    qi, c0 = (LEFT_Q, 0) if side == "left" else (RIGHT_Q, 8)
    ap = np.asarray(plan_row["object_poses"]["apple"]["pos"], float)
    a = np.asarray(plan_row["actions"], float); ar = np.asarray(plan_row.get("actions_reference") or plan_row["actions"], float)
    k = min(7, len(a) - 1); T = []
    for row in (a[k], ar[k]):
        d.qpos[:] = plan_row["qpos"]; d.qpos[qi] = row[c0:c0 + 7]; mujoco.mj_kinematics(m, d); T.append(0.5 * (d.xpos[b1] + d.xpos[b2]))
    return 1e3 * float(np.linalg.norm(T[0] - ap)), 1e3 * float(np.linalg.norm(T[1] - ap)), 1e3 * float(np.linalg.norm(T[0] - T[1]))


def server_chunks(seg, plan, side="left"):
    by_seq = {p["seq"]: p for p in plan}
    out = []
    for f in seg:
        z = np.load(f, allow_pickle=True); s = json.loads(str(z["summary_json"])); n_robot = int(z["sphere_radii"].shape[0])
        to = s.get("to") or {}; m = to.get("metrics") or {}
        mv = lit(m.get("max_violation_pair")) or {}; pc = lit(m.get("path_check")) or {}
        g = s.get("grasp") or {}; att = lit(g.get("attached"))
        held = (att or {}).get("held") if isinstance(att, dict) else None
        p = by_seq.get(int(z["t_step"]))
        e = {"seq": int(z["t_step"]), "t": p["t_step"] if p else None, "n_robot_spheres": n_robot,
             "to_status": to.get("status"), "solve_ms": to.get("solve_ms"), "sqp_iterations": to.get("sqp_iterations"),
             "time_budget_hit": to.get("time_budget_hit"), "timing_ms": lit(m.get("timing_ms")),
             "initial_source": (lit(m.get("initial_source")) or {}).get("source") if m.get("initial_source") else None,
             "grasp_state": g.get("state"), "attached": bool(att), "held_body_cover": bool(isinstance(held, dict) and held.get("body_cover")),
             "wp": {"clearance_mm": None if m.get("clearance_m") is None else 1e3 * m["clearance_m"], "block": mv.get("block"),
                    "query": mv.get("query"), "link": mv.get("link"), "tier": mv.get("tier"), "class": mv.get("obstacle_margin_class"),
                    "category": category(mv.get("query"), n_robot) if mv.get("block") == "esdf" else mv.get("block")},
             "path": None}
        if pc and "clearance_m" in pc:
            w = pc.get("worst") or {}
            e["path"] = {"clearance_mm": None if pc.get("clearance_m") is None else 1e3 * pc["clearance_m"], "mode": pc.get("mode"),
                         "query": w.get("query"), "link": w.get("link"), "at_kind": w.get("at_kind"), "floor_mm": None if w.get("floor_m") is None else 1e3 * w["floor_m"],
                         "category": category(w.get("query"), n_robot), "n_configs": pc.get("n_configs"), "max_sweep_mm": pc.get("max_sweep_mm"),
                         "servo_source": (pc.get("servo_state") or {}).get("source")}
        wc = e["wp"]["clearance_mm"]; pcm = (e["path"] or {}).get("clearance_mm")
        if pcm is not None and (wc is None or pcm < wc):
            e["binding"] = {"rows": "path", "category": e["path"]["category"], "link": e["path"]["link"], "clearance_mm": pcm}
        else:
            e["binding"] = {"rows": "waypoint", "category": e["wp"]["category"], "link": e["wp"]["link"], "clearance_mm": wc}
        e["tcp_deflection_mm"] = tcp_dev(p, side) if p else None
        st_ = lit((s.get("esdf_stats") or {}).get("target_free")) or {}
        ball = st_.get("ball") if isinstance(st_, dict) else None
        e["carve_ball_radius_mm"] = (ball or {}).get("radius_mm") if isinstance(ball, dict) else None
        e["attach_now"] = bool(g.get("attach"))
        e["grasp_detach"] = bool(g.get("detach")); e["held_slip"] = g.get("held_slip"); e["grasp_state_before"] = g.get("state_before")
        _ev = g.get("evidence") or {}; e["slip_evidence"] = _ev.get("slip"); e["attach_revoked"] = g.get("attach_revoked")
        e["gripper_after"] = _ev.get("opening_after"); e["gripper_last_command"] = _ev.get("last_command")
        if e["attach_now"] and isinstance(held, dict):
            bc = held.get("body_cover") if isinstance(held.get("body_cover"), dict) else {}
            e["attach_held"] = {"mode": held.get("mode"), "max_radius_mm": held.get("max_radius_mm"), "n_spheres": len(held.get("spheres") or []),
                                "radii_mm": [x.get("radius_mm") for x in (held.get("spheres") or [])], "merged": held.get("merged"),
                                "fit_radius_mm": (held.get("fit") or {}).get("radius_mm"), "n_points": held.get("n_points"),
                                "body_cover_radii_mm": [x.get("radius_mm") for x in (bc.get("spheres") or [])], "body_cover_mode": bc.get("mode")}
        if p and e["grasp_state"] == "closing" and not e["attached"] and (p.get("object_poses") or {}).get("apple"):
            dr, dref, sh = tip_radial(p, side)
            e["closing_tip"] = {"d_refined_mm": dr, "d_reference_mm": dref, "outward_mm": dr - dref, "shift_mm": sh}
        wpc = e["wp"]
        e["held_row_violation"] = bool(e["attached"] and wpc.get("category") == "held" and (wpc.get("clearance_mm") is not None and wpc["clearance_mm"] < 0))
        e["held_row_violation_path"] = bool(e["attached"] and e["path"] and e["path"].get("category") == "held" and e["path"].get("clearance_mm") is not None and e["path"]["clearance_mm"] < 0)
        e["wp_point_m"] = mv.get("point_m"); e["wp_obstacle"] = mv.get("obstacle")
        adm = s.get("admissibility") or {}; dec = adm.get("decision") or {}
        e["decision_reason"] = dec.get("reason"); e["latched_hold"] = dec.get("latched_hold"); e["manipulated_id"] = (s.get("manipulated") or {}).get("id")
        e["n_not_supported"] = sum(1 for c in (adm.get("clusters") or []) if isinstance(c, dict) and c.get("reason") == "not_supported")
        e["deflected"] = bool(e["tcp_deflection_mm"] is not None and e["tcp_deflection_mm"] > DEFLECT_MM)
        out.append(e)
    return out


def cnt(it):
    c = {}
    for x in it:
        c[str(x)] = c.get(str(x), 0) + 1
    return dict(sorted(c.items()))


def stat(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    a = np.asarray(v, float)
    return {"n": len(a), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max()), "min": float(a.min()), "mean": float(a.mean())}


def one(arm, stage, rd):
    m = NAME.match(rd.name); ep, seed = int(m.group(2)), int(m.group(3))
    side = SIDE.get(ep, "left"); HM.set_arm(side)
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    plan = [r for r in rows if r["kind"] == "planning"]
    seg = None; n_hits = None
    if arm not in ("E0", "E0B"):
        hits = pair(segments(ROOT / arm), plan); n_hits = len(hits)
        seg = hits[0] if hits else None
    r, _ = A39.run_metrics("E3b" if arm not in ("E0", "E0B") else "E0", ep, "s", seed, rd, [seg] if seg else ([] if arm not in ("E0", "E0B") else None), set(), False)
    r["arm"] = arm; r["stage"] = stage; r["side"] = side; r["n_server_segments_matching_content"] = n_hits
    try:
        r["check"] = json.loads((rd.parent / f"{rd.name}.check.json").read_text())["checks"]
    except Exception as ex:
        r["check"] = {"error": repr(ex)}
    ob = obstacle_of(rd.parent / f"{rd.name}.start")
    if ob is not None:
        HM._M.clear()   # T43W: fresh model per run (the overbar manager resizes its geoms and refuses a model another manager resized)
        h = HM.run_hurdle(rd, ob[0], ob[1], r["grasp_t"], r["place_t"], True, seg)
        lift_t = h["lift_t"]; end_carry = r["place_t"] if r["place_t"] is not None else (h["apple_left_hand_t"] if h["apple_left_hand_t"] is not None else h["series"]["t"][-1])
        cr = {}
        for k in ("Af", "Ae", "B", "C"):
            tt = [t for t, v in zip(h["series"]["t"], h["series"][k]) if v <= 0.0]
            cr[k] = {"n_rows_le_0": len(tt), "first_t": tt[0] if tt else None, "first_phase": phase_of(tt[0], lift_t, end_carry) if tt else None}
        h["contact_rows"] = cr
        r["hurdle"] = h
    hold = [c for c in r["chunks"] if c["hold"]]
    trail = 0
    for c in reversed(r["chunks"]):
        if c["hold"]:
            trail += 1
        else:
            break
    longest = cur = 0
    for c in r["chunks"]:
        cur = cur + 1 if c["hold"] else 0; longest = max(longest, cur)
    r["trailing_hold_chunks"] = trail; r["longest_hold_run_chunks"] = longest
    if seg:
        sc = server_chunks(seg, plan, side)
        r["server_chunks"] = sc
        rest = sc[1:]
        r["to_ms_excl_first"] = {"solve": stat([c["solve_ms"] for c in rest]),
                                 **{k: stat([(c["timing_ms"] or {}).get(k) for c in rest]) for k in ("linearize", "assemble", "qp", "check", "path")}}
        r["to_status_counts"] = cnt(c["to_status"] for c in sc)
        r["time_budget_hit_count"] = sum(bool(c["time_budget_hit"]) for c in sc)
        r["initial_source_counts"] = cnt(c["initial_source"] for c in sc)
        dfl = [c for c in sc if c["deflected"]]
        r["n_deflected_chunks"] = len(dfl)
        r["deflected_binding_counts"] = cnt(f'{c["binding"]["rows"]}:{c["binding"]["category"]}' for c in dfl)
        r["deflected_wp_category_counts"] = cnt(c["wp"]["category"] for c in dfl)
        r["deflected_path_category_counts"] = cnt((c["path"] or {}).get("category") for c in dfl) if any(c["path"] for c in dfl) else {}
        r["all_binding_counts"] = cnt(f'{c["binding"]["rows"]}:{c["binding"]["category"]}' for c in sc)
        r["held_body_cover_chunks"] = sum(c["held_body_cover"] for c in sc)
        r["attached_chunks"] = sum(c["attached"] for c in sc)
        cl = [c["closing_tip"] for c in sc if c.get("closing_tip")]
        r["closing"] = {"n_chunks": len(cl), "n_outward_gt_2mm": sum(x["outward_mm"] > 2.0 for x in cl),
                        "outward_mm_max": max([x["outward_mm"] for x in cl], default=None),
                        "shift_mm_median": float(np.median([x["shift_mm"] for x in cl])) if cl else None,
                        "carve_ball_radius_mm": [c["carve_ball_radius_mm"] for c in sc if c.get("closing_tip")]}
        r["decision_reason_counts"] = cnt(c["decision_reason"] for c in sc)
        r["latched_hold_counts"] = cnt(c["latched_hold"] for c in sc)
        r["manipulated_id_switches"] = sum(1 for a_, b_ in zip(sc, sc[1:]) if a_["manipulated_id"] is not None and b_["manipulated_id"] is not None and a_["manipulated_id"] != b_["manipulated_id"])
        r["chunks_with_not_supported_cluster"] = sum(c["n_not_supported"] > 0 for c in sc)
        r["attach_held"] = [dict(c["attach_held"], seq=c["seq"], t=c["t"]) for c in sc if c.get("attach_held")]
        r["held_row_violation_chunks"] = [{"seq": c["seq"], "t": c["t"], "clearance_mm": c["wp"]["clearance_mm"], "status": c["to_status"], "tier": c["wp"]["tier"],
                                           "class": c["wp"]["class"], "point_m": c["wp_point_m"], "obstacle": c["wp_obstacle"]} for c in sc if c["held_row_violation"]]
        r["held_row_violation_path_chunks"] = [{"seq": c["seq"], "t": c["t"], "clearance_mm": c["path"]["clearance_mm"], "status": c["to_status"]} for c in sc if c["held_row_violation_path"]]
    return r


def main():
    out_p = pathlib.Path(sys.argv[1])
    res = {"definitions": __doc__, "hurdle_definitions": HM.__doc__, "root": str(ROOT), "runs": {}}
    prev = json.loads(out_p.read_text()).get("runs", {}) if out_p.exists() else {}  # cache: a run is re-analysed only if its .done changed
    for arm in ("V5D",):
        for stage in ("H", "B"):
            sd = ROOT / arm / stage
            if not sd.exists():
                continue
            for rd in sorted(p for p in sd.iterdir() if p.is_dir() and NAME.match(p.name)):
                if not (sd / f"{rd.name}.done").exists():
                    continue
                key = f"{arm}/{stage}/{rd.name}"; mt = (sd / f"{rd.name}.done").stat().st_mtime
                if key in prev and prev[key].get("_done_mtime") == mt and prev[key].get("_v") == 4:
                    res["runs"][key] = prev[key]; continue
                r = one(arm, stage, rd); r["_done_mtime"] = mt; r["_v"] = 4
                res["runs"][f"{arm}/{stage}/{rd.name}"] = r
                h = r.get("hurdle") or {}
                mins = {k: (((h.get("phases") or {}).get("all") or {}).get(k) or {}).get("min_m") for k in ("Af", "Ae", "B", "C")}
                print(arm, stage, rd.name, "S" if r["success"] else "-", "g", r["grasp"], "p", r["place"], "hold", r["n_hold_chunks"], "trail", r["trailing_hold_chunks"],
                      "min_mm", {k: (None if v is None else round(1e3 * v, 1)) for k, v in mins.items()}, "defl", r.get("n_deflected_chunks"), r.get("deflected_binding_counts"), flush=True)
                out_p.write_text(json.dumps(res, indent=1, default=float))
    out_p.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
