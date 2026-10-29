"""T43W — numbers from analysis/analysis.json (+ analysis/hold_w.json) into the partial record (and, with --final OUT, the numbers block
of the final verify json). Numbers and paths only. Keeps the keys update_partial.py writes (runs, run_counts, updated).

summary per arm (hurdle stage H; and B for the no-obstacle V2 runs):
  n, success, grasp, place (analyze_e0 rules), runs with client robot_collision / payload_collision (MuJoCo contact with the hurdle seen
  by the client every sim step), runs ending in an obstacle stop, HOLD chunks (sum), runs with HOLD, trailing HOLD chunks per run
  (consecutive HOLD chunks that end at the last planning row), min hurdle distance per set over all control rows (mm; Af fingers, Ae palm,
  B apple, C other robot geoms): min over runs and median of per-run minima, TO solve ms per chunk (server summary to.solve_ms,
  first chunk of each run excluded, pooled over runs), client round trip ms per chunk (first chunk excluded, pooled), share > 533.3 ms.
closing / attach / held rows (V arms): see analyze_t43w.py "focus"; largest_radius_mm = max of grasp.attached.held.spheres radii (the record's
  max_radius_mm is a cap, 35.731 mm in every attach, and is not used).
pairs: by (episode, seed) on the hurdle runs; 2x2 counts of success / grasp / place and the exact two-sided McNemar p
  (analyze_t39.mcnemar_exact) for V2 vs V0 and V2 vs V1 (and V1 vs V0, E0 vs V0 for reference)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import fcntl, json, pathlib, sys, datetime, math
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
AN = pathlib.Path(__import__("os").environ.get("T43W_AN", _ag3s_workspace_path('outputs/verify/T43/W/analysis')))
PARTIAL = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43W.verify.partial.json'))
LOCK = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/W/p/.partial.lock'))
PERIOD = 1000.0 * 8 / 15
SETS = ("Af", "Ae", "B", "C")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


_MJ = {}


def body_of(gname):
    """'geom#N' -> body name of geom N in the snapshot obstacle scene (same model file the client loads); other names unchanged."""
    if not gname.startswith("geom#"):
        return gname
    if "m" not in _MJ:
        import os; os.environ.setdefault("MUJOCO_GL", "osmesa"); import mujoco
        _MJ["mj"] = mujoco
        _MJ["m"] = mujoco.MjModel.from_xml_path("${AG3S_PROJECT_ROOT}-t43w/robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml")
    mj, m = _MJ["mj"], _MJ["m"]
    return f"{gname}({mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[int(gname[5:])]))})"


def st(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    a = np.asarray(v, float)
    return {"n": int(len(a)), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max()), "min": float(a.min())}


def per_run(key, r, hold):
    h = r.get("hurdle") or {}
    cs = h.get("client_obstacle_safety") or {}
    allp = (h.get("phases") or {}).get("all") or {}
    rest = r["chunks"][1:]
    rt = [c["client_round_trip_ms"] for c in rest]
    solve = [c["solve_ms"] for c in (r.get("server_chunks") or [])[1:]]
    hv = hold.get(f"T43/W/{__import__('os').environ.get('T43W_RUNS', 'runs')}/{key.split('/')[0]}/{key.split('/')[1]}/{key.split('/')[2]}") or {}
    out = {"arm": r["arm"], "stage": r["stage"], "episode": r["episode"], "seed": r["key"],
           "success": r["success"], "grasp": r["grasp"], "place": r["place"], "grasp_t": r["grasp_t"], "place_t": r["place_t"],
           "n_control": r["n_control"], "check_mode": (r.get("check") or {}).get("mode"),
           "client_robot_collision": cs.get("robot_collision"), "client_payload_collision": cs.get("payload_collision"),
           "client_contact_steps": cs.get("contact_steps"), "client_contact_pairs": cs.get("contact_pairs"),
           "obstacle_stop": bool(h.get("client_obstacle_stop_lines")), "obstacle_stop_lines": h.get("client_obstacle_stop_lines"),
           "lift_t": h.get("lift_t"),
           "hurdle_min_mm": {k: (None if not allp.get(k) else 1e3 * allp[k]["min_m"]) for k in SETS} if allp else None,
           "hurdle_min_t": {k: (None if not allp.get(k) else allp[k]["t"]) for k in SETS} if allp else None,
           "hurdle_min_pair": {k: (None if not allp.get(k) else allp[k].get("pair")) for k in SETS} if allp else None,
           "hurdle_min_mm_by_phase": {ph: {k: (None if not (v or {}).get(k) else 1e3 * v[k]["min_m"]) for k in SETS}
                                      for ph, v in (h.get("phases") or {}).items() if ph != "all"} if h else None,
           "contact_rows": h.get("contact_rows"),
           "physics_replay_contact_events": [{k: e[k] for k in ("t_first", "t_last", "robot_substeps", "payload_substeps", "pairs", "in_hold_first_row")}
                                             for e in hv.get("contact_events", [])] if hv else None,
           "physics_replay_qpos_mismatch_rows": hv.get("qpos_mismatch_rows") if hv else None,
           "n_hold_chunks": r["n_hold_chunks"], "hold_kind_counts": r["hold_kind_counts"], "hold_t": r["hold_t"],
           "trailing_hold_chunks": r["trailing_hold_chunks"], "longest_hold_run_chunks": r["longest_hold_run_chunks"],
           "round_trip_ms": st(rt), "n_round_trip_gt_533": int(sum(1 for x in rt if x is not None and x > PERIOD)), "n_chunks_excl_first": len(rest),
           "to_solve_ms": st(solve), "server_ms": {k: (r["time_ms_excl_first"].get("server_" + k) or {}).get("median") for k in ("infer", "ag3s", "trajopt", "total")},
           "wall_s": r.get("wall_s")}
    ser = h.get("series") or {}
    afm = ser.get("apple_finger_mid_m")
    if afm is None:  # B runs: no hurdle series; apple to finger-midpoint from the physics replay is not run for B
        out["apple_finger_mid_min_mm"] = None
    else:
        i = int(np.argmin(afm)); out["apple_finger_mid_min_mm"] = 1e3 * float(afm[i]); out["apple_finger_mid_min_t"] = int(ser["t"][i])
    end_carry = r["place_t"] if r["place_t"] is not None else (h.get("apple_left_hand_t") if h.get("apple_left_hand_t") is not None else (ser.get("t") or [None])[-1])
    lt = h.get("lift_t")
    ph = lambda t: None if t is None else ("before_lift" if (lt is None or t < lt) else ("carry" if t <= end_carry else "after"))
    if out["physics_replay_contact_events"] is not None:
        for e in out["physics_replay_contact_events"]:
            e["pairs"] = [[body_of(x) for x in pr] for pr in e["pairs"]]
            e["phase"] = ph(e["t_first"]); e["kind"] = ("robot" if e["robot_substeps"] else "") + ("+payload" if e["payload_substeps"] else "")
    if r.get("server_chunks") is not None:
        out["grasp_state_counts"] = {}
        for c in r["server_chunks"]:
            out["grasp_state_counts"][str(c["grasp_state"])] = out["grasp_state_counts"].get(str(c["grasp_state"]), 0) + 1
        clo = r.get("closing") or {}
        cr = [x for x in (clo.get("carve_ball_radius_mm") or []) if x is not None]
        out["closing"] = {"n_chunks": clo.get("n_chunks"), "n_outward_gt_2mm": clo.get("n_outward_gt_2mm"), "outward_mm_max": clo.get("outward_mm_max"),
                          "shift_mm_median": clo.get("shift_mm_median"), "carve_ball_radius_mm": st(cr)}
        ah = r.get("attach_held") or []
        out["attach_held"] = [{"t": h["t"], "mode": h["mode"], "largest_radius_mm": max(h["radii_mm"]) if h["radii_mm"] else None, "n_spheres": h["n_spheres"],
                               "radii_mm": h["radii_mm"], "body_cover_radii_mm": h["body_cover_radii_mm"], "fit_radius_mm": h["fit_radius_mm"], "merged": h["merged"]} for h in ah]
        out["decision_reason_counts"] = r.get("decision_reason_counts"); out["latched_hold_counts"] = r.get("latched_hold_counts")
        out["manipulated_id_switches"] = r.get("manipulated_id_switches"); out["chunks_with_not_supported_cluster"] = r.get("chunks_with_not_supported_cluster")
        out["held_row_violation_chunks"] = r.get("held_row_violation_chunks"); out["held_row_violation_path_chunks"] = r.get("held_row_violation_path_chunks")
        dm = [c["tcp_deflection_mm"] for c in r["server_chunks"] if c["tcp_deflection_mm"] is not None]
        out["tcp_deflection_mm"] = st(dm)
        out.update({"to_status_counts": r["to_status_counts"], "time_budget_hit_count": r["time_budget_hit_count"],
                    "initial_source_counts": r["initial_source_counts"], "n_deflected_chunks": r["n_deflected_chunks"],
                    "deflected_binding_counts": r["deflected_binding_counts"], "all_binding_counts": r["all_binding_counts"],
                    "attached_chunks": r["attached_chunks"], "held_body_cover_chunks": r["held_body_cover_chunks"],
                    "sqp_iterations_counts": {str(k): v for k, v in sorted({}.items())}})
        it = {}
        for c in r["server_chunks"]:
            it[str(c["sqp_iterations"])] = it.get(str(c["sqp_iterations"]), 0) + 1
        out["sqp_iterations_counts"] = it
    return out


def summary(runs, arm, stage):
    rs = [v for v in runs.values() if v["arm"] == arm and v["stage"] == stage]
    if not rs:
        return None
    mins = {k: [v["hurdle_min_mm"][k] for v in rs if v.get("hurdle_min_mm") and v["hurdle_min_mm"].get(k) is not None] for k in SETS}
    return {"n": len(rs), "success": sum(v["success"] for v in rs), "grasp": sum(v["grasp"] for v in rs), "place": sum(v["place"] for v in rs),
            "runs_client_robot_collision": sum(bool(v["client_robot_collision"]) for v in rs),
            "runs_client_payload_collision": sum(bool(v["client_payload_collision"]) for v in rs),
            "runs_obstacle_stop": sum(v["obstacle_stop"] for v in rs),
            "hold_chunks": sum(v["n_hold_chunks"] for v in rs), "runs_with_hold": sum(v["n_hold_chunks"] > 0 for v in rs),
            "trailing_hold_chunks_per_run": {f'{v["episode"]}_s{v["seed"]}': v["trailing_hold_chunks"] for v in rs},
            "hurdle_min_mm_min": {k: (min(x) if x else None) for k, x in mins.items()},
            "hurdle_min_mm_median_of_runs": {k: (float(np.median(x)) if x else None) for k, x in mins.items()},
            "round_trip_ms_pooled_median": None, "n_round_trip_gt_533": sum(v["n_round_trip_gt_533"] for v in rs),
            "n_chunks_excl_first": sum(v["n_chunks_excl_first"] for v in rs)}


def pairs(runs, a, b, stage="H"):
    A = {(v["episode"], v["seed"]): v for v in runs.values() if v["arm"] == a and v["stage"] == stage}
    B = {(v["episode"], v["seed"]): v for v in runs.values() if v["arm"] == b and v["stage"] == stage}
    keys = sorted(set(A) & set(B))
    out = {"n_pairs": len(keys), "keys": [f"{e}_s{s}" for e, s in keys]}
    for f in ("success", "grasp", "place"):
        t = {"both": 0, f"{a}_only": 0, f"{b}_only": 0, "neither": 0}
        for k in keys:
            x, y = A[k][f], B[k][f]
            t["both" if x and y else f"{a}_only" if x else f"{b}_only" if y else "neither"] += 1
        t["mcnemar_exact_p"] = mcnemar_exact(t[f"{a}_only"], t[f"{b}_only"])
        out[f] = t
    return out


def numbers():
    an = json.loads((AN / "analysis.json").read_text())
    hold = json.loads((AN / "hold_w.json").read_text())["runs"] if (AN / "hold_w.json").exists() else {}
    hold = {k.replace("T43/W/runs/", "T43/W/runs/"): v for k, v in hold.items()}
    runs = {k: per_run(k, r, hold) for k, r in an["runs"].items()}
    # pooled timing per arm/stage
    summ = {}
    for arm in ("E0", "V1", "V3", "V4", "V5", "V6", "E0B", "V4B", "V5B", "V5S"):
        for stage in ("H", "B"):
            s = summary(runs, arm, stage)
            if s is None:
                continue
            rt, so = [], []
            for k, r in an["runs"].items():
                if r["arm"] == arm and r["stage"] == stage:
                    rt += [c["client_round_trip_ms"] for c in r["chunks"][1:]]
                    so += [c["solve_ms"] for c in (r.get("server_chunks") or [])[1:]]
            s["round_trip_ms_pooled"] = st(rt); s["to_solve_ms_pooled"] = st(so)
            rr = [v for v in runs.values() if v["arm"] == arm and v["stage"] == stage]
            agg = lambda key: {kk: sum((v.get(key) or {}).get(kk, 0) for v in rr) for kk in sorted({kk for v in rr for kk in (v.get(key) or {})})}
            s["deflected_binding_counts"] = agg("deflected_binding_counts"); s["all_binding_counts"] = agg("all_binding_counts")
            s["to_status_counts"] = agg("to_status_counts"); s["initial_source_counts"] = agg("initial_source_counts")
            s["grasp_state_counts"] = agg("grasp_state_counts"); s["hold_kind_counts"] = agg("hold_kind_counts")
            s["time_budget_hit_count"] = sum(v.get("time_budget_hit_count") or 0 for v in rr)
            s["n_deflected_chunks"] = sum(v.get("n_deflected_chunks") or 0 for v in rr)
            s["physics_replay_contact_events"] = sum(len(v["physics_replay_contact_events"] or []) for v in rr)
            s["runs_with_physics_replay_contact"] = sum(bool(v["physics_replay_contact_events"]) for v in rr)
            s["closing_chunks"] = sum((v.get("closing") or {}).get("n_chunks") or 0 for v in rr)
            s["closing_chunks_outward_gt_2mm"] = sum((v.get("closing") or {}).get("n_outward_gt_2mm") or 0 for v in rr)
            s["closing_outward_mm_max_per_run"] = {f'{v["episode"]}_s{v["seed"]}': (v.get("closing") or {}).get("outward_mm_max") for v in rr}
            s["closing_carve_ball_radius_mm_median_per_run"] = {f'{v["episode"]}_s{v["seed"]}': ((v.get("closing") or {}).get("carve_ball_radius_mm") or {}).get("median") for v in rr}
            s["attach_largest_held_radius_mm_per_run"] = {f'{v["episode"]}_s{v["seed"]}': [h["largest_radius_mm"] for h in v.get("attach_held") or []] for v in rr}
            s["attach_modes"] = {}
            for v in rr:
                for h in v.get("attach_held") or []:
                    s["attach_modes"][str(h["mode"])] = s["attach_modes"].get(str(h["mode"]), 0) + 1
            s["decision_reason_counts"] = agg("decision_reason_counts"); s["manipulated_id_switches"] = sum(v.get("manipulated_id_switches") or 0 for v in rr)
            s["chunks_with_not_supported_cluster"] = sum(v.get("chunks_with_not_supported_cluster") or 0 for v in rr)
            s["held_row_violation_chunks"] = sum(len(v.get("held_row_violation_chunks") or []) for v in rr)
            s["held_row_violation_path_chunks"] = sum(len(v.get("held_row_violation_path_chunks") or []) for v in rr)
            s["runs_with_held_row_violation"] = sum(bool(v.get("held_row_violation_chunks")) for v in rr)
            s["apple_finger_mid_min_mm_per_run"] = {f'{v["episode"]}_s{v["seed"]}': v.get("apple_finger_mid_min_mm") for v in rr}
            s.pop("round_trip_ms_pooled_median", None)
            summ[f"{arm}/{stage}"] = s
    pr = {f"{a}_vs_{b}": pairs(runs, a, b) for a, b in (("V3", "V4"), ("V4", "V1"), ("V3", "V1"), ("V1", "E0"), ("V5", "V4"), ("V6", "V5"), ("V6", "V3"), ("V5", "V1"), ("V6", "V4"))}
    return {"_comment": "mm / ms unless noted; definitions in outputs/verify/T43/W/p/partial_w.py and analyze_t43w.py docstrings",
            "summary_per_arm": summ, "pairs": pr, "per_run": runs}


def main():
    num = numbers()
    if len(sys.argv) > 2 and sys.argv[1] == "--final":
        pathlib.Path(sys.argv[2]).write_text(json.dumps(num, indent=1, default=float)); print("final numbers ->", sys.argv[2]); return
    LOCK.touch(exist_ok=True)
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {"step": "T43W"}
        key = __import__("os").environ.get("T43W_PARTIAL_KEY", "numbers_partial")
        cur[key] = num
        cur[key + "_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        tmp = PARTIAL.with_suffix(".json.tmp2"); tmp.write_text(json.dumps(cur, indent=1, ensure_ascii=False, default=float)); tmp.replace(PARTIAL)
        fcntl.flock(lk, fcntl.LOCK_UN)
    print(cur[key + "_updated"], {k: (v["n"], v["success"]) for k, v in num["summary_per_arm"].items()})


if __name__ == "__main__":
    main()
