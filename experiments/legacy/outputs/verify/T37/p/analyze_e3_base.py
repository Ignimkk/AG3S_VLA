"""T37 copy of T34/p/analyze_e3_base.py (paths -> T37; T37 adds to_metrics continuity_active / continuity_reason / continuity_off_by and exec_continuity {source, off_by, mode, used, n_exec, previous_chunk}). T34 docstring: T34 copy of T33/p/analyze_e3_base.py (paths -> T34). T33 docstring: T33 E3a/E3b analysis (copied from T30/p/analyze_e3_base.py; paths -> T33; T33 per-chunk fields added at the end:
to_metrics {returned, returned_index, projection, sqp_iterations, min_iterations, initial_eligible, initial_limit_overshoot},
held {grasp.attached.n_spheres, held record}, esdf held_free / tsdf_truncation, manipulated.hand, manipulated.split summary,
anchor.release_split). usage: analyze_e3.py COND

Per run: ../p2r/analyze_p2r.py:run_one (success / grasp_t / place_t / dz max / first close cmd / min opening / HOLD control steps /
comms HOLD / client timing / client refined-reference per chunk) — same definitions as Phase 2R (E0/E1/E2).
Per chunk (local planning row <-> server chunk_XXXXX.npz of that port's server_constraints, paired by exact match of the npz
refined_chunk with the local planning 'actions' AND reference_chunk with 'actions_reference', max abs <= 1e-5):
  gate            : local planning 'gate' (E3a: would_hold / would_hold_kinds; E3b: action / hold_kinds) and 'safe', 'ipc'
  executed        : all control rows of that chunk_seq executed
  manipulated     : server summary manipulated {id, state, admissibility.narrowest_extent_mm, reason}; its centroid (base) -> world
                    (MuJoCo base pose from the planning row qpos) -> |. - GT apple pos| mm and the nearest object centre
                    (apple/banana/orange/pear/crate)
  destination     : server summary destination {id, source} (None = not registered)
  exclusion       : server summary exclusion {source, active, grasp_active}; target_free ball radius / centre (esdf_stats.target_free.ball)
  finger_joints   : server summary finger_joints.opening_norm.left (value the models got, from request state) vs
                    local planning qpos[27]/0.045 (measured at that planning row) and |difference|
  worst row       : local planning max_violation_pair {link, tier, obstacle, clearance_m}; point_m (base) -> world ->
                    nearest GT object by mj_geomDistance (0.1 mm probe, P1-2 helper); inside_apple = signed distance to apple < 0
  refined-ref     : max over arm joints (16D 0-6, 8-14), executed rows 0-7 of |actions - actions_reference| (deg), local
  grasp latch     : server summary grasp {state, attach, detach, note}
  gripper         : refined chunk left gripper cmd rows 0-7 (min), applied_ctrl.gripper[0] of the chunk's control rows (min),
                    measured opening qpos[27]/0.045 over the chunk's control rows (min / last)
  server time     : local planning verdict.timing_ms {infer, ag3s, trajopt, total} + server profile_ms.esdf (ESDF build inside AG3S)
Grasp phase of a chunk: 'pre_grasp' = t_step < run grasp_t (GT apple dz >= 50 mm, planning rows) or no grasp;
'lifted' = t_step >= grasp_t and apple not yet in crate at that row; 'placed' = apple in crate at that row (analyze_e0 rule).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, re
import numpy as np

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/p2r'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/E0'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import analyze_p2r as AP
import analyze_e0 as A
import p12_records as P

COND = sys.argv[1]
OUT = pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T37/{COND}_rec")
ARM = list(range(0, 7)) + list(range(8, 15))
CENTRES = ("apple", "banana", "orange", "pear", "crate")


def server_index():
    idx = []
    for rec in sorted(OUT.glob("server_*/server_constraints/run_*")):
        for f in sorted(rec.glob("chunk_*.npz")):
            z = np.load(f, allow_pickle=True)
            idx.append((f, np.asarray(z["refined_chunk"], np.float64), np.asarray(z["reference_chunk"], np.float64)))
    return idx


def main():
    bits = P.build()
    m, d, probe, obj, fing = bits
    sidx = server_index()
    print("server chunks", len(sidx), flush=True)
    runs = sorted([p for p in OUT.iterdir() if p.is_dir() and p.name.startswith(COND + "_ep")])
    if len(sys.argv) > 2:
        runs = [r for r in runs if r.name in sys.argv[2:]]
    res = {"condition": COND, "definitions": __doc__, "runs": {}}
    used = set()
    for rd in runs:
        base = AP.run_one(rd, bits)
        rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
        plan = [x for x in rows if x["kind"] == "planning"]
        ctrl = [x for x in rows if x["kind"] == "control"]
        grasp_t = base["grasp_t"]
        chunks = []
        for p in plan:
            e = {"seq": p["seq"], "t": p["t_step"], "ipc": p.get("ipc"), "safe": p.get("safe"), "gate": p.get("gate"),
                 "hold_reason": p.get("hold_reason")}
            cs = [c for c in ctrl if c["chunk_seq"] == p["seq"]]
            e["n_control"] = len(cs); e["n_executed"] = sum(1 for c in cs if c.get("executed"))
            op = p["object_poses"]
            ic, _ = A.in_crate(op["apple"]["pos"], op["crate"]["pos"], op["crate"]["quat"])
            e["phase"] = "placed" if (ic and grasp_t is not None and p["t_step"] >= grasp_t) else (
                "lifted" if (grasp_t is not None and p["t_step"] >= grasp_t) else "pre_grasp")
            e["apple_dz_mm"] = 1000 * (op["apple"]["pos"][2] - plan[0]["object_poses"]["apple"]["pos"][2])
            a, ar = p.get("actions"), p.get("actions_reference")
            if a is not None and ar is not None:
                a = np.asarray(a, np.float64); ar = np.asarray(ar, np.float64)
                e["refined_minus_reference_deg"] = float(np.degrees(np.abs(a[:8][:, ARM] - ar[:8][:, ARM]).max()))
                e["grip_l_cmd_refined_min_rows0_7"] = float(a[:8, 7].min())
                e["grip_l_cmd_reference_min_rows0_7"] = float(ar[:8, 7].min())
            e["grip_l_applied_min"] = min((c["applied_ctrl"]["gripper"][0] for c in cs if c.get("applied_ctrl")), default=None)
            e["grip_l_measured_min"] = min((abs(c["qpos"][27]) / 0.045 for c in cs), default=None)
            e["grip_l_measured_at_planning"] = abs(p["qpos"][27]) / 0.045
            vt = (p.get("verdict") or {}).get("timing_ms")
            e["server_timing_ms"] = vt
            e["client_policy_infer_ms"] = (p.get("timing_ms") or {}).get("policy_infer")
            # worst row
            mv = p.get("max_violation_pair") or {}
            e["worst"] = {k: mv.get(k) for k in ("link", "tier", "obstacle", "clearance_m", "step", "query")}
            P.set_q(m, d, p["qpos"])
            if mv.get("point_m") is not None:
                pw = P.base_to_world(m, d, mv["point_m"])
                body, cat, per = P.nearest(m, d, probe, obj, pw)
                P.set_q(m, d, p["qpos"])
                e["worst"].update({"nearest_body": body, "nearest_surface_mm": 1000 * per[body], "apple_surface_mm": 1000 * per["apple"],
                                   "inside_apple": bool(per["apple"] < 0), "per_body_surface_mm": {b: 1000 * v for b, v in per.items()},
                                   "nearest_body_excl_apple": min((b for b in per if b != "apple"), key=lambda b: per[b])})
            # pair server chunk
            match = None
            if a is not None and ar is not None:
                for i, (f, rf, rr) in enumerate(sidx):
                    if i in used or rf.shape != a.shape:
                        continue
                    if np.abs(rf - a).max() <= 1e-5 and np.abs(rr - ar).max() <= 1e-5:
                        match = i; break
            if match is not None:
                used.add(match)
                f = sidx[match][0]
                z = np.load(f, allow_pickle=True)
                s = json.loads(str(z["summary_json"]))
                e["server_chunk"] = str(f.relative_to(OUT))
                man = s.get("manipulated")
                if man:
                    cw = P.base_to_world(m, d, man["centroid"])
                    cd = {b: float(1000 * np.linalg.norm(cw - np.asarray(op[b]["pos"]))) for b in CENTRES if b in op}
                    adm = man.get("admissibility") or {}
                    e["manipulated"] = {"id": man["id"], "state": man["state"], "age": man.get("age_frames"),
                                        "narrowest_extent_mm": adm.get("narrowest_extent_mm"), "admissible_reason": adm.get("reason"),
                                        "to_apple_mm": cd["apple"], "nearest_centre": min(cd, key=cd.get)}
                else:
                    e["manipulated"] = None
                dest = s.get("destination")
                e["destination"] = None if not dest else {"id": dest.get("id"), "source": dest.get("source"), "extents_mm": dest.get("extents_mm")}
                ex = s.get("exclusion") or {}
                e["exclusion"] = {"source": ex.get("source"), "active": ex.get("active"), "grasp_active": ex.get("grasp_active"),
                                  "mechanism": ex.get("mechanism")}
                tf = ((s.get("esdf_stats") or {}).get("target_free") or {})
                if isinstance(tf, str):
                    import ast
                    tf = ast.literal_eval(tf)
                ball = tf.get("ball") if isinstance(tf, dict) else None
                if ball:
                    bw = P.base_to_world(m, d, ball["centre_m"])
                    bd = {b: float(1000 * np.linalg.norm(bw - np.asarray(op[b]["pos"]))) for b in CENTRES if b in op}
                    e["ball"] = {"radius_mm": ball["radius_mm"], "centre_to_apple_mm": bd["apple"], "centre_nearest": min(bd, key=bd.get)}
                else:
                    e["ball"] = None
                e["invariant_violation"] = s.get("invariant_violation") is not None
                fj = s.get("finger_joints") or {}
                on = (fj.get("opening_norm") or {}).get("left")
                e["finger_joints"] = {"source": fj.get("source"), "model_left_norm": on,
                                      "measured_left_norm_planning_row": e["grip_l_measured_at_planning"],
                                      "abs_diff": None if on is None else abs(on - e["grip_l_measured_at_planning"])}
                g = s.get("grasp") or {}
                e["grasp_latch"] = {"state": g.get("state"), "attach": g.get("attach"), "detach": g.get("detach"),
                                    "rejected": g.get("rejected"), "note": g.get("note")}
                e["server_profile_ms"] = s.get("profile_ms")
                e["grounding_status"] = s.get("grounding_status")
                e["verdict_kinds_server"] = [r.get("kind") for r in (s.get("verdict") or {}).get("reasons", [])]
                to = s.get("to") or {}
                e["to"] = {k: to.get(k) for k in ("status", "sqp_iterations", "time_budget_hit", "solve_ms")}
                e["local_to_block"] = {k: (p.get("to") or {}).get(k) for k in ("continuity_active", "continuity_reason")} if isinstance(p.get("to"), dict) else None
                import ast as _ast
                def _lit(x):
                    try:
                        return _ast.literal_eval(x) if isinstance(x, str) else x
                    except Exception:
                        return x
                tm = to.get("metrics") or {}
                e["to_metrics"] = {k: _lit(tm.get(k)) for k in ("returned", "returned_index", "projection", "sqp_iterations", "min_iterations",
                                                                 "max_iterations", "initial_eligible", "initial_limit_overshoot", "limit_overshoot",
                                                                 "n_accepted", "n_candidates", "time_budget_hit", "clearance_m", "initial_violation_m",
                                                                 "continuity_active", "continuity_reason", "continuity_off_by")}
                ec = s.get("exec_continuity") or {}
                e["exec_continuity"] = {k: ec.get(k) for k in ("source", "off_by", "mode", "used", "n_exec", "previous_chunk")}
                e["to_metrics"]["max_violation_pair"] = _lit(tm.get("max_violation_pair"))
                att = g.get("attached") or {}
                e["held"] = {"hold_link": g.get("hold_link"), "n_spheres": att.get("n_spheres") if isinstance(att, dict) else None,
                             "held": _lit(att.get("held")) if isinstance(att, dict) else None}
                es = s.get("esdf_stats") or {}
                e["esdf_extra"] = {"held_free": _lit(es.get("held_free")), "tsdf_truncation": _lit(es.get("tsdf_truncation")),
                                   "n_attached_points": es.get("n_attached_points"), "n_attached_seeds_excluded": es.get("n_attached_seeds_excluded")}
                if man:
                    sp = man.get("split") or {}
                    e["manip_extra"] = {"hand": man.get("hand"), "split_n_triggered": sp.get("n_triggered"), "split_n_split": sp.get("n_split"),
                                        "split_reasons": [dd.get("reason") for dd in (sp.get("decisions") or [])],
                                        "split_release": sp.get("release"), "from_split": sp.get("manipulated_from_split"),
                                        "anchor": man.get("anchor"), "association": man.get("association"), "switched_from": man.get("switched_from")}
            chunks.append(e)
        res["runs"][rd.name] = {"summary": base, "chunks": chunks, "mp4": base.get("mp4")}
        print(rd.name, "success", base["success"], "grasp_t", grasp_t, "place_t", base["place_t"], "dz", round(base["apple_dz_max_mm"], 1),
              "paired", sum(1 for c in chunks if "server_chunk" in c), "/", len(chunks), flush=True)
    res["n_server_chunks"] = len(sidx); res["n_server_chunks_paired"] = len(used)
    res["n_invalid_geomdistance_pairs_skipped"] = P.N_INVALID[0]
    (pathlib.Path(_ag3s_workspace_path('outputs/verify/T37/p')) / (f"{COND}_analysis.json" if len(sys.argv) <= 2 else f"{COND}_analysis_partial.json")).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
