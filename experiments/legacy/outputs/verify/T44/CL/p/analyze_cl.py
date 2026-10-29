"""T44d CL per-run analysis (verifier, CPU only). usage: analyze_cl.py [RUN_DIR ...]   (default: every runs/<ARM>/R/<ARM>_ep*_s*.done)
Writes analysis/<NAME>.json (cached; recomputed when frames.jsonl is newer). Numbers only.

Per run:
  outcome   : analyze_t42.analyze (T28 analyze_e0 rule: grasp = target z >= z0 + 50 mm, place = target centre in the crate interior at the last planning row, success = grasp and place)
              grasp_robust.robust (T42-1; LB2 rule B): n_close_cycles, n_close_before_lift, regrasp (>= 2 closes before lift), slip_B, robust_grasp, failure_mode
              first_grasp_ok = grasp and not regrasp ; final grasp = grasp
  hold      : planning rows whose control rows were not all executed (HOLD chunks), gate hold_kinds, ipc counts (timeout / stale / error), trailing HOLD rows, client '[safe] HOLD' summary line
  timing    : planning-row verdict.timing_ms {infer, ag3s, trajopt, total} and timing_ms.policy_infer (client round trip); first planning row excluded (JIT)
  server    : server_constraints segment paired to the run (every refined_chunk == planning actions <= 1e-6); per chunk: to.solve_ms, T44 trace of target_exclusion (n_seeds_in_ball, n_removed,
              n_kept_nontarget, n_ambiguous, n_unowned_removed/kept, coarse n_shared_kept ...), manipulated.split (ran, n_triggered, n_split, n_components), target_permission n_rows_excluded,
              to.metrics.path_check target_rows_unauthorized_kept when present
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, glob, os
import numpy as np
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/LB2/p'))
import analyze_t42 as AN  # noqa: E402
import grasp_robust as GR  # noqa: E402

CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL')); RUNS = CL / "runs"; ANA = CL / "analysis"
ARMS = ("C0", "C1a", "C1b", "C2")
CHUNK_PERIOD_MS = 533.3


def lit(x):
    import ast
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return None
    return x


def stat(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    a = np.asarray(xs)
    return {"n": len(xs), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "mean": float(a.mean()), "max": float(a.max())}


def segments(arm_dir):
    segs = []
    for rec in sorted(arm_dir.glob("server_*/server_constraints/run_*")):
        cur = []
        for f in sorted(rec.glob("chunk_*.npz")):
            try:
                ts = int(np.load(f, allow_pickle=True)["t_step"])
            except Exception:
                continue
            if ts == 1 and cur:
                segs.append(cur); cur = []
            cur.append(f)
        if cur:
            segs.append(cur)
    return segs


def pair_seg(segs, plan, used):
    acts = [np.asarray(p["actions"], np.float64) for p in plan]
    for i, sg in enumerate(segs):
        if i in used or len(sg) != len(acts):
            continue
        ok = True
        for f, a in zip(sg, acts):
            r = np.asarray(np.load(f, allow_pickle=True)["refined_chunk"], np.float64)
            if r.shape != a.shape or np.abs(r - a).max() > 1e-6:
                ok = False; break
        if ok:
            return i
    return None


def chunk_trace(f):
    z = np.load(f, allow_pickle=True); s = json.loads(str(z["summary_json"]))
    te = s.get("target_exclusion") or {}; co = te.get("coarse") or {}; tp = s.get("target_permission") or {}
    to = s.get("to") or {}; tm = to.get("metrics") or {}
    sp = (s.get("manipulated") or {}).get("split") or {}
    dec = sp.get("decisions") or []
    pc = lit(tm.get("path_check")) or {}
    la = lit(tm.get("lookahead")) or {}
    own = te.get("owned") or {}
    e = {"seq": int(z["t_step"]), "to_solve_ms": to.get("solve_ms"), "to_status": to.get("status"), "sqp_iterations": to.get("sqp_iterations"),
         "time_budget_hit": to.get("time_budget_hit"), "ag3s_status": s.get("status"), "validity": s.get("validity"), "has_target": s.get("has_target"),
         "grasp_state": (s.get("grasp") or {}).get("state"), "attached": bool((s.get("grasp") or {}).get("attached")),
         "verdict_reasons": [(r.get("kind"), r.get("action")) for r in ((s.get("verdict") or {}).get("reasons") or [])],
         "excl_mode": te.get("mode"), "excl_method": te.get("exclusion_method"),
         "n_seeds_in_ball": te.get("n_seeds_in_ball"), "n_removed": te.get("n_removed"), "n_kept_nontarget": te.get("n_kept_nontarget"),
         "n_ambiguous": te.get("n_ambiguous"), "n_unowned_removed": te.get("n_unowned_removed"), "n_unowned_kept": te.get("n_unowned_kept"),
         "n_target_owned": te.get("n_target_owned"), "classify_ms": te.get("classify_ms"), "build_ms": te.get("build_ms"), "n_target_points": own.get("n_target_points"),
         "n_nontarget_points": own.get("n_nontarget_points"),
         "coarse_mode": co.get("mode"), "coarse_n_seeds_in_ball": co.get("n_seeds_in_ball"), "coarse_n_removed": co.get("n_removed"), "coarse_n_shared_kept": co.get("n_shared_kept"),
         "coarse_n_seeds_shared": co.get("n_seeds_shared"), "coarse_build_ms": co.get("build_ms"), "coarse_sign_z_min_m": (co.get("params") or {}).get("sign_z_min_m"),
         "split_ran": bool(sp.get("ran")), "split_n_triggered": sp.get("n_triggered"), "split_n_split": sp.get("n_split"),
         "split_n_components": int(sum(len(d.get("components") or []) for d in dec if d.get("accepted"))),
         "split_from_split": sp.get("manipulated_from_split"),
         "perm_policy": tp.get("policy"), "perm_n_rows": tp.get("n_rows"), "perm_n_rows_excluded": tp.get("n_rows_excluded"),
         "path_target_rows_unauthorized_kept": pc.get("target_rows_unauthorized_kept") if isinstance(pc, dict) else None,
         "lookahead_gate": la.get("gate") if isinstance(la, dict) else None,
         "profile_ms": s.get("profile_ms")}
    return e


def analyze_run(rd, segs, used):
    rd = pathlib.Path(rd); name = rd.name; arm = name.split("_ep")[0]
    ep = int(name.split("_ep")[1].split("_s")[0]); seed = int(name.split("_s")[-1])
    a = AN.analyze(rd)
    r = GR.robust(rd); r.pop("series", None)
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    plan = [x for x in rows if x["kind"] == "planning"]; ctrl = [x for x in rows if x["kind"] == "control"]
    man = json.loads((rd / "manifest.json").read_text())
    out = {"name": name, "arm": arm, "episode": ep, "seed": seed, "run_dir": str(rd), "target_fruit": a["target_fruit"], "used_arm": a["used_arm"],
           "success": bool(a["success"]), "grasp": bool(a["grasp"]), "place": bool(a["place"]), "grasp_t": a.get("grasp_t"), "place_t": a.get("place_t"),
           "robust_grasp": bool(r["robust_grasp"]), "robust_grasp_strict": bool(r["robust_grasp_strict"]), "n_close_cycles": r.get("n_close_cycles"),
           "n_close_before_lift": r.get("n_close_before_lift"), "regrasp": bool(r.get("regrasp")), "slip_B": bool(r.get("slip_B", r.get("slip"))),
           "slip_A": bool(r.get("slip")), "drift_mm_max": r.get("drift_mm_max"), "failure_mode": GR.failure_mode(r), "lift_t": r.get("lift_t"), "release_t_robust": r.get("release_t"),
           "n_planning": len(plan), "n_control": len(ctrl)}
    out["first_grasp_ok"] = bool(out["grasp"] and not out["regrasp"])
    out["robust_keys"] = {k: r[k] for k in r if k in ("lift_t", "release_t", "place_t", "finger_gap_mm_max", "n_rows_no_finger_within_3mm", "dz_drop_mm_max", "first_slip_t")}
    # HOLD / ipc / timing
    hold = []; ipc = {}; kinds = {}
    for p in plan:
        cs = [c for c in ctrl if c["chunk_seq"] == p["seq"]]
        ne = sum(1 for c in cs if c.get("executed", True))
        h = ne < len(cs)
        ip = p.get("ipc"); ipc[str(ip)] = ipc.get(str(ip), 0) + 1
        if h:
            ks = list((p.get("gate") or {}).get("hold_kinds") or [])
            if p.get("ipc") in ("timeout", "stale", "error"):
                ks = ["comms:" + p["ipc"]]
            hold.append({"seq": p["seq"], "t": p["t_step"], "n_exec": ne, "n_control": len(cs), "kinds": ks, "reason": p.get("hold_reason") or None})
            for k in ks:
                kinds[k] = kinds.get(k, 0) + 1
    trailing = 0
    hs = {h["seq"] for h in hold}
    for p in reversed(plan):
        if p["seq"] in hs: trailing += 1
        else: break
    out.update({"n_hold_chunks": len(hold), "hold": hold, "hold_kinds": kinds, "ipc_counts": ipc, "trailing_hold_chunks": trailing,
                "n_ipc_not_ok": sum(v for k, v in ipc.items() if k not in ("ok", "None")), "n_timeout": ipc.get("timeout", 0)})
    log = (rd.parent / f"{name}.client.log").read_text(errors="replace")
    out["client_hold_line"] = next((l.strip() for l in log.splitlines()[::-1] if "[safe] HOLD" in l), None)
    out["obstacle_stop_in_log"] = "[SAFETY] obstacle stop" in log
    rest = plan[1:]
    tm = lambda key: [((p.get("verdict") or {}).get("timing_ms") or {}).get(key) for p in rest]
    out["time_ms_excl_first"] = {"client_round_trip": stat([(p.get("timing_ms") or {}).get("policy_infer") for p in rest]),
                                 **{"server_" + k: stat(tm(k)) for k in ("infer", "ag3s", "trajopt", "total")}}
    out["_samples"] = {"client_round_trip": [(p.get("timing_ms") or {}).get("policy_infer") for p in rest], **{"server_" + k: tm(k) for k in ("infer", "ag3s", "trajopt", "total")}}
    out["n_round_trip_gt_chunk_period"] = int(sum(1 for p in rest if ((p.get("timing_ms") or {}).get("policy_infer") or 0) > CHUNK_PERIOD_MS))
    out["wall_s"] = None
    try:
        import datetime
        st = datetime.datetime.fromisoformat((rd.parent / f"{name}.start").read_text().splitlines()[0]); en = datetime.datetime.fromisoformat((rd.parent / f"{name}.end").read_text().splitlines()[0])
        out["wall_s"] = (en - st).total_seconds()
    except Exception:
        pass
    # server pairing
    si = pair_seg(segs, plan, used)
    out["server_segment"] = None
    if si is not None:
        used.add(si)
        files = segs[si]
        out["server_segment"] = {"first": str(files[0]), "last": str(files[-1]), "n": len(files)}
        ch = [chunk_trace(f) for f in files]
        by = {p["seq"]: p for p in plan}
        for c in ch:
            p = by.get(c["seq"]); c["t"] = p["t_step"] if p else None
            c["hold"] = (c["seq"] in hs)
            c["server_file"] = None
        for c, f in zip(ch, files):
            c["server_file"] = str(f)
        out["server_chunks"] = ch
        S = lambda k: [c[k] for c in ch if c.get(k) is not None]
        out["server_to_solve_ms"] = stat([c["to_solve_ms"] for c in ch[1:]])
        out["trace"] = {
            "n_chunks": len(ch),
            "split_ran_chunks": sum(1 for c in ch if c["split_ran"]), "split_triggered_chunks": sum(1 for c in ch if (c["split_n_triggered"] or 0) > 0),
            "split_fired_chunks": sum(1 for c in ch if (c["split_n_split"] or 0) > 0), "split_decisions_accepted": int(sum(c["split_n_split"] or 0 for c in ch)),
            "split_components": int(sum(c["split_n_components"] for c in ch)), "manipulated_from_split_chunks": sum(1 for c in ch if c["split_from_split"]),
            "ball_chunks": sum(1 for c in ch if c["excl_mode"] == "ball"),
            "n_seeds_in_ball_sum": int(sum(S("n_seeds_in_ball"))), "n_removed_sum": int(sum(S("n_removed"))), "n_kept_nontarget_sum": int(sum(S("n_kept_nontarget"))),
            "n_ambiguous_sum": int(sum(S("n_ambiguous"))), "n_unowned_removed_sum": int(sum(S("n_unowned_removed"))), "n_unowned_kept_sum": int(sum(S("n_unowned_kept"))),
            "chunks_with_kept_nontarget": sum(1 for c in ch if (c["n_kept_nontarget"] or 0) > 0),
            "coarse_chunks": sum(1 for c in ch if c["coarse_mode"] not in (None, "off")), "coarse_n_shared_kept_sum": int(sum(S("coarse_n_shared_kept"))),
            "coarse_n_removed_sum": int(sum(S("coarse_n_removed"))), "coarse_n_seeds_in_ball_sum": int(sum(S("coarse_n_seeds_in_ball"))),
            "coarse_chunks_shared_kept": sum(1 for c in ch if (c["coarse_n_shared_kept"] or 0) > 0),
            "perm_n_rows_excluded_max": max(S("perm_n_rows_excluded") or [0]), "perm_chunks_with_exclusion": sum(1 for c in ch if (c["perm_n_rows_excluded"] or 0) > 0),
            "path_target_rows_unauthorized_kept_sum": int(sum(S("path_target_rows_unauthorized_kept"))),
            "path_target_rows_unauthorized_kept_present": bool(S("path_target_rows_unauthorized_kept")),
            "time_budget_hit_chunks": sum(1 for c in ch if c["time_budget_hit"]), "to_status_counts": {},
            "classify_ms": stat(S("classify_ms")), "build_ms": stat(S("build_ms")), "coarse_build_ms": stat(S("coarse_build_ms")),
        }
        for c in ch:
            k = str(c["to_status"]); out["trace"]["to_status_counts"][k] = out["trace"]["to_status_counts"].get(k, 0) + 1
    return out


def main():
    ANA.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        rds = [pathlib.Path(p) for p in sys.argv[1:]]
    else:
        rds = sorted(pathlib.Path(p).with_suffix("") for p in glob.glob(str(RUNS / "*" / "R" / "*_ep*_s*.done")))
    segcache = {}; used = {}
    for rd in rds:
        arm = rd.name.split("_ep")[0]
        outf = ANA / f"{rd.name}.json"
        if outf.exists() and outf.stat().st_mtime >= (rd / "frames.jsonl").stat().st_mtime:
            print("cached", rd.name, flush=True); continue
        if arm not in segcache:
            segcache[arm] = segments(RUNS / arm); used[arm] = set()
        # pair in start order: used set avoids reuse
        o = analyze_run(rd, segcache[arm], used[arm])
        outf.write_text(json.dumps(o, indent=1, default=float))
        print(rd.name, "success", o["success"], "grasp", o["grasp"], "robust", o["robust_grasp"], "hold", o["n_hold_chunks"], "seg", bool(o["server_segment"]), flush=True)


if __name__ == "__main__":
    main()
