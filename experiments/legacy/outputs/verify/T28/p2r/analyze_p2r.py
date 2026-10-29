"""T28 Phase 2R analysis — one condition at a time.

usage: analyze_p2r.py COND OUT_JSON RUN_DIR [RUN_DIR ...] [--server-records DIR ...]

Per run (reuses ../E0/analyze_e0.py:analyze and the ../E0/e0_detail.py finger/gripper logic):
  success / grasp_t / place_t / apple_dz_max  — analyze_e0 definitions (object_in_crate rule, +50 mm grasp)
  first close command t (applied_ctrl.gripper[0] < 0.5), min applied / measured left gripper opening
  HOLD count = control rows with executed == False; comms HOLD from the client log line
  "comms HOLD n" (safe client) + planning ipc in {timeout, stale, error}
  infer_ms = planning timing_ms.policy_infer (client round trip); server split = planning verdict.timing_ms
  per chunk (client side): max over arm joints (16D idx 0-6, 8-14) and the 8 executed rows of |actions - actions_reference| in deg
Server records (E1/E2): chunks.jsonl segments (seq resets to 1 at every new client connection) are matched to the
run whose first planning actions_reference equals chunk_0000-of-segment reference (npz) — exact array match.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, re
import numpy as np

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/E0'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import analyze_e0 as A
import p12_records as P

ARM_IDX = list(range(0, 7)) + list(range(8, 15))
EXEC_ROWS = 8


def client_log_summary(log_path):
    s = {}
    if not log_path.exists():
        return s
    txt = log_path.read_text(errors="replace")
    m = re.search(r"comms HOLD (\d+)", txt)
    s["comms_hold_from_log"] = int(m.group(1)) if m else None
    m = re.search(r"\[safe\] HOLD \(fixed\): (\{.*\})", txt)
    s["hold_summary_line"] = m.group(1) if m else None
    m = re.search(r"client_ipc_stats: (\{.*\})", txt)
    s["client_ipc_stats_line"] = m.group(1) if m else None
    m = re.search(r"exit=(\-?\d+)", txt)
    s["exit"] = int(m.group(1)) if m else None
    m = re.search(r"saving (\d+) frames -> (\S+)", txt)
    s["mp4_frames_saved"] = int(m.group(1)) if m else None
    return s


def run_one(run_dir, model_bits):
    run_dir = pathlib.Path(run_dir)
    r = A.analyze(run_dir)
    rows = [json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines() if l.strip()]
    plan = [x for x in rows if x["kind"] == "planning"]
    ctrl = [x for x in rows if x["kind"] == "control"]
    m, d, probe, obj, fing = model_bits
    fa, gl_cmd, gl_meas, ts = [], [], [], []
    for c in ctrl:
        P.set_q(m, d, c["qpos"])
        fa.append(1000 * min(P.gdist(m, d, gf, ga, 1.0) for gf in fing for ga in obj["apple"]))
        gl_cmd.append(c["applied_ctrl"]["gripper"][0]); gl_meas.append(abs(c["qpos"][27]) / 0.045); ts.append(c["t_step"])
    fa = np.array(fa); gl_cmd = np.array(gl_cmd); gl_meas = np.array(gl_meas)
    close_t = next((t for t, g in zip(ts, gl_cmd) if g < 0.5), None)
    dz = np.array([e["apple_dz_mm"] for e in r["timeline_control"]])
    tz = [e["t"] for e in r["timeline_control"]]
    # per chunk client-side refined - reference
    per_chunk = []
    for p in plan:
        a, ar = p.get("actions"), p.get("actions_reference")
        if a is None or ar is None:
            continue
        a = np.asarray(a)[:EXEC_ROWS][:, ARM_IDX]; ar = np.asarray(ar)[:EXEC_ROWS][:, ARM_IDX]
        per_chunk.append({"seq": p["seq"], "t": p["t_step"], "max_abs_refined_minus_reference_deg": float(np.degrees(np.abs(a - ar).max()))})
    hold_ctrl = sum(1 for c in ctrl if c.get("executed") is False)
    ipc_counts = {}
    for p in plan:
        ipc_counts[p.get("ipc")] = ipc_counts.get(p.get("ipc"), 0) + 1
    comms_ipc = sum(v for k, v in ipc_counts.items() if k in ("timeout", "stale", "error"))
    vt = [(p.get("verdict") or {}).get("timing_ms") for p in plan]
    vt = [x for x in vt if x]
    split = None
    if vt:
        def st(key):
            v = np.array([x[key] for x in vt[1:] if x.get(key) is not None])
            return {"n": int(v.size), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())} if v.size else None
        split = {"first_chunk": vt[0], "excl_first": {k: st(k) for k in ("infer", "trajopt", "total")}}
    log = client_log_summary(run_dir.parent / f"{run_dir.name}.client.log")
    mp4 = run_dir / "third_person_front.mp4"
    out = {k: v for k, v in r.items() if k not in ("timeline_planning", "timeline_control")}
    out.update({
        "recorded": mp4.exists(), "mp4": str(mp4) if mp4.exists() else None,
        "t_first_close_cmd": close_t,
        "finger_to_apple_mm_at_first_close": float(fa[ts.index(close_t)]) if close_t is not None else None,
        "min_finger_to_apple_surface_mm": float(fa.min()), "t_min_finger_to_apple": int(ts[int(fa.argmin())]),
        "min_grip_l_applied": float(gl_cmd.min()), "min_grip_l_measured_open_frac": float(gl_meas.min()),
        "n_close_cycles_applied_lt_0p5": int(np.sum((gl_cmd[1:] < 0.5) & (gl_cmd[:-1] >= 0.5))),
        "t_apple_dz_max": int(tz[int(dz.argmax())]),
        "hold_control_steps": hold_ctrl,
        "hold_chunks_planning_field_last": plan[-1].get("hold_chunks"),
        "ipc_counts": ipc_counts, "comms_ipc_planning_rows": comms_ipc,
        "server_timing_split_ms": split,
        "client_refined_minus_reference_per_chunk": per_chunk,
        "client_refined_minus_reference_max_deg": max((c["max_abs_refined_minus_reference_deg"] for c in per_chunk), default=None),
        "client_log": log,
        "timeline_control": {"t": ts, "apple_dz_mm": dz.tolist(), "grip_l_applied": gl_cmd.tolist(),
                             "grip_l_measured": gl_meas.tolist(), "finger_to_apple_surface_mm": fa.tolist()},
    })
    return out


def server_records(rec_dirs, runs, run_dirs):
    """Split chunks.jsonl into per-connection segments and match each to a run."""
    first_ref = {}
    for rd in run_dirs:
        rows = [json.loads(l) for l in (pathlib.Path(rd) / "frames.jsonl").read_text().splitlines() if l.strip()]
        p = [x for x in rows if x["kind"] == "planning" and x.get("actions_reference") is not None]
        if p:
            first_ref[pathlib.Path(rd).name] = np.asarray(p[0]["actions_reference"], dtype=np.float32)
    out = {}
    for rec in rec_dirs:
        rec = pathlib.Path(rec)
        L = [json.loads(l) for l in (rec / "chunks.jsonl").read_text().splitlines() if l.strip()]
        segs, cur = [], []
        for c in L:
            if c["seq"] == 1 and cur:
                segs.append(cur); cur = []
            cur.append(c)
        if cur:
            segs.append(cur)
        for si, seg in enumerate(segs):
            z = np.load(rec / f"chunk_{seg[0]['chunk_index']:04d}.npz")
            ref = z["reference"]
            match = [k for k, v in first_ref.items() if v.shape == ref.shape and np.array_equal(v, ref)]
            key = match[0] if len(match) == 1 else f"{rec.name}_segment{si}_unmatched{len(match)}"
            chunks = []
            for c in seg:
                s = c.get("sqp", {}); to = c.get("to", {})
                chunks.append({
                    "seq": c["seq"], "chunk_index": c["chunk_index"],
                    "max_abs_deviation_deg": c["max_abs_deviation_deg"],
                    "per_joint_max_abs_deviation_deg": c["per_joint_max_abs_deviation_deg"],
                    "limit_overshoot_reference": c["limit_overshoot_reference"],
                    "limit_overshoot_refined": c["limit_overshoot_refined"],
                    "sqp_returned": s.get("returned"), "sqp_initial_limit_overshoot": s.get("initial_limit_overshoot"),
                    "sqp_iterations": s.get("sqp_iterations", to.get("sqp_iterations")),
                    "qp_iterations": s.get("qp_iterations"), "solve_ms": to.get("solve_ms"), "status": to.get("status"),
                    "time_budget_hit": s.get("time_budget_hit"), "max_iterations_hit": s.get("max_iterations_hit"),
                    "n_accepted": s.get("n_accepted"), "qp_failures": s.get("qp_failures"),
                    "continuity_mode": (c.get("continuity") or {}).get("mode"),
                })
            dev = np.array([x["max_abs_deviation_deg"] for x in chunks])
            lo_ref = {k: float(max(x["limit_overshoot_reference"][k] for x in chunks)) for k in ("position", "velocity", "acceleration")}
            lo_ret = {k: float(max(x["limit_overshoot_refined"][k] for x in chunks)) for k in ("position", "velocity", "acceleration")}
            it = [x["sqp_iterations"] for x in chunks]
            ret = {}
            for x in chunks:
                ret[x["sqp_returned"]] = ret.get(x["sqp_returned"], 0) + 1
            out[key] = {
                "record_dir": str(rec), "chunk_index_range": [seg[0]["chunk_index"], seg[-1]["chunk_index"]], "n_chunks": len(seg),
                "n_matching_runs": len(match),
                "deviation_deg": {"max": float(dev.max()), "median": float(np.median(dev)), "n_gt_0": int((dev > 0).sum()),
                                  "n_gt_0p01": int((dev > 0.01).sum()), "n_gt_1": int((dev > 1).sum())},
                "limit_overshoot_reference_max": lo_ref, "limit_overshoot_refined_max": lo_ret,
                "n_chunks_reference_overshoot_gt0": {k: int(sum(1 for x in chunks if x["limit_overshoot_reference"][k] > 0)) for k in lo_ref},
                "n_chunks_refined_overshoot_gt0": {k: int(sum(1 for x in chunks if x["limit_overshoot_refined"][k] > 0)) for k in lo_ret},
                "sqp_iterations_counts": {str(v): it.count(v) for v in sorted(set(i for i in it if i is not None))},
                "sqp_returned_counts": ret,
                "n_time_budget_hit": int(sum(1 for x in chunks if x["time_budget_hit"])),
                "n_max_iterations_hit": int(sum(1 for x in chunks if x["max_iterations_hit"])),
                "solve_ms_excl_first": {"median": float(np.median([x["solve_ms"] for x in chunks[1:]])),
                                        "p95": float(np.percentile([x["solve_ms"] for x in chunks[1:]], 95)),
                                        "max": float(max(x["solve_ms"] for x in chunks[1:]))},
                "chunks": chunks,
            }
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    cond, out_path = args[0], pathlib.Path(args[1])
    rest = args[2:]
    recs = []
    if "--server-records" in rest:
        i = rest.index("--server-records"); recs = rest[i + 1:]; rest = rest[:i]
    bits = P.build()
    res = {"condition": cond, "definitions": __doc__, "analyze_e0_definitions": A.__doc__, "runs": {}}
    for rd in rest:
        res["runs"][pathlib.Path(rd).name] = run_one(rd, bits)
        r = res["runs"][pathlib.Path(rd).name]
        print(pathlib.Path(rd).name, "rec" if r["recorded"] else "old", "success", r["success"], "grasp_t", r["grasp_t"], "place_t", r["place_t"],
              "dz_max", round(r["apple_dz_max_mm"], 1), "close_t", r["t_first_close_cmd"], "min_meas", round(r["min_grip_l_measured_open_frac"], 3),
              "hold", r["hold_control_steps"], "comms", r["client_log"].get("comms_hold_from_log"), r["comms_ipc_planning_rows"],
              "dev_client_max", r["client_refined_minus_reference_max_deg"], "infer_med", r["infer_ms"]["median_excl_first"], flush=True)
    if recs:
        res["server_records"] = server_records(recs, res["runs"], rest)
        for k, v in res["server_records"].items():
            print("server", k, v["n_chunks"], "dev", v["deviation_deg"], "iters", v["sqp_iterations_counts"], "ret", v["sqp_returned_counts"],
                  "lo_ref", v["limit_overshoot_reference_max"], "lo_refined", v["limit_overshoot_refined_max"])
    res["n_invalid_geomdistance_pairs_skipped"] = P.N_INVALID[0]
    out_path.write_text(json.dumps(res, indent=1))
