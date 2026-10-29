"""T39 S2-3 — E0 vs E3b paired analysis (verifier). CPU only (shim or openpi-live python), MUJOCO_GL=osmesa.

usage: analyze_t39.py OUT_JSON --e0 DIR [DIR ...] --e3b DIR [DIR ...] [--key seed|rep] [--no-contact] [--cache DIR]

Run discovery: sub-directories <COND>_ep<EP>_s<SEED> (T39) or <COND>_ep<EP>_r<K> (T28/T37) that hold frames.jsonl.
If a directory uses check_run markers (any *.done present), only runs with NAME.done are used.
Pair key: --key seed -> (episode, seed)  [T39: seed = 10*ep + rep, same seed in both conditions]
          --key rep  -> (episode, rep K) [dry check on T28/T37 records only; those runs are NOT seed-paired]

Per run (definitions unchanged from T28-T37):
  grasp / place / success : outputs/verify/T28/E0/analyze_e0.py:analyze — grasp = apple z (planning-row object_poses)
                            >= initial + 50 mm; place = at the LAST planning row the apple centre is inside the crate,
                            crate-local |x| < 0.077, |y| < 0.137, -0.052 < z < 0.075; success = grasp and place.
                            qpos addresses apple 38 / crate 31 / left finger 27 were checked by name for all 24 episodes
                            (outputs/verify/T39/s20/address_check.json).
  contact height          : outputs/verify/T37/geom/a_runs.py:run_one + the settle.py rule — t_settle = last control row of the
                            FIRST close cycle (applied left gripper cmd < 0.5); contact_mid_z_mm = z of (midpoint of the two
                            finger-side witness points to the apple) - apple centre z, mm, at t_settle; contact_lost_t = first row
                            in the next 24 with min(d_l1, d_l2) > 1 mm.
  HOLD chunk (E3b)        : planning row whose control rows were not all executed (summarize_e3.py rule); kinds = gate.hold_kinds,
                            comms = ipc in {timeout, stale, error}.
  verdict distribution    : gate.action per planning row; verdict_reasons[].kind per planning row (counted once per kind per row).
  chunk time              : server verdict.timing_ms {infer, ag3s, trajopt, total} and client round trip timing_ms.policy_infer,
                            per chunk, the run's first chunk (JIT / first request) excluded; share of round trips > 533.3 ms
                            (chunk period 8 / 15 Hz).
  executed actions        : traj.npz executed_actions (600 x 16) when present, else control rows applied_ctrl mapped to 16D
                            (arm[0:7] -> 0-6, gripper[0] -> 7, arm[7:14] -> 8-14, gripper[1] -> 15; a_runs.py convention).
Per pair:
  2 x 2 table (E0 success x E3b success) and exact two-sided McNemar p = min(1, 2 * P[Binom(b + c, 1/2) <= min(b, c)]).
  first divergence: first control step where max_j |exec_E0 - exec_E3b| > 1e-3 (and > 0), its chunk (step // 8, seq = +1);
  at that chunk: E3b gate action / hold kinds / hold_reason, verdict max_violation_m, worst row {link, tier, obstacle,
  clearance_m}, verdict_reasons kinds, E3b refined - reference (rad, executed rows 0-7, all 16 cols), E0 predicted chunk vs
  E3b actions_reference (rad, rows 0-7; only when E0 traj.npz exists), manipulated target from the paired server chunk
  {id, state, nearest object centre to its centroid (MuJoCo base -> world at that planning row qpos), distance to apple}.
Server chunk pairing (E3b): server_constraints segments (server t_step == client planning seq; a segment starts at t_step 1)
  in <group dir>/server_*/server_constraints/run_*; a run takes the segment whose first chunk has summary
  policy_seed.applied == its seed, else (T37 records) the segment whose first refined_chunk equals the run's first planning
  actions within 1e-5. Each paired chunk is re-checked (refined_chunk vs planning actions <= 1e-5, counted)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import argparse, json, math, pathlib, re, sys, datetime
import numpy as np

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/E0'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import analyze_e0 as A  # noqa: E402
import p12_records as P  # noqa: E402

NAME = re.compile(r"^(E0|E3b)_ep(\d+)_(s|r)(\d+)$")
CHUNK_PERIOD_MS = 1000.0 * 8 / 15
CENTRES = ("apple", "banana", "orange", "pear", "crate")
DIV_TOL = 1e-3
_GEOM = {}
MARK = None  # (epoch start, epoch end) of a window to flag (another session's GPU job); chunks within +-2 s are flagged


def geom():
    if "bits" not in _GEOM:
        _GEOM["bits"] = P.build()
    return _GEOM["bits"]


def contact_fn():
    if "a_runs" not in _GEOM:
        sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T37/geom'))
        import a_runs  # noqa: E402  (builds its own model at import)
        _GEOM["a_runs"] = a_runs
    return _GEOM["a_runs"].run_one


def settle(r):
    """outputs/verify/T37/geom/settle.py rule, one run (r = a_runs.run_one output)."""
    tl = r["timeline"]
    i0 = next((i for i, e in enumerate(tl) if e["grip_cmd"] < 0.5), None)
    if i0 is None:
        return None
    j = i0
    while j + 1 < len(tl) and tl[j + 1]["grip_cmd"] < 0.5:
        j += 1
    e = tl[j]; w = tl[j + 1:j + 25]
    lost = next((x for x in w if min(x["d_l1_mm"], x["d_l2_mm"]) > 1.0), None)
    return {"t_close": r["t_close"], "t_settle": e["t"], "n_close_rows": j - i0 + 1, "open_settle": e["open_meas"],
            "d_l1_settle_mm": e["d_l1_mm"], "d_l2_settle_mm": e["d_l2_mm"],
            "contact_mid_rel_apple_mm": e["mid_rel_apple_mm"], "contact_mid_z_mm": e["mid_rel_apple_mm"][2],
            "contact_lost_t": lost["t"] if lost else None, "first_grasp_held": lost is None,
            "hold_min_open": min(x["open_meas"] for x in w) if w else None,
            "n_close_cycles": r["n_close_cycles"]}


def stat(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    a = np.asarray(v, float)
    return {"n": int(a.size), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "min": float(a.min()), "mean": float(a.mean())}


def cnt(items):
    out = {}
    for x in items:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items()))


def iso(p):
    try:
        return datetime.datetime.fromisoformat(p.read_text().splitlines()[0].strip())
    except Exception:
        return None


def discover(dirs):
    runs = []
    for d in dirs:
        d = pathlib.Path(d).resolve()
        use_done = any(d.glob("*.done"))
        for rd in sorted(p for p in d.iterdir() if p.is_dir()):
            m = NAME.match(rd.name)
            if not m or not (rd / "frames.jsonl").exists():
                continue
            if use_done and not (d / f"{rd.name}.done").exists():
                continue
            runs.append((m.group(1), int(m.group(2)), m.group(3), int(m.group(4)), rd))
    return runs


def executed_actions(rd, ctrl):
    z = rd / "traj.npz"
    if z.exists():
        return np.load(z)["executed_actions"].astype(np.float64), "traj.npz"
    a = np.zeros((len(ctrl), 16))
    for i, c in enumerate(ctrl):
        ac = c["applied_ctrl"]
        a[i, 0:7] = ac["arm"][0:7]; a[i, 8:15] = ac["arm"][7:14]; a[i, 7] = ac["gripper"][0]; a[i, 15] = ac["gripper"][1]
    return a, "frames.applied_ctrl"


def server_segments(group_dir):
    segs = []
    for rec in sorted(group_dir.glob("server_*/server_constraints/run_*")):
        cur = []
        for f in sorted(rec.glob("chunk_*.npz")):
            z = np.load(f, allow_pickle=True)
            ts = int(z["t_step"])
            if ts == 1 and cur:
                segs.append(cur); cur = []
            cur.append(f)
        if cur:
            segs.append(cur)
    return segs


def load_chunk(f):
    z = np.load(f, allow_pickle=True)
    return {"t_step": int(z["t_step"]), "refined": np.asarray(z["refined_chunk"], np.float64),
            "reference": np.asarray(z["reference_chunk"], np.float64), "summary": json.loads(str(z["summary_json"]))}


def pair_segment(segs, seed, key_kind, first_actions, used):
    for i, sg in enumerate(segs):
        if i in used:
            continue
        c0 = load_chunk(sg[0])
        if key_kind == "s" and (c0["summary"].get("policy_seed") or {}).get("applied") == seed:
            if first_actions is None or (c0["refined"].shape == first_actions.shape and np.abs(c0["refined"] - first_actions).max() <= 1e-5):
                return i
    for i, sg in enumerate(segs):
        if i in used:
            continue
        c0 = load_chunk(sg[0])
        if first_actions is not None and c0["refined"].shape == first_actions.shape and np.abs(c0["refined"] - first_actions).max() <= 1e-5:
            return i
    return None


def run_metrics(cond, ep, kk, k, rd, segs, used, do_contact):
    base = A.analyze(rd)
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    plan = [x for x in rows if x["kind"] == "planning"]
    ctrl = [x for x in rows if x["kind"] == "control"]
    man = json.loads((rd / "manifest.json").read_text())
    exe, exe_src = executed_actions(rd, ctrl)
    out = {"cond": cond, "episode": ep, "key_kind": kk, "key": k, "run_dir": str(rd),
           "policy_seed_manifest": man.get("policy_seed"), "warmup_steps_manifest": (man.get("timing") or {}).get("warmup_steps"),
           "success": base["success"], "grasp": base["grasp"], "place": base["place"], "grasp_t": base["grasp_t"],
           "place_t": base["place_t"], "apple_dz_max_mm": base["apple_dz_max_mm"], "apple_dz_final_mm": base["apple_dz_final_mm"],
           "apple_final_in_crate_frame": base["apple_final_in_crate_frame"],
           "n_planning": len(plan), "n_control": len(ctrl), "executed_source": exe_src}
    st, en = iso(rd.parent / f"{rd.name}.start"), iso(rd.parent / f"{rd.name}.end")
    out["wall_s"] = (en - st).total_seconds() if st and en else None
    chunks = []
    w0, m0 = man.get("recorded_at_wall"), man.get("recorded_at_monotonic")
    for p in plan:
        cs = [c for c in ctrl if c["chunk_seq"] == p["seq"]]
        # wall clock of the chunk's first control row: manifest wall + (row monotonic 'now' - manifest monotonic)
        wall = (w0 + (cs[0]["now"] - m0)) if (cs and cs[0].get("now") is not None and w0 is not None and m0 is not None) else None
        g = p.get("gate") or {}
        v = p.get("verdict") or {}
        mv = p.get("max_violation_pair") or {}
        e = {"seq": p["seq"], "t": p["t_step"], "ipc": p.get("ipc"), "safe": p.get("safe"),
             "n_control": len(cs), "n_executed": sum(1 for c in cs if c.get("executed", True)),
             "gate_action": g.get("action"), "hold_kinds": g.get("hold_kinds"), "hold_reason": p.get("hold_reason") or None,
             "verdict_reason_kinds": sorted({r.get("kind") for r in (p.get("verdict_reasons") or []) if isinstance(r, dict)}),
             "verdict_reason_hold_kinds": sorted({r.get("kind") for r in (p.get("verdict_reasons") or []) if isinstance(r, dict) and r.get("action") == "hold"}),
             "max_violation_m": v.get("max_violation_m"), "trajopt_status": v.get("trajopt_status"), "ag3s_status": v.get("ag3s_status"),
             "worst": {kk2: mv.get(kk2) for kk2 in ("link", "tier", "obstacle", "clearance_m", "block")} if mv else None,
             "server_ms": v.get("timing_ms"), "client_round_trip_ms": (p.get("timing_ms") or {}).get("policy_infer"),
             "wall_first_control": wall, "hold_abort": p.get("hold_abort")}
        e["in_mark_window"] = bool(wall is not None and MARK and MARK[0] - 2.0 <= wall <= MARK[1] + 2.0)
        e["hold"] = e["n_executed"] < e["n_control"]
        if e["hold"] and e["ipc"] in ("timeout", "stale", "error"):
            e["hold_kinds"] = ["comms:" + e["ipc"]]
        a, ar = p.get("actions"), p.get("actions_reference")
        if a is not None and ar is not None:
            a = np.asarray(a, np.float64); ar = np.asarray(ar, np.float64)
            e["refined_minus_reference_rad_rows0_7"] = float(np.abs(a[:8] - ar[:8]).max())
        chunks.append(e)
    # server pairing (E3b)
    out["server_segment"] = None
    if cond == "E3b" and segs is not None:
        fa = np.asarray(plan[0]["actions"], np.float64) if plan and plan[0].get("actions") is not None else None
        si = pair_segment(segs, k, kk, fa, used)
        if si is not None:
            used.add(si)
            m, d, probe, obj, fing = geom()
            files = segs[si]; bad = 0
            out["server_segment"] = {"first": str(files[0]), "last": str(files[-1]), "n": len(files)}
            by_seq = {c["seq"]: c for c in chunks}
            for f in files:
                sc = load_chunk(f); e = by_seq.get(sc["t_step"])
                if e is None:
                    continue
                p = plan[[c["seq"] for c in chunks].index(e["seq"])]
                if p.get("actions") is not None and np.abs(sc["refined"] - np.asarray(p["actions"], np.float64)).max() > 1e-5:
                    bad += 1
                s = sc["summary"]; man_ = s.get("manipulated")
                e["server_chunk"] = str(f)
                e["server_verdict_kinds"] = [r.get("kind") for r in (s.get("verdict") or {}).get("reasons", [])]
                e["esdf_ms"] = (s.get("profile_ms") or {}).get("esdf")
                if man_:
                    P.set_q(m, d, p["qpos"])
                    cw = P.base_to_world(m, d, man_["centroid"]); op = p["object_poses"]
                    cd = {b: float(1000 * np.linalg.norm(cw - np.asarray(op[b]["pos"]))) for b in CENTRES if b in op}
                    e["manipulated"] = {"id": man_.get("id"), "state": man_.get("state"), "nearest_centre": min(cd, key=cd.get),
                                        "to_apple_mm": cd.get("apple")}
                else:
                    e["manipulated"] = None
            out["server_segment"]["n_refined_mismatch_gt_1e-5"] = bad
    hold = [c for c in chunks if c["hold"]]
    out["n_hold_chunks"] = len(hold)
    out["hold_kind_counts"] = cnt(kx for c in hold for kx in (c["hold_kinds"] or []))
    out["hold_t"] = [c["t"] for c in hold]
    out["gate_action_counts"] = cnt(c["gate_action"] for c in chunks)
    out["verdict_reason_kind_counts"] = cnt(kx for c in chunks for kx in c["verdict_reason_kinds"])
    out["ipc_counts"] = cnt(c["ipc"] for c in chunks)
    rest = chunks[1:]
    out["time_ms_excl_first"] = {"client_round_trip": stat([c["client_round_trip_ms"] for c in rest])}
    for key in ("infer", "ag3s", "trajopt", "total"):
        out["time_ms_excl_first"]["server_" + key] = stat([(c["server_ms"] or {}).get(key) for c in rest])
    rt = [c["client_round_trip_ms"] for c in rest if c["client_round_trip_ms"] is not None]
    out["n_round_trip_gt_chunk_period"] = int(sum(1 for x in rt if x > CHUNK_PERIOD_MS))
    out["first_chunk_round_trip_ms"] = chunks[0]["client_round_trip_ms"] if chunks else None
    if do_contact:
        try:
            out["contact"] = settle(contact_fn()(rd))
        except Exception as ex:
            out["contact"] = {"error": repr(ex)}
    out["chunks"] = chunks
    return out, exe


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def table2x2(pairs, field):
    t = {"both": 0, "E0_only": 0, "E3b_only": 0, "neither": 0}
    for p in pairs:
        a, b = p["E0"][field], p["E3b"][field]
        t["both" if a and b else "E0_only" if a else "E3b_only" if b else "neither"] += 1
    t["n_pairs"] = len(pairs)
    t["mcnemar_exact_p"] = mcnemar_exact(t["E0_only"], t["E3b_only"])
    return t


def divergence(e0, e3, x0, x3, rd0):
    n = min(len(x0), len(x3))
    dif = np.abs(x0[:n] - x3[:n]).max(axis=1)
    out = {"n_steps_compared": int(n), "max_abs_overall": float(dif.max()) if n else None,
           "first_step_gt0": int(np.argmax(dif > 0)) if (dif > 0).any() else None,
           "first_step_gt_tol": int(np.argmax(dif > DIV_TOL)) if (dif > DIV_TOL).any() else None, "tol": DIV_TOL}
    # extra thresholds (facts only): first step over 1e-2 / 1e-1 rad and the E3b chunk record there; first E3b HOLD chunk
    out["by_threshold"] = {}
    for th in (1e-2, 1e-1):
        st = int(np.argmax(dif > th)) if (dif > th).any() else None
        e = {"first_step": st}
        if st is not None:
            sq = st // 8 + 1; c3_ = next((c for c in e3["chunks"] if c["seq"] == sq), None)
            e.update({"chunk_seq": sq, "chunk_t": (sq - 1) * 8,
                      "E3b_gate_action": (c3_ or {}).get("gate_action"), "E3b_hold_kinds": (c3_ or {}).get("hold_kinds"),
                      "E3b_max_violation_m": (c3_ or {}).get("max_violation_m"), "E3b_worst": (c3_ or {}).get("worst"),
                      "E3b_refined_minus_reference_rad_rows0_7": (c3_ or {}).get("refined_minus_reference_rad_rows0_7"),
                      "E3b_manipulated": (c3_ or {}).get("manipulated")})
        out["by_threshold"][f"{th:g}"] = e
    out["E3b_first_hold_chunk_seq"] = next((c["seq"] for c in e3["chunks"] if c["hold"]), None)
    s = out["first_step_gt_tol"]
    if s is None:
        return out
    seq = s // 8 + 1
    out["chunk_seq"] = seq; out["chunk_t"] = (seq - 1) * 8
    out["max_abs_at_step"] = float(dif[s])
    c3 = next((c for c in e3["chunks"] if c["seq"] == seq), None)
    if c3:
        out["E3b_chunk"] = {k: c3.get(k) for k in ("gate_action", "hold", "hold_kinds", "hold_reason", "max_violation_m", "worst",
                                                  "verdict_reason_kinds", "verdict_reason_hold_kinds", "trajopt_status", "ag3s_status",
                                                  "refined_minus_reference_rad_rows0_7", "manipulated", "server_verdict_kinds", "ipc")}
    z0 = rd0 / "traj.npz"
    p3 = [json.loads(l) for l in (pathlib.Path(e3["run_dir"]) / "frames.jsonl").read_text().splitlines() if l.strip()]
    p3 = {r["seq"]: r for r in p3 if r["kind"] == "planning"}
    if z0.exists() and seq in p3 and p3[seq].get("actions_reference") is not None:
        pc = np.load(z0)["predicted_chunks"]
        ref = np.asarray(p3[seq]["actions_reference"], np.float64)
        if seq - 1 < len(pc) and pc[seq - 1].shape == ref.shape:
            out["E0_predicted_minus_E3b_reference_rad_rows0_7"] = float(np.abs(pc[seq - 1][:8] - ref[:8]).max())
        if 1 in p3 and p3[1].get("actions_reference") is not None and pc[0].shape == np.asarray(p3[1]["actions_reference"]).shape:
            out["chunk1_E0_predicted_minus_E3b_reference_max_abs"] = float(np.abs(pc[0] - np.asarray(p3[1]["actions_reference"], np.float64)).max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--e0", nargs="+", required=True); ap.add_argument("--e3b", nargs="+", required=True)
    ap.add_argument("--key", choices=("seed", "rep"), default="seed"); ap.add_argument("--no-contact", action="store_true")
    ap.add_argument("--mark-window", nargs=2, type=float, default=None, help="epoch start end; chunks whose first control row falls inside are flagged")
    a = ap.parse_args()
    global MARK
    MARK = tuple(a.mark_window) if a.mark_window else None
    res = {"definitions": __doc__, "args": vars(a), "runs": {}}
    exe = {}
    for cond, dirs in (("E0", a.e0), ("E3b", a.e3b)):
        segs_by_dir = {}; used_by_dir = {}
        for c, ep, kk, k, rd in discover(dirs):
            if c != cond:
                continue
            g = rd.parent
            if cond == "E3b" and g not in segs_by_dir:
                segs_by_dir[g] = server_segments(g); used_by_dir[g] = set()
            r, x = run_metrics(c, ep, kk, k, rd, segs_by_dir.get(g), used_by_dir.get(g), not a.no_contact)
            name = f"{g.name}/{rd.name}"
            res["runs"][name] = r; exe[name] = x
            print(name, "S" if r["success"] else "-", "grasp_t", r["grasp_t"], "place_t", r["place_t"], "dz", round(r["apple_dz_max_mm"], 1),
                  "hold", r["n_hold_chunks"], "seg", None if r["server_segment"] is None else r["server_segment"]["n"],
                  "contact_z", (r.get("contact") or {}).get("contact_mid_z_mm"), flush=True)
    # pairs
    idx = {}
    for name, r in res["runs"].items():
        idx.setdefault((r["episode"], r["key"]), {}).setdefault(r["cond"], []).append(name)
    pairs = []
    for (ep, k), d in sorted(idx.items()):
        if "E0" in d and "E3b" in d:
            n0, n3 = d["E0"][-1], d["E3b"][-1]
            r0, r3 = res["runs"][n0], res["runs"][n3]
            pairs.append({"episode": ep, "key": k, "E0_run": n0, "E3b_run": n3,
                          "E0": {f: r0[f] for f in ("success", "grasp", "place")}, "E3b": {f: r3[f] for f in ("success", "grasp", "place")},
                          "discordant_success": r0["success"] != r3["success"],
                          "divergence": divergence(r0, r3, exe[n0], exe[n3], pathlib.Path(r0["run_dir"]))})
    res["pairs"] = pairs
    res["paired_2x2"] = {f: table2x2(pairs, f) for f in ("success", "grasp", "place")}
    res["discordant"] = {"E3b_fail_E0_success": [p for p in pairs if p["E0"]["success"] and not p["E3b"]["success"]],
                         "E0_fail_E3b_success": [p for p in pairs if p["E3b"]["success"] and not p["E0"]["success"]]}
    # condition summaries
    summ = {}
    for cond in ("E0", "E3b"):
        rs = [r for r in res["runs"].values() if r["cond"] == cond]
        if not rs:
            continue
        allc = [c for r in rs for c in r["chunks"][1:]]
        rt = [c["client_round_trip_ms"] for c in allc if c["client_round_trip_ms"] is not None]
        summ[cond] = {
            "n_runs": len(rs), "n_success": sum(r["success"] for r in rs), "n_grasp": sum(r["grasp"] for r in rs),
            "n_place": sum(r["place"] for r in rs),
            "contact_mid_z_mm": [((r.get("contact") or {}).get("contact_mid_z_mm")) for r in rs],
            "contact_mid_z_mm_stat": stat([((r.get("contact") or {}).get("contact_mid_z_mm")) for r in rs]),
            "n_first_grasp_held": sum(1 for r in rs if (r.get("contact") or {}).get("first_grasp_held")),
            "n_hold_chunks_total": sum(r["n_hold_chunks"] for r in rs), "n_runs_with_hold": sum(1 for r in rs if r["n_hold_chunks"]),
            "hold_chunks_per_run": stat([r["n_hold_chunks"] for r in rs]),
            "hold_kind_counts_total": cnt(kx for r in rs for c in r["chunks"] if c["hold"] for kx in (c["hold_kinds"] or [])),
            "gate_action_counts_total": cnt(c["gate_action"] for r in rs for c in r["chunks"]),
            "verdict_reason_kind_counts_total": cnt(kx for r in rs for c in r["chunks"] for kx in c["verdict_reason_kinds"]),
            "ipc_counts_total": cnt(c["ipc"] for r in rs for c in r["chunks"]),
            "time_ms_pooled_excl_first": dict({"client_round_trip": stat(rt)}, **{
                "server_" + key: stat([(c["server_ms"] or {}).get(key) for c in allc]) for key in ("infer", "ag3s", "trajopt", "total")}),
            "esdf_ms_pooled_excl_first": stat([c.get("esdf_ms") for c in allc]),
            "n_round_trip_gt_chunk_period": int(sum(1 for x in rt if x > CHUNK_PERIOD_MS)), "n_round_trip": len(rt),
            "mark_window": MARK, "n_chunks_in_mark_window": sum(1 for c in allc if c["in_mark_window"]),
            "runs_with_chunks_in_mark_window": sorted({r["run_dir"].split("/")[-1] for r in rs for c in r["chunks"] if c["in_mark_window"]}),
            "time_ms_pooled_excl_first_and_mark_window": dict({"client_round_trip": stat([c["client_round_trip_ms"] for c in allc if not c["in_mark_window"]])}, **{
                "server_" + key: stat([(c["server_ms"] or {}).get(key) for c in allc if not c["in_mark_window"]]) for key in ("infer", "ag3s", "trajopt", "total")}),
            "n_hold_abort_rows": sum(1 for r in rs for c in r["chunks"] if c.get("hold_abort")),
            "first_chunk_round_trip_ms": stat([r["first_chunk_round_trip_ms"] for r in rs]),
            "wall_s": stat([r["wall_s"] for r in rs]),
            "n_server_segment_paired": sum(1 for r in rs if r["server_segment"]),
            "n_server_refined_mismatch": sum((r["server_segment"] or {}).get("n_refined_mismatch_gt_1e-5", 0) for r in rs),
        }
    res["condition_summary"] = summ
    res["matrix"] = [{"episode": p["episode"], "key": p["key"], "E0": p["E0"], "E3b": p["E3b"]} for p in pairs]
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"condition_summary": {c: {k: v for k, v in s.items() if k in ("n_runs", "n_success", "n_grasp", "n_place",
                                                                                         "n_hold_chunks_total", "gate_action_counts_total")}
                                            for c, s in summ.items()},
                      "paired_2x2": res["paired_2x2"]}, indent=1))
    for p in pairs:
        dv = p["divergence"]
        print("pair", p["episode"], p["key"], "E0", int(p["E0"]["success"]), "E3b", int(p["E3b"]["success"]),
              "first_step>0", dv["first_step_gt0"], "first_step>1e-3", dv["first_step_gt_tol"], "chunk", dv.get("chunk_seq"),
              "E3b gate", (dv.get("E3b_chunk") or {}).get("gate_action"), "manip", ((dv.get("E3b_chunk") or {}).get("manipulated") or {}).get("nearest_centre"))


if __name__ == "__main__":
    main()
