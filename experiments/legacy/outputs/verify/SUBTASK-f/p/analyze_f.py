"""SUBTASK-f analysis — three sides paired by (episode, seed), CPU only:
  off  = gate off, T39 S2-2 E3b runs                       (${AG3S_PROJECT_ROOT}/outputs/verify/T39/runs/E3b)
  on_e = gate on, SUBTASK-e code (snapshot ${AG3S_PROJECT_ROOT}-se) (${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-e/runs/E3bG)
  on_f = gate on, SUBTASK-f code (snapshot ${AG3S_PROJECT_ROOT}-sf) (${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-f/runs/E3bG)
usage: analyze_f.py OUT_JSON [--off DIR] [--on-e DIR] [--on-f DIR]

Per-run base numbers: p/analyze_v2.py run_one + p/analyze_e.py extra (both copied unchanged from SUBTASK-e/p): success / grasp /
place (T28 analyze_e0 rule), HOLD chunk (planning row whose control rows were not all executed), hold kinds (client gate.hold_kinds),
reason kinds (client planning-row verdict_reasons[].kind), region of a chunk (before_first_held / held_before_placed /
placed_before_release / from_release; t_rel = first server chunk with subtask.released True), clearance (server summary
clearance_min_mm), ready pose (traj.npz measured_qpos[0]; dL = || q[s, 0:7] - q0[0:7] || rad; dEE = || p_link_left_arm_6(s) - p(0) ||
mm by MuJoCo FK of the control-row qpos), contacts (mj_forward per control row, MuJoCo contacts dist < 0 robot vs non-robot geom, by
category apple / crate / table / other_fruit / other).

Added here (definitions):
  snt_gs of a chunk        : evidence.grounding_status of the client planning-row verdict_reasons entry with kind 'subtask_no_target'
                             (None if the row has no such entry).
  new chunk (SUBTASK-f)    : 'subtask_no_target' in reason kinds and snt_gs == 'no_admissible'.
  old chunk (SUBTASK-e)    : 'subtask_no_target' in reason kinds and snt_gs == 'subtask_gated'.
  after PLACED             : t >= t_placed (first server chunk with grasp.state 'placed'); counted alongside from_release.
  trailing HOLD streak     : number of consecutive HOLD chunks ending at the last planning row (t = 75); 0 if t = 75 is not HOLD.
                             'held to the end' = trailing streak >= 1 and every chunk of the streak is HOLD with grounding
                             no_admissible (server summary grounding_status).
  contacts of new chunks   : contact steps (above) whose control row chunk_seq is a new chunk; per category n steps, min dist.
  pairs                    : 2x2 (A success x B success) + exact two-sided McNemar for (off, on_f) and (on_e, on_f); grasp, place.
  on_f vs on_e identity    : first control step where traj.npz executed_actions differ (> 0) -> chunk = step // 8 + 1; first
                             planning seq where frames 'actions' differ; first t where server refined_chunk differs; first t where
                             the decision tuple differs (analyze_e.DEC); first t where client (gate action, hold kinds) differ.
                             t_na_e = first on_e chunk that is HOLD with hold kind 'uncertified', server grounding_status
                             'no_admissible', confirmed label 'home' and placed_seen True. Rows of both sides are listed at the first
                             gate difference and at the chunk before it.
  off vs on_f exec         : first control step where executed actions differ, chunk = step // 8 + 1."""
from common.workspace import workspace_path as _ag3s_workspace_path
import argparse, json, pathlib, sys, collections
import numpy as np
P_DIR = _ag3s_workspace_path('outputs/verify/SUBTASK-f/p')
sys.path.insert(0, P_DIR)
import analyze_v2 as V  # noqa: E402
import analyze_e as E  # noqa: E402

SIDES = ("off", "on_e", "on_f")
ROWF = ("t", "region", "hold", "hold_kinds", "reason_kinds", "gate_action", "latch", "label", "placed_seen", "blocked", "released",
        "grounding_status", "snt_gs", "mode", "manip_id", "excl_active", "to_status", "clearance_min_mm")


def C(x):
    return E.C(x)


def add(cs):
    return E.add(cs)


def f_extra(r):
    rd = pathlib.Path(r["run_dir"])
    snt_gs = {}; ctrl_chunk = {}
    for l in (rd / "frames.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        x = json.loads(l)
        if x["kind"] == "planning":
            g = [((v.get("evidence") or {}).get("grounding_status")) for v in (x.get("verdict_reasons") or [])
                 if isinstance(v, dict) and v.get("kind") == "subtask_no_target"]
            snt_gs[x["seq"]] = g[0] if g else None
        else:
            ctrl_chunk[x["t_step"]] = x.get("chunk_seq")
    rows = r["_rows"]
    for x in rows:
        x["snt_gs"] = snt_gs.get(x["t"])
    tP = r["t_placed"]
    new = [x for x in rows if "subtask_no_target" in x["reason_kinds"] and x["snt_gs"] == "no_admissible"]
    old = [x for x in rows if "subtask_no_target" in x["reason_kinds"] and x["snt_gs"] == "subtask_gated"]
    newt = {x["t"] for x in new}
    cl = [x["clearance_min_mm"] for x in new if x["clearance_min_mm"] is not None]
    cst = [c for c in r["_contact_steps"] if ctrl_chunk.get(c["step"]) in newt]
    r["new"] = {"n_chunks": len(new), "t": sorted(newt), "n_hold_among": sum(x["hold"] for x in new),
                "gate_action": C(x["gate_action"] for x in new), "by_region": C(x["region"] for x in new),
                "by_latch": C(x["latch"] for x in new), "by_label": C(x["label"] for x in new),
                "by_placed_seen": C(x["placed_seen"] for x in new), "by_released": C(x["released"] for x in new),
                "by_server_grounding": C(x["grounding_status"] for x in new), "to_status": C(x["to_status"] for x in new),
                "n_before_placed": sum(1 for x in new if tP is None or x["t"] < tP),
                "clearance_min_mm": min(cl) if cl else None, "n_clearance_lt0": sum(1 for v in cl if v < 0),
                "n_control_steps": sum(1 for s, c in ctrl_chunk.items() if c in newt),
                "contacts": {cg: {"n_steps": sum(1 for c in cst if cg in c),
                                  "min_dist_mm": min([c[cg] for c in cst if cg in c], default=None)} for cg in E.CATS}}
    r["old"] = {"n_chunks": len(old), "n_hold_among": sum(x["hold"] for x in old)}
    r["snt_gs_counts"] = C(x["snt_gs"] for x in rows if "subtask_no_target" in x["reason_kinds"])
    ap = [x for x in rows if tP is not None and x["t"] >= tP]
    r["after_placed_f"] = {"n_chunks": len(ap), "n_hold": sum(x["hold"] for x in ap),
                           "hold_kinds": C(k for x in ap if x["hold"] for k in x["hold_kinds"]),
                           "hold_grounding": C(x["grounding_status"] for x in ap if x["hold"]),
                           "hold_label": C(x["label"] for x in ap if x["hold"])}
    hold = [x["hold"] for x in sorted(rows, key=lambda y: y["t"])]
    k = 0
    for h in reversed(hold):
        if not h:
            break
        k += 1
    streak = sorted(rows, key=lambda y: y["t"])[len(rows) - k:] if k else []
    r["trailing_hold"] = {"n": k, "first_t": streak[0]["t"] if streak else None,
                          "grounding": C(x["grounding_status"] for x in streak), "kinds": C(h for x in streak for h in x["hold_kinds"]),
                          "all_no_admissible": bool(streak) and all(x["grounding_status"] == "no_admissible" for x in streak)}
    return r


SKIPPED = []


def segments(group):
    """= analyze_v2.segments, but a chunk file that numpy cannot open is skipped and listed (SUBTASK-f: the servers OOM-killed at
    12:21 left one truncated chunk_*.npz each in server_8231_20261003T120639 and server_8232_20261003T121039; those segments
    belong to killed runs that were redone on later servers)."""
    segs = []
    for rec in sorted(group.glob("server_*/server_constraints/run_*")):
        cur = []
        for f in sorted(rec.glob("chunk_*.npz")):
            try:
                with np.load(f, allow_pickle=False) as z:
                    t = int(z["t_step"])
            except Exception as e:
                SKIPPED.append({"file": str(f), "error": repr(e)}); continue
            if t == 1 and cur:
                segs.append(cur); cur = []
            cur.append(f)
        if cur:
            segs.append(cur)
    by_seed = {}
    for sg in segs:
        with np.load(sg[0], allow_pickle=False) as z:
            s = json.loads(str(z["summary_json"]))
        seed = (s.get("policy_seed") or {}).get("applied")
        by_seed.setdefault(seed, []).append(sg)
    return by_seed


def _work(tk):
    side, cond, ep, seed, rd, bs = tk
    r = V.run_one(cond, ep, seed, rd, bs); r["side"] = side; E.extra(r); f_extra(r)
    return side, ep, seed, r


def plan_actions(r):
    rows = [json.loads(l) for l in (pathlib.Path(r["run_dir"]) / "frames.jsonl").read_text().splitlines() if l.strip()]
    return {x["seq"]: x.get("actions") for x in rows if x["kind"] == "planning"}


def exec_first(ra, rb):
    xa = np.load(pathlib.Path(ra["run_dir"]) / "traj.npz")["executed_actions"]; xb = np.load(pathlib.Path(rb["run_dir"]) / "traj.npz")["executed_actions"]
    dif = np.abs(xa.astype(np.float64) - xb.astype(np.float64)).max(axis=1)
    s0 = int(np.argmax(dif > 0)) if (dif > 0).any() else None
    return s0, (None if s0 is None else s0 // 8 + 1), float(dif.max())


def row(r, t):
    b = {x["t"]: x for x in r["_rows"]}
    x = b.get(t)
    return None if x is None else {f: x.get(f) for f in ROWF}


def identity(rf, re_):
    out = {}
    s0, c0, mx = exec_first(rf, re_)
    out["exec_first_step"] = s0; out["exec_first_chunk"] = c0; out["exec_max_abs"] = mx
    pf, pe = plan_actions(rf), plan_actions(re_)
    out["planning_actions_first_t"] = next((t for t in sorted(set(pf) & set(pe)) if pf[t] != pe[t]), None)
    sf, se = rf["_srv"], re_["_srv"]
    out["refined_chunk_first_t"] = next((t for t in sorted(set(sf) & set(se)) if not np.array_equal(sf[t]["refined"], se[t]["refined"])), None)
    bf = {x["t"]: x for x in rf["_rows"]}; be = {x["t"]: x for x in re_["_rows"]}
    ts = sorted(set(bf) & set(be))
    dj = lambda a, b, f: json.dumps(a[f], default=str) != json.dumps(b[f], default=str)
    tdec = next((t for t in ts if any(dj(bf[t], be[t], f) for f in E.DEC)), None)
    out["decision_first_t"] = tdec
    out["decision_fields_at_first"] = None if tdec is None else [f for f in E.DEC if dj(bf[tdec], be[tdec], f)]
    tg = next((t for t in ts if (bf[t]["gate_action"], bf[t]["hold_kinds"]) != (be[t]["gate_action"], be[t]["hold_kinds"])), None)
    out["gate_first_t"] = tg
    tr = next((t for t in ts if (bf[t]["reason_kinds"], bf[t]["snt_gs"]) != (be[t]["reason_kinds"], be[t]["snt_gs"])), None)
    out["reason_first_t"] = tr
    if tg is not None:
        out["rows_at_gate_first"] = {"on_f": row(rf, tg), "on_e": row(re_, tg)}
        out["rows_before_gate_first"] = {"on_f": row(rf, tg - 1), "on_e": row(re_, tg - 1)}
        e = be[tg]
        out["on_e_at_gate_first_is_no_admissible_hold"] = bool(e["hold"] and "uncertified" in (e["hold_kinds"] or []) and e["grounding_status"] == "no_admissible")
        out["on_e_at_gate_first_home_placed_seen"] = bool(e["label"] == "home" and e["placed_seen"] is True)
        out["on_f_at_gate_first_new_chunk"] = bool("subtask_no_target" in bf[tg]["reason_kinds"] and bf[tg]["snt_gs"] == "no_admissible")
    if c0 is not None:
        out["rows_at_exec_first_chunk"] = {"on_f": row(rf, c0), "on_e": row(re_, c0)}
    out["t_na_e"] = next((x["t"] for x in sorted(re_["_rows"], key=lambda y: y["t"]) if x["hold"] and "uncertified" in (x["hold_kinds"] or [])
                          and x["grounding_status"] == "no_admissible" and x["label"] == "home" and x["placed_seen"] is True), None)
    out["t_first_new_f"] = rf["new"]["t"][0] if rf["new"]["t"] else None
    out["t_rel_e"] = re_["t_release"]; out["t_rel_f"] = rf["t_release"]
    out["t_placed_e"] = re_["t_placed"]; out["t_placed_f"] = rf["t_placed"]
    firsts = [v for v in (c0, out["planning_actions_first_t"], out["refined_chunk_first_t"], tdec, tg, tr) if v is not None]
    out["earliest_any_t"] = min(firsts) if firsts else None
    lim = out["t_na_e"] if out["t_na_e"] is not None else 76
    out["identical_before_t_na_e"] = out["earliest_any_t"] is None or out["earliest_any_t"] >= lim
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--off", default=_ag3s_workspace_path('outputs/verify/T39/runs/E3b'))
    ap.add_argument("--on-e", default=_ag3s_workspace_path('outputs/verify/SUBTASK-e/runs/E3bG'))
    ap.add_argument("--on-f", default=_ag3s_workspace_path('outputs/verify/SUBTASK-f/runs/E3bG'))
    a = ap.parse_args()
    runs = {s: {} for s in SIDES}
    tasks = []
    for side, dd in (("off", a.off), ("on_e", a.on_e), ("on_f", a.on_f)):
        g = pathlib.Path(dd); bs = segments(g)
        for cond, ep, seed, rd in V.discover(g):
            tasks.append((side, cond, ep, seed, rd, {seed: bs.get(seed, [])}))
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(24, len(tasks))) as pool:
        for side, ep, seed, r in pool.imap_unordered(_work, tasks):
            runs[side][(ep, seed)] = r
            print(side, ep, seed, "S" if r["success"] else "-", "rel", r["t_release"], "hold", r["n_hold_chunks"], "new", r["new"]["n_chunks"],
                  "hold_from_rel", r["regions"]["from_release"]["n_hold"], "trail", r["trailing_hold"]["n"], "dL_last", round(r["pose"]["dL_rad_last"], 3), flush=True)
    keys = sorted(set(runs["off"]) & set(runs["on_e"]) & set(runs["on_f"]))
    res = {"definitions": __doc__, "args": vars(a), "n_runs": {s: len(v) for s, v in runs.items()}, "n_triples": len(keys),
           "skipped_unreadable_chunk_files": SKIPPED,
           "n_server_segments_for_seed": {s: {f"{k[0]}_s{k[1]}": r["n_server_segments_for_seed"] for k, r in rr.items() if r["n_server_segments_for_seed"] != 1} for s, rr in runs.items()}}
    t2 = {}
    for A_, B_ in (("off", "on_f"), ("on_e", "on_f"), ("off", "on_e")):
        for f in ("success", "grasp", "place"):
            t = {"both": 0, f"{A_}_only": 0, f"{B_}_only": 0, "neither": 0}
            for k in keys:
                x, y = runs[A_][k][f], runs[B_][k][f]
                t["both" if x and y else f"{A_}_only" if x else f"{B_}_only" if y else "neither"] += 1
            t["n_pairs"] = len(keys); t["mcnemar_exact_p"] = E.mcnemar(t[f"{A_}_only"], t[f"{B_}_only"]); t2[f"{A_}_vs_{B_}:{f}"] = t
    res["paired_2x2"] = t2
    idn = {f"{k[0]}_s{k[1]}": identity(runs["on_f"][k], runs["on_e"][k]) for k in keys}
    res["identity_on_f_vs_on_e"] = idn
    iv = list(idn.values())
    res["identity_summary"] = {
        "n_pairs": len(iv), "n_fully_identical": sum(1 for x in iv if x["earliest_any_t"] is None),
        "n_pairs_with_t_na_e": sum(1 for x in iv if x["t_na_e"] is not None),
        "n_identical_before_t_na_e": sum(x["identical_before_t_na_e"] for x in iv),
        "earliest_minus_t_na_e": C(x["earliest_any_t"] - x["t_na_e"] for x in iv if x["earliest_any_t"] is not None and x["t_na_e"] is not None),
        "gate_first_minus_t_na_e": C(x["gate_first_t"] - x["t_na_e"] for x in iv if x["gate_first_t"] is not None and x["t_na_e"] is not None),
        "exec_first_chunk_minus_gate_first_t": C(x["exec_first_chunk"] - x["gate_first_t"] for x in iv if x["exec_first_chunk"] is not None and x["gate_first_t"] is not None),
        "n_on_e_at_gate_first_no_admissible_hold": sum(1 for x in iv if x.get("on_e_at_gate_first_is_no_admissible_hold")),
        "n_on_e_at_gate_first_home_placed_seen": sum(1 for x in iv if x.get("on_e_at_gate_first_home_placed_seen")),
        "n_on_f_at_gate_first_new_chunk": sum(1 for x in iv if x.get("on_f_at_gate_first_new_chunk")),
        "n_pairs_with_gate_diff": sum(1 for x in iv if x["gate_first_t"] is not None),
        "decision_fields_at_first": C(",".join(x["decision_fields_at_first"]) for x in iv if x["decision_fields_at_first"]),
        "pairs_without_t_na_e_but_differing": [k for k, x in idn.items() if x["t_na_e"] is None and x["earliest_any_t"] is not None],
        "pairs_with_t_na_e_but_identical": [k for k, x in idn.items() if x["t_na_e"] is not None and x["earliest_any_t"] is None]}
    ex = {}
    for k in keys:
        s0, c0, mx = exec_first(runs["off"][k], runs["on_f"][k])
        ex[f"{k[0]}_s{k[1]}"] = {"first_step": s0, "first_chunk": c0, "max_abs": mx, "t_rel_f": runs["on_f"][k]["t_release"],
                                 "t_placed_off": runs["off"][k]["t_placed"], "t_placed_on_f": runs["on_f"][k]["t_placed"]}
    res["off_vs_on_f_exec"] = {"per_pair": ex, "n_identical": sum(1 for v in ex.values() if v["first_step"] is None),
                               "first_chunk_minus_t_rel_f": C(v["first_chunk"] - v["t_rel_f"] for v in ex.values() if v["first_chunk"] is not None and v["t_rel_f"] is not None)}
    summ = {}
    for side, rr in runs.items():
        rs = [rr[k] for k in keys]
        pose_ref = [r for r in rs if r["s_ref"] is not None]
        newr = [r for r in rs if r["new"]["n_chunks"]]
        ncl = [r["new"]["clearance_min_mm"] for r in newr if r["new"]["clearance_min_mm"] is not None]
        summ[side] = {
            "n_runs": len(rs), "n_success": sum(r["success"] for r in rs), "n_grasp": sum(r["grasp"] for r in rs), "n_place": sum(r["place"] for r in rs),
            "n_placed_latch": sum(1 for r in rs if r["placed"]), "n_runs_released": sum(1 for r in rs if r["t_release"] is not None),
            "n_hold_chunks_total": sum(r["n_hold_chunks"] for r in rs), "n_runs_with_hold": sum(1 for r in rs if r["n_hold_chunks"]),
            "hold_kind_counts": add(r["hold_kind_counts"] for r in rs), "gate_action_counts": add(r["gate_action_counts"] for r in rs),
            "reason_kind_counts": add(r["reason_kind_counts"] for r in rs),
            "regions": {g: {"n_chunks": sum(r["regions"][g]["n_chunks"] for r in rs), "n_hold": sum(r["regions"][g]["n_hold"] for r in rs),
                            "n_runs_with_hold": sum(1 for r in rs if r["regions"][g]["n_hold"]),
                            "hold_kinds": add(r["regions"][g]["hold_kinds"] for r in rs), "reason_kinds": add(r["regions"][g]["reason_kinds"] for r in rs),
                            "grounding_status": add(r["regions"][g]["grounding_status"] for r in rs), "hold_grounding": add(r["regions"][g]["hold_grounding"] for r in rs),
                            "gate_action": add(r["regions"][g]["gate_action"] for r in rs), "to_status": add(r["regions"][g]["to_status"] for r in rs),
                            "clearance_min_mm": min([r["regions"][g]["clearance_min_mm"] for r in rs if r["regions"][g]["clearance_min_mm"] is not None], default=None),
                            "n_clearance_lt0": sum(r["regions"][g]["n_clearance_lt0"] for r in rs)}
                        for g in ("before_first_held", "held_before_placed", "placed_before_release", "from_release")},
            "after_placed": {"n_chunks": sum(r["after_placed_f"]["n_chunks"] for r in rs), "n_hold": sum(r["after_placed_f"]["n_hold"] for r in rs),
                             "n_runs_with_hold": sum(1 for r in rs if r["after_placed_f"]["n_hold"]),
                             "hold_kinds": add(r["after_placed_f"]["hold_kinds"] for r in rs), "hold_grounding": add(r["after_placed_f"]["hold_grounding"] for r in rs),
                             "hold_label": add(r["after_placed_f"]["hold_label"] for r in rs)},
            "snt_gs_counts": add(r["snt_gs_counts"] for r in rs),
            "new": {"n_chunks": sum(r["new"]["n_chunks"] for r in rs), "n_runs": len(newr), "n_hold_among": sum(r["new"]["n_hold_among"] for r in rs),
                    **{k: add(r["new"][k] for r in rs) for k in ("gate_action", "by_region", "by_latch", "by_label", "by_placed_seen", "by_released", "by_server_grounding", "to_status")},
                    "n_before_placed": sum(r["new"]["n_before_placed"] for r in rs),
                    "clearance_min_mm": min(ncl) if ncl else None, "n_clearance_lt0": sum(r["new"]["n_clearance_lt0"] for r in rs),
                    "n_control_steps": sum(r["new"]["n_control_steps"] for r in rs),
                    "contacts": {cg: {"n_runs": sum(1 for r in newr if r["new"]["contacts"][cg]["n_steps"]), "n_steps": sum(r["new"]["contacts"][cg]["n_steps"] for r in newr),
                                      "min_dist_mm": min([r["new"]["contacts"][cg]["min_dist_mm"] for r in newr if r["new"]["contacts"][cg]["min_dist_mm"] is not None], default=None)}
                                 for cg in E.CATS},
                    "per_run": {f'{r["episode"]}_s{r["seed"]}': {"n": r["new"]["n_chunks"], "first_t": r["new"]["t"][0], "last_t": r["new"]["t"][-1],
                                                                 "clearance_min_mm": r["new"]["clearance_min_mm"]} for r in newr}},
            "old_subtask_gated": {"n_chunks": sum(r["old"]["n_chunks"] for r in rs), "n_runs": sum(1 for r in rs if r["old"]["n_chunks"]),
                                  "n_hold_among": sum(r["old"]["n_hold_among"] for r in rs)},
            "trailing_hold": {"n_runs_hold_at_t75": sum(1 for r in rs if r["trailing_hold"]["n"]),
                              "n_runs_held_to_end_no_admissible": sum(1 for r in rs if r["trailing_hold"]["all_no_admissible"]),
                              "per_run": {f'{r["episode"]}_s{r["seed"]}': r["trailing_hold"] for r in rs if r["trailing_hold"]["n"]}},
            "clearance_min_mm_overall": min(r["clearance_min_mm"] for r in rs if r["clearance_min_mm"] is not None),
            "n_chunks_clearance_lt0": sum(r["n_chunks_clearance_lt0"] for r in rs),
            "pose": {"n_runs_with_ref": len(pose_ref), "ref_kind": C(r["s_ref_kind"] for r in pose_ref),
                     **{k: V.stat([r["pose"].get(k) for r in pose_ref]) for k in ("dL_rad_at_ref", "dL_rad_last", "dL_rad_min_from_ref", "dEE_mm_at_ref", "dEE_mm_last",
                                                                                  "dEE_mm_min_from_ref", "L_path_rad_from_ref")},
                     "dL_rad_last_all_runs": V.stat([r["pose"]["dL_rad_last"] for r in rs]), "dEE_mm_last_all_runs": V.stat([r["pose"]["dEE_mm_last"] for r in rs])},
            "contacts_from_ref": {cg: {"n_runs": sum(1 for r in pose_ref if r["contacts"]["from_ref"][cg]["n_steps"]),
                                       "n_steps": sum(r["contacts"]["from_ref"][cg]["n_steps"] for r in pose_ref),
                                       "min_dist_mm": min([r["contacts"]["from_ref"][cg]["min_dist_mm"] for r in pose_ref if r["contacts"]["from_ref"][cg]["min_dist_mm"] is not None], default=None)}
                                  for cg in E.CATS},
            "contacts_all": {cg: {"n_runs": sum(1 for r in rs if r["contacts"]["all"][cg]["n_steps"]), "n_steps": sum(r["contacts"]["all"][cg]["n_steps"] for r in rs),
                                  "min_dist_mm": min([r["contacts"]["all"][cg]["min_dist_mm"] for r in rs if r["contacts"]["all"][cg]["min_dist_mm"] is not None], default=None)}
                             for cg in E.CATS},
            "n_budget_hit": sum(r["n_budget_hit"] for r in rs), "to_status_counts": add(r["to_status_counts"] for r in rs),
            "n_refined_mismatch": sum(r["n_refined_mismatch_gt_1e-5"] for r in rs),
            "n_runs_segment_found": sum(1 for r in rs if r["n_server_chunks"] == 75),
        }
    res["summary"] = summ
    # per-run table of the runs that differ between on_e and on_f in HOLD after PLACED or in new chunks
    tab = []
    for k in keys:
        o, e, f = runs["off"][k], runs["on_e"][k], runs["on_f"][k]
        if not (e["after_placed_f"]["n_hold"] or f["after_placed_f"]["n_hold"] or f["new"]["n_chunks"]):
            continue
        tab.append({"run": f"{k[0]}_s{k[1]}", "success_off_e_f": [o["success"], e["success"], f["success"]],
                    "t_placed_e_f": [e["t_placed"], f["t_placed"]], "t_rel_e_f": [e["t_release"], f["t_release"]],
                    "hold_after_placed_e_f": [e["after_placed_f"]["n_hold"], f["after_placed_f"]["n_hold"]],
                    "hold_from_release_e_f": [e["regions"]["from_release"]["n_hold"], f["regions"]["from_release"]["n_hold"]],
                    "trailing_hold_e_f": [e["trailing_hold"]["n"], f["trailing_hold"]["n"]],
                    "new_chunks_f": f["new"]["n_chunks"], "new_first_t_f": f["new"]["t"][0] if f["new"]["t"] else None,
                    "new_clearance_min_mm_f": f["new"]["clearance_min_mm"],
                    "new_contacts_f": {cg: v["n_steps"] for cg, v in f["new"]["contacts"].items() if v["n_steps"]},
                    "dL_rad_last_off_e_f": [o["pose"]["dL_rad_last"], e["pose"]["dL_rad_last"], f["pose"]["dL_rad_last"]],
                    "dEE_mm_last_off_e_f": [o["pose"]["dEE_mm_last"], e["pose"]["dEE_mm_last"], f["pose"]["dEE_mm_last"]],
                    "gate_first_t_f_vs_e": idn[f"{k[0]}_s{k[1]}"]["gate_first_t"], "t_na_e": idn[f"{k[0]}_s{k[1]}"]["t_na_e"]})
    res["changed_runs_table"] = tab
    res["per_run"] = {}
    for side, rr in runs.items():
        for k, r in rr.items():
            res["per_run"][f"{side}/{k[0]}_s{k[1]}"] = {
                **{f: r[f] for f in ("cond", "episode", "seed", "run_dir", "success", "grasp", "place", "t_first_held", "t_placed", "t_release", "s_ref", "s_ref_kind",
                                     "n_hold_chunks", "hold_t", "hold_kind_counts", "reason_kind_counts", "snt", "regions", "pose", "contacts", "clearance_min_mm",
                                     "n_chunks_clearance_lt0", "n_server_chunks", "n_refined_mismatch_gt_1e-5", "new", "old", "snt_gs_counts", "after_placed_f", "trailing_hold")},
                "rows": r["_rows"], "series": r["_series"], "contact_steps": r["_contact_steps"]}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"paired_2x2": t2, "identity_summary": res["identity_summary"],
                      "summary": {s: {k: v for k, v in x.items() if k in ("n_runs", "n_success", "n_hold_chunks_total", "hold_kind_counts", "new", "after_placed",
                                                                         "trailing_hold", "n_runs_released")} for s, x in summ.items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
