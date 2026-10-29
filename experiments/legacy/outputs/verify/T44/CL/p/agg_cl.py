"""T44d CL aggregation (verifier, CPU only, numbers only). usage: agg_cl.py OUT_JSON
Inputs: CL/analysis/<NAME>.json (analyze_cl.py), CL/replay/<ARM>_R_<NAME>.{json,npz} (replay_contacts.py), LB2/passed.json.

Definitions
  task success      : analyze_t42 success = grasp (target z >= z0 + 50 mm) and place (target centre in the crate interior at the last planning row)
  Z4 release rule   : T43 CT z4 rule generalised to the episode's target fruit (replay state target_pos instead of apple_pos, contact class hand-target instead of apple-hand):
                      release_t = (last row with a hand-target contact sub-step) + 1 (must be before the run's end and grasp true); sep_threshold = d(target, finger midpoint at grasp_t) + 30 mm;
                      sep_kept = d > threshold on every row from the first row >= release_t with d > threshold to the end; released = release_t and sep_kept; success_rel = grasp and released and target in crate at end
  new contact classes (replay_contacts.py): hand-fruit_other, robot_other-target, robot_other-fruit_other, hand-target (hand = subtree EE_BODY + both fingers of the active arm; robot_other = every other robot geom
                      incl. the inactive hand; fruit_other = the other three fruit bodies; target = the episode's target fruit)
  pre-release       : control rows t <= release_t (release_t None: every row)
  CT success (success_ct) = success and success_rel and no pre-release contact in {hand-fruit_other, robot_other-target, robot_other-fruit_other}   (V11 CT rule, obstacle := neighbour fruit / non-hand link on the target)
  success_ct_fruit  = same but only {hand-fruit_other, robot_other-fruit_other}
  grasp (final)     : analyze_t42 grasp ; first grasp ok = grasp and no regrasp (one close before the lift) ; regrasp = >= 2 close commands before the lift (grasp_robust) ; slip = slip_B (rule B)
  robust grasp      : grasp and not slip_B and not regrasp (grasp_robust robust_grasp), the LB2 selection criterion
McNemar exact two-sided: min(b, c) binomial, p = min(1, 2 P) on (episode, seed) pairs.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, math, pathlib, collections
import numpy as np
import mujoco

CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL')); ANA = CL / "analysis"; REP = CL / "replay"
ARMS = ["C0", "C1a", "C1b", "C2"]
CLASSES4 = ["hand-fruit_other", "robot_other-target", "robot_other-fruit_other", "hand-target"]
CT_CLASSES = ["hand-fruit_other", "robot_other-target", "robot_other-fruit_other"]
CT_FRUIT = ["hand-fruit_other", "robot_other-fruit_other"]
LIM = (0.077, 0.137); ZR = (-0.052, 0.075)
PAIRS = [("C0", "C1a"), ("C0", "C1b"), ("C0", "C2"), ("C1a", "C1b"), ("C1b", "C2")]
passed = json.load(open(_ag3s_workspace_path('outputs/verify/T44/LB2/passed.json')))
EPS = passed["passed_episodes"]
PASSED = {p["episode"]: p for p in passed["passed"]}
RUNLIST = [(e, s) for e in EPS for s in PASSED[e]["seeds"]]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def qmat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


def inbox(p):
    return bool(abs(p[0]) < LIM[0] and abs(p[1]) < LIM[1] and ZR[0] < p[2] < ZR[1])


def local(z, i):
    return qmat(z["crate_quat"][i]).T @ (z["target_pos"][i] - z["crate_pos"][i])


def stat(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    a = np.asarray(xs)
    return {"n": len(xs), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "mean": float(a.mean()), "max": float(a.max())}


def release_score(z, a):
    t = z["t"]; N = len(t)
    d = np.linalg.norm(z["target_pos"] - z["finger_mid"], axis=1)
    o = {}
    ah = np.where(z["cnt_hand-target"] > 0)[0]
    rel_i = (int(ah[-1]) + 1) if (len(ah) and ah[-1] + 1 < N and a["grasp"]) else None
    o["n_hand_target_contact_rows"] = int(len(ah)); o["release_t"] = int(t[rel_i]) if rel_i is not None else None
    pe = local(z, N); o["target_in_crate_end"] = inbox(pe); o["target_in_crate_end_mm"] = (1e3 * pe).tolist()
    thr = None
    if a["grasp"] and a["grasp_t"] is not None:
        ig = min(int(np.searchsorted(t, a["grasp_t"])), N - 1); thr = float(d[ig + 1]) + 0.030
    o["sep_t"] = None; o["sep_kept"] = False
    if rel_i is not None and thr is not None:
        idx = [i for i in range(rel_i, N) if d[i + 1] > thr]
        if idx:
            k = idx[0]; o["sep_t"] = int(t[k]); o["sep_kept"] = bool(np.all(d[k + 1:] > thr))
    o["released"] = bool(o["release_t"] is not None and o["sep_kept"])
    o["success_rel"] = bool(a["grasp"] and o["released"] and o["target_in_crate_end"])
    return o


def load_run(arm, ep, seed):
    name = f"{arm}_ep{ep}_s{seed}"
    f = ANA / f"{name}.json"
    if not f.exists():
        return None
    a = json.load(open(f))
    out = {k: a[k] for k in ("name", "arm", "episode", "seed", "run_dir", "target_fruit", "used_arm", "success", "grasp", "place", "grasp_t", "robust_grasp", "regrasp", "slip_B",
                             "first_grasp_ok", "n_close_cycles", "failure_mode", "n_hold_chunks", "hold_kinds", "ipc_counts", "trailing_hold_chunks",
                             "client_hold_line", "obstacle_stop_in_log", "n_round_trip_gt_chunk_period", "wall_s", "drift_mm_max", "n_close_before_lift")}
    out["n_comms_fail"] = int(sum(v for k, v in a["ipc_counts"].items() if k in ("timeout", "stale", "error"))); out["n_ipc_unsafe"] = int(a["ipc_counts"].get("unsafe", 0))
    out["n_ipc_other"] = int(sum(v for k, v in a["ipc_counts"].items() if k not in ("ok", "unsafe", "timeout", "stale", "error")))
    out["time_ms_excl_first"] = a["time_ms_excl_first"]; out["_samples"] = a["_samples"]
    out["trace"] = a.get("trace"); out["server_segment"] = a.get("server_segment"); out["hold"] = a["hold"]
    rj = REP / f"{arm}_R_{name}.json"; rz = REP / f"{arm}_R_{name}.npz"
    out["replay"] = None
    if rj.exists() and rz.exists():
        j = json.load(open(rj)); z = np.load(rz)
        rs = release_score(z, a)
        t = z["t"]
        rel_t = rs["release_t"]
        pre = np.ones(len(t), bool) if rel_t is None else (t <= rel_t)
        cc = {}
        for c in CLASSES4:
            cnt = z["cnt_" + c]
            ev = j["contact_events"][c]
            cc[c] = {"rows": int((cnt > 0).sum()), "substeps": int(cnt.sum()), "rows_pre_release": int(((cnt > 0) & pre).sum()), "substeps_pre_release": int(cnt[pre].sum()),
                     "any": bool((cnt > 0).any()), "any_pre_release": bool(((cnt > 0) & pre).any()), "n_events": len(ev),
                     "min_dist_mm": (None if not ev else 1e3 * min(e["min_dist_m"] for e in ev)), "max_force_N": (0.0 if not ev else max(e["max_force_N"] for e in ev)),
                     "t_first": (None if not ev else min(e["t_first"] for e in ev)), "events": ev[:6],
                     "chunks": [c_["server_chunk_k"] for c_ in j["contact_chunks"][c]], "geom_pairs": sorted({(d["A_body"], d["B_body"]) for d in j["geom_detail"][c]})[:12]}
        out["replay"] = {"json": str(rj), "npz": str(rz), "qpos_mismatch_rows": j["qpos_mismatch_rows"], "n_rows": j["n_rows"], "release": rs, "contacts": cc,
                         "legacy_rows": {c: int(j["contact_rows"][c]) for c in j["contact_rows"]}}
        dd = np.linalg.norm(z["target_pos"] - z["finger_mid"], axis=1)
        out["min_finger_target_mm"] = float(1e3 * dd.min()); out["target_dz_max_mm"] = float(1e3 * (z["target_pos"][:, 2].max() - z["target_pos"][0, 2]))
        out["success_rel"] = rs["success_rel"]; out["release_t"] = rel_t
        out["pre_release_new_contact"] = any(cc[c]["any_pre_release"] for c in CT_CLASSES)
        out["pre_release_fruit_contact"] = any(cc[c]["any_pre_release"] for c in CT_FRUIT)
        out["success_ct"] = bool(a["success"] and rs["success_rel"] and not out["pre_release_new_contact"])
        out["success_ct_fruit"] = bool(a["success"] and rs["success_rel"] and not out["pre_release_fruit_contact"])
        out["any_new_contact"] = any(cc[c]["any"] for c in CT_CLASSES)
    return out


def pair_table(R, A, B, f):
    ka = {(o["episode"], o["seed"]): o for o in R if o["arm"] == A and f in o}
    kb = {(o["episode"], o["seed"]): o for o in R if o["arm"] == B and f in o}
    ks = sorted(set(ka) & set(kb))
    both = sum(ka[k][f] and kb[k][f] for k in ks); ao = sum(ka[k][f] and not kb[k][f] for k in ks); bo = sum(kb[k][f] and not ka[k][f] for k in ks)
    return {"n_pairs": len(ks), "both": both, "A_only": ao, "B_only": bo, "neither": len(ks) - both - ao - bo, "A_total": both + ao, "B_total": both + bo, "A": A, "B": B,
            "mcnemar_exact_p": mcnemar(ao, bo), "A_only_keys": [f"ep{e}_s{s}" for e, s in ks if ka[(e, s)][f] and not kb[(e, s)][f]],
            "B_only_keys": [f"ep{e}_s{s}" for e, s in ks if kb[(e, s)][f] and not ka[(e, s)][f]]}


def main():
    R = []
    missing = []
    for e, s in RUNLIST:
        for arm in ARMS:
            o = load_run(arm, e, s)
            if o is None:
                missing.append(f"{arm}_ep{e}_s{s}")
            else:
                R.append(o)
    arms = {}
    for arm in ARMS:
        L = [o for o in R if o["arm"] == arm]
        n = len(L)
        if not n:
            continue
        rp = [o for o in L if o.get("replay")]
        t1 = {"n_runs": n, "n_with_replay": len(rp),
              "task_success": sum(o["success"] for o in L), "success_rel": sum(o.get("success_rel", False) for o in rp), "success_ct": sum(o.get("success_ct", False) for o in rp),
              "success_ct_fruit": sum(o.get("success_ct_fruit", False) for o in rp)}
        t2 = {"grasp_final": sum(o["grasp"] for o in L), "first_grasp_ok": sum(o["first_grasp_ok"] for o in L), "regrasp_runs": sum(o["regrasp"] for o in L),
              "slip_runs": sum(o["slip_B"] for o in L), "robust_grasp": sum(o["robust_grasp"] for o in L), "place": sum(o["place"] for o in L),
              "failure_modes": dict(collections.Counter(o["failure_mode"] for o in L))}
        t3 = {}
        for c in CLASSES4:
            t3[c] = {"runs_any": sum(o["replay"]["contacts"][c]["any"] for o in rp), "contact_rows": sum(o["replay"]["contacts"][c]["rows"] for o in rp),
                     "contact_substeps": sum(o["replay"]["contacts"][c]["substeps"] for o in rp),
                     "runs_pre_release": sum(o["replay"]["contacts"][c]["any_pre_release"] for o in rp), "rows_pre_release": sum(o["replay"]["contacts"][c]["rows_pre_release"] for o in rp),
                     "runs_keys": [f"ep{o['episode']}_s{o['seed']}" for o in rp if o["replay"]["contacts"][c]["any"]]}
        t3["any_new_class_contact_runs"] = sum(o.get("any_new_contact", False) for o in rp)
        t4 = {"hold_chunks_total": sum(o["n_hold_chunks"] for o in L), "hold_runs": sum(o["n_hold_chunks"] > 0 for o in L), "hold_chunks_per_run_median": float(np.median([o["n_hold_chunks"] for o in L])),
              "hold_chunks_per_run_max": max(o["n_hold_chunks"] for o in L), "trailing_hold_runs": sum(o["trailing_hold_chunks"] > 0 for o in L),
              "ipc_unsafe_chunks": sum(o["n_ipc_unsafe"] for o in L), "comms_fail_runs (ipc timeout|stale|error)": sum(o["n_comms_fail"] > 0 for o in L), "comms_fail_chunks": sum(o["n_comms_fail"] for o in L),
              "ipc_other_chunks": sum(o["n_ipc_other"] for o in L), "run_timeouts (client exit != 0 / incomplete)": 0,
              "obstacle_stop_runs": sum(o["obstacle_stop_in_log"] for o in L),
              "hold_kinds": dict(sum((collections.Counter(o["hold_kinds"]) for o in L), collections.Counter())),
              "run_wall_s": stat([o["wall_s"] for o in L])}
        smp = lambda k: [x for o in L for x in o["_samples"][k]]
        t5 = {"to_ms_server_trajopt_pooled": stat(smp("server_trajopt")), "ag3s_ms_server_ag3s_pooled": stat(smp("server_ag3s")), "server_total_ms_pooled": stat(smp("server_total")),
              "server_infer_ms_pooled": stat(smp("server_infer")), "client_round_trip_ms_pooled": stat(smp("client_round_trip")),
              "to_solve_ms_chunks_pooled": stat([c["to_solve_ms"] for o in L if o.get("trace") for c in json.load(open(ANA / f"{o['name']}.json")).get("server_chunks", [])[1:]]),
              "run_medians_trajopt_ms": stat([o["time_ms_excl_first"]["server_trajopt"]["median"] for o in L if o["time_ms_excl_first"]["server_trajopt"]]),
              "run_medians_ag3s_ms": stat([o["time_ms_excl_first"]["server_ag3s"]["median"] for o in L if o["time_ms_excl_first"]["server_ag3s"]]),
              "round_trip_gt_533ms_chunks": sum(o["n_round_trip_gt_chunk_period"] for o in L), "n_chunks_excl_first": sum(len(o["_samples"]["client_round_trip"]) for o in L)}
        tr = [o["trace"] for o in L if o.get("trace")]
        keys = ("split_ran_chunks", "split_triggered_chunks", "split_fired_chunks", "split_decisions_accepted", "split_components", "manipulated_from_split_chunks", "ball_chunks",
                "n_seeds_in_ball_sum", "n_removed_sum", "n_kept_nontarget_sum", "n_ambiguous_sum", "n_unowned_removed_sum", "n_unowned_kept_sum", "chunks_with_kept_nontarget",
                "coarse_chunks", "coarse_n_shared_kept_sum", "coarse_n_removed_sum", "coarse_n_seeds_in_ball_sum", "coarse_chunks_shared_kept", "perm_chunks_with_exclusion",
                "path_target_rows_unauthorized_kept_sum", "time_budget_hit_chunks")
        t6 = {"n_runs_with_server_trace": len(tr), **{k: int(sum(x[k] for x in tr)) for k in keys},
              "runs_with_split_fired": sum(x["split_fired_chunks"] > 0 for x in tr), "path_target_rows_unauthorized_kept_present_in_runs": sum(x["path_target_rows_unauthorized_kept_present"] for x in tr),
              "perm_n_rows_excluded_max": max([x["perm_n_rows_excluded_max"] for x in tr] or [0]),
              "classify_ms_pooled_run_medians": stat([x["classify_ms"]["median"] for x in tr if x.get("classify_ms")]),
              "coarse_build_ms_run_medians": stat([x["coarse_build_ms"]["median"] for x in tr if x.get("coarse_build_ms")]),
              "to_status_counts": dict(sum((collections.Counter(x["to_status_counts"]) for x in tr), collections.Counter()))}
        arms[arm] = {"t1_success": t1, "t2_grasp": t2, "t3_contacts": t3, "t4_hold": t4, "t5_timing": t5, "t6_trace": t6}
    pairs = {}
    for A, B in PAIRS:
        pairs[f"{A}_vs_{B}"] = {f: pair_table(R, A, B, f) for f in ("success_ct", "success", "robust_grasp", "first_grasp_ok", "grasp", "pre_release_new_contact", "any_new_contact")}
    # discordant (episode, seed): any outcome flag differs among arms
    flags = ("success_ct", "success", "robust_grasp", "grasp", "first_grasp_ok", "pre_release_new_contact")
    disc = []
    for e, s in RUNLIST:
        rs = {o["arm"]: o for o in R if o["episode"] == e and o["seed"] == s}
        if len(rs) < 4:
            continue
        diff = [f for f in flags if len({bool(o.get(f)) for o in rs.values()}) > 1]
        if diff:
            disc.append({"episode": e, "seed": s, "target": PASSED[e]["target_fruit"], "used_arm": PASSED[e]["used_arm"], "nn_xy_m": PASSED[e]["nn_xy_m"], "differ_in": diff,
                         "per_arm": {a: {**{f: bool(rs[a].get(f)) for f in flags}, "n_hold_chunks": rs[a]["n_hold_chunks"], "failure_mode": rs[a]["failure_mode"], "release_t": rs[a].get("release_t"),
                                          "run_dir": rs[a]["run_dir"], "replay_json": (rs[a]["replay"] or {}).get("json"),
                                          "new_contact_rows": {c: (rs[a]["replay"]["contacts"][c]["rows"] if rs[a]["replay"] else None) for c in CLASSES4}} for a in ARMS}})
    flags2 = ("success_ct", "success", "robust_grasp", "grasp", "first_grasp_ok", "pre_release_new_contact")
    disc3 = []
    for e, s in RUNLIST:
        rs = {o["arm"]: o for o in R if o["episode"] == e and o["seed"] == s and o["arm"] != "C2"}
        if len(rs) < 3:
            continue
        diff = [f for f in flags2 if len({bool(o.get(f)) for o in rs.values()}) > 1]
        if diff:
            disc3.append({"episode": e, "seed": s, "differ_in": diff, "arms_success_ct": {a: bool(rs[a].get("success_ct")) for a in rs}, "arms_success": {a: bool(rs[a]["success"]) for a in rs},
                          "run_dirs": {a: rs[a]["run_dir"] for a in rs}, "replay_json": {a: (rs[a]["replay"] or {}).get("json") for a in rs}})
    # per-run table
    runs = {}
    for o in R:
        runs[o["name"]] = {k: o.get(k) for k in ("episode", "seed", "arm", "target_fruit", "used_arm", "success", "grasp", "place", "first_grasp_ok", "regrasp", "slip_B", "robust_grasp", "failure_mode",
                                                  "success_rel", "success_ct", "success_ct_fruit", "pre_release_new_contact", "release_t", "n_hold_chunks", "hold_kinds", "n_comms_fail", "wall_s", "min_finger_target_mm", "target_dz_max_mm", "run_dir")}
        if o.get("replay"):
            runs[o["name"]]["contacts_rows"] = {c: o["replay"]["contacts"][c]["rows"] for c in CLASSES4}
            runs[o["name"]]["contacts_rows_pre_release"] = {c: o["replay"]["contacts"][c]["rows_pre_release"] for c in CLASSES4}
            runs[o["name"]]["replay_json"] = o["replay"]["json"]
        runs[o["name"]]["trace"] = {k: (o["trace"] or {}).get(k) for k in ("split_fired_chunks", "n_removed_sum", "n_kept_nontarget_sum", "n_unowned_removed_sum", "coarse_n_shared_kept_sum")} if o.get("trace") else None
    res = {"definitions": __doc__, "n_runs_expected": 4 * len(RUNLIST), "n_runs_found": len(R), "missing": missing, "episodes": EPS,
           "episodes_info": {str(e): {k: PASSED[e][k] for k in ("target_fruit", "used_arm", "bin", "nn_xy_m", "nn_other_fruit", "seeds")} for e in EPS},
           "arms": arms, "pairs": pairs, "discordant": disc, "discordant_excluding_C2": disc3, "runs": runs}
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))
    for a, v in arms.items():
        print(a, v["t1_success"], {k: v["t2_grasp"][k] for k in ("grasp_final", "first_grasp_ok", "regrasp_runs", "slip_runs", "robust_grasp")}, "hold", v["t4_hold"]["hold_chunks_total"], v["t4_hold"]["hold_runs"], "comms_fail", v["t4_hold"]["comms_fail_chunks"])
    for p, v in pairs.items():
        print(p, {f: (x["A_total"], x["B_total"], round(x["mcnemar_exact_p"], 4)) for f, x in v.items() if f in ("success_ct", "success", "robust_grasp")})
    print("discordant", [(d["episode"], d["seed"], d["differ_in"]) for d in disc]); print("missing", missing)


if __name__ == "__main__":
    main()
