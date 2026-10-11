"""T43 V11 extra numbers (verifier, CPU only, numbers only). usage: V11_ARM=V11A extra_v11.py OUT_JSON
(copy of V8/p/extra_v8.py structure; base = V8, new = ARM)
 - G / P / success under the Z4 release rule and the CT rule, paired new vs base (from V11/ct_<ARM>.json), per-run TO solve median pairs, loadavg during the round, per-server facts, chain facts, verdict kinds.
 - TO timing from the raw server chunk records of BOTH arms (every chunk of the arm's server dirs; chunk with t_step == 1 of each run excluded = analyze_t39 rule):
   to.solve_ms (median / p90 / max), time_budget_hit, max_iterations_hit, sqp_iterations, qp_iterations.
 - T43 TA record fields (new arm): to.metrics.lookahead (gate open / closed, tail_reference_clearance_m, tail violation), to.metrics.deflection_rate, to.metrics.servo_accel."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, math, glob, re, collections, os
import numpy as np
ARM = os.environ.get("V11_ARM", "V11A")
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); V = T / "V11"
ct = json.load(open(V / f"ct_{ARM}.json")); agg = json.load(open(V / f"agg_{ARM}.json")); R = ct["runs"]


def mc(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def pair(f, A=ARM, B="V8"):
    ka = {(o["episode"], o["seed"]): o for o in R.values() if o["arm"] == A}; kb = {(o["episode"], o["seed"]): o for o in R.values() if o["arm"] == B}
    ks = sorted(set(ka) & set(kb)); both = sum(ka[k][f] and kb[k][f] for k in ks); ao = sum(ka[k][f] and not kb[k][f] for k in ks); bo = sum(kb[k][f] and not ka[k][f] for k in ks)
    return {"n_pairs": len(ks), "new_total": both + ao, "base_total": both + bo, "both": both, "new_only": ao, "base_only": bo, "neither": len(ks) - both - ao - bo, "mcnemar_exact_p": mc(ao, bo),
            "new_only_keys": [f"{e}_s{s}" for e, s in ks if ka[(e, s)][f] and not kb[(e, s)][f]], "base_only_keys": [f"{e}_s{s}" for e, s in ks if kb[(e, s)][f] and not ka[(e, s)][f]]}


out = {"labels": {"new": ARM, "base": "V8"}}
out["H_pairs_z4_rule"] = {f: pair(f) for f in ("grasp", "place_rel", "released", "success_rel", "success_ct")}
out["H_totals"] = {a: {f: sum(o[f] for o in R.values() if o["arm"] == a) for f in ("success", "grasp", "place", "released", "place_rel", "success_rel", "success_ct")} for a in ("V8", ARM)}
t = agg["timing"]; pr = {}
for run in sorted(set(t[ARM]["ALL"]["per_run"]) & set(t["V8"]["ALL"]["per_run"])):
    a, b = t[ARM]["ALL"]["per_run"][run], t["V8"]["ALL"]["per_run"][run]
    pr[run] = {"new_median_ms": a["to_solve_median_ms"], "base_median_ms": b["to_solve_median_ms"], "ratio": a["to_solve_median_ms"] / b["to_solve_median_ms"]}
rr = [x["ratio"] for x in pr.values()]; dd = [x["new_median_ms"] - x["base_median_ms"] for x in pr.values()]
out["per_run_to_solve_median"] = {"runs": pr, "ratio_median": float(np.median(rr)), "ratio_min": min(rr), "ratio_max": max(rr), "diff_ms_median": float(np.median(dd)),
                                  "n_runs_new_higher": int(sum(d > 0 for d in dd)), "n_runs": len(dd)}
# load during the round
L = [l.split() for l in open(V / "raw/load_samples.txt") if len(l.split()) >= 6 and l.split()[-1].startswith("wset=")]
if L:
    l1 = np.array([float(x[1]) for x in L]); ws = np.array([int(x[-1].split("=")[1]) for x in L])
    out["loadavg_during_round"] = {"n_samples": len(L), "load1_median": float(np.median(l1)), "load1_p90": float(np.percentile(l1, 90)), "load1_max": float(l1.max()), "working_set_gib_max": float(ws.max() / 2 ** 30),
                                   "start": int(L[0][0]), "end": int(L[-1][0])}
v8L = [l.split() for l in open(T / "V8/raw/load_samples.txt") if len(l.split()) >= 6 and l.split()[-1].startswith("wset=")]
l8 = np.array([float(x[1]) for x in v8L]); out["loadavg_during_V8_round"] = {"n_samples": len(v8L), "load1_median": float(np.median(l8)), "load1_p90": float(np.percentile(l8, 90)), "load1_max": float(l8.max())}


def srv_facts(rootdir):
    srv = {}
    for d in sorted(glob.glob(str(rootdir / "server_*"))):
        d = pathlib.Path(d)
        if not (d / "gpu_samples.txt").exists():
            continue
        pk = max(int(l.split()[1]) for l in open(d / "gpu_samples.txt") if len(l.split()) == 2)
        log = (d / "serve_safe.log").read_text(errors="replace")
        srv[d.name] = {"gpu_total_used_peak_mib": pk, "gpu_used_before_mib": int(open(d / "gpu_used_before_mib").read()),
                       "cmdline": (d / "server_proc.txt").read_text().splitlines()[-1][-900:],
                       "log_gate_on": bool(re.search(r"subtask label: .* gate ON", log)),
                       "log_finger_cover_line": next((l[:260] for l in log.splitlines() if "constraint finger cover (T43 FC)" in l), None),
                       "log_ta_lines": [l.split("WARNING:__main__:")[-1][:230] for l in log.splitlines() if re.search(r"TO (lookahead|QP accuracy|deflection rate|servo accel) \(T43 TA", l)],
                       "log_path_check_line_present": "TO path check (T43 Q)" in log,
                       "log_autotune_off": "XLA autotuning off" in log, "log_obstacle_margin_10": "TO obstacle_margin: 10.0 mm" in log,
                       "n_constraint_spheres_in_records": sorted({int(np.load(f, allow_pickle=True)["sphere_radii"].shape[0]) for f in glob.glob(str(d / "server_constraints/run_*/chunk_*.npz"))[::25]})}
    return srv


out["servers"] = srv_facts(V / "runs" / ARM)
chain = {}
for p in sorted(glob.glob(str(V / "runs" / ARM / "chain_*.log"))):
    Lc = open(p, errors="replace").read().splitlines()
    chain[pathlib.Path(p).name] = {"n_end_lines": sum(l.startswith("end ") for l in Lc), "n_attempt2": sum("attempt=2" in l for l in Lc), "refused": [l[:160] for l in Lc if "REFUSED" in l], "died": [l[:120] for l in Lc if "died" in l],
                                  "code_md5_equal": sum("code md5 == V11-0" in l for l in Lc), "import_ok": sum("import check server+client: all under ${AG3S_PROJECT_ROOT}-t43v11" in l for l in Lc),
                                  "start_guard_ok": [l[:170] for l in Lc if l.startswith("start guard ok")],
                                  "server_log_line": [l[:700] for l in Lc if l.startswith("server log:")]}
out["chains"] = chain; out["failed_dirs"] = sorted(glob.glob(str(V / "runs" / ARM / "*/failed/*")))
rcl = []
for f in sorted(glob.glob(str(V / "runs" / f"chain*_{ARM}.out"))):
    rcl += open(f).read().strip().splitlines()
out["chain_rc_lines"] = rcl
an = json.load(open(V / f"analysis/analysis_{ARM}.json"))["runs"]; an8 = json.load(open(T / "V8/analysis/analysis.json"))["runs"]
for nm, A in ((ARM, an), ("V8", an8)):
    vk = collections.Counter()
    for r in A.values():
        for k, v in r["verdict_kind_counts"].items():
            vk[k] += v
    out[f"verdict_kind_counts_{nm}"] = dict(vk)
    out.setdefault("n_control_lt_600", {})[nm] = {k.split("/")[-1] + "/" + k.split("/")[1]: r["n_control"] for k, r in A.items() if r["n_control"] < 600}
    out.setdefault("wall_s_median", {})[nm] = float(np.median([r["wall_s"] for r in A.values()]))


# ---------- TO timing + TA fields from raw server chunk records (both arms) ----------
def lit(x):
    import ast
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return None
    return x


def st(v):
    v = [float(x) for x in v if x is not None]
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": len(a), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max()), "min": float(a.min()), "mean": float(a.mean())}


def raw_to(rootdir, arm):
    solve, sqp, qpit, lin, asm, qp, chk = [], [], [], [], [], [], []
    n = 0; hit_budget = 0; hit_iter = 0; n_all = 0; n_first = 0; status = collections.Counter()
    gate = collections.Counter(); tail_clear = []; tail_viol = []; gate_by_phase = collections.defaultdict(collections.Counter)
    dr = []; sa_n = 0; sa = collections.Counter(); sa_keys = set(); la_keys = set(); la_extra = []
    for d in sorted(glob.glob(str(rootdir / "server_*"))):
        for f in sorted(glob.glob(d + "/server_constraints/run_*/chunk_*.npz")):
            z = np.load(f, allow_pickle=True); s = json.loads(str(z["summary_json"])); to = s.get("to") or {}; m = to.get("metrics") or {}
            n_all += 1
            first = int(z["t_step"]) == 1
            n_first += first
            hit_budget += bool(to.get("time_budget_hit")); hit_iter += to.get("max_iterations_hit") in (True, "True"); status[str(to.get("status"))] += 1
            la = lit(m.get("lookahead"))
            if isinstance(la, dict):
                la_keys |= set(la)
                gate[str(la.get("gate"))] += 1
                g = s.get("grasp") or {}
                ph = "carry" if lit(g.get("attached")) else str(g.get("state"))
                gate_by_phase[ph][str(la.get("gate"))] += 1
                if la.get("tail_reference_clearance_m") is not None:
                    tail_clear.append(1e3 * float(la["tail_reference_clearance_m"]))
                for k in ("tail_violation_m", "tail_max_violation_m", "tail_clearance_m"):
                    if la.get(k) is not None:
                        tail_viol.append((k, 1e3 * float(la[k])))
            if first:
                continue
            n += 1
            solve.append(to.get("solve_ms")); sqp.append(to.get("sqp_iterations")); qpit.append(to.get("qp_iterations"))
            tm = lit(m.get("timing_ms")) or {}
            lin.append(tm.get("linearize")); asm.append(tm.get("assemble")); qp.append(tm.get("qp")); chk.append(tm.get("check"))
            x = lit(m.get("deflection_rate"))
            if isinstance(x, dict):
                dr.append(x)
            x = lit(m.get("servo_accel"))
            if isinstance(x, dict):
                sa_n += 1; sa_keys |= set(x)
    return {"arm": arm, "n_chunks_all": n_all, "n_first_chunks": n_first, "n_chunks_excl_first": n, "to_solve_ms": st(solve), "time_budget_hit_count_all": hit_budget, "max_iterations_hit_count_all": hit_iter,
            "max_iterations_hit_share_all": hit_iter / max(1, n_all), "sqp_iterations": st(sqp), "qp_iterations": st(qpit), "linearize_ms": st(lin), "assemble_ms": st(asm), "qp_ms": st(qp), "check_ms": st(chk),
            "to_status_counts": dict(status), "lookahead_gate_counts_all": dict(gate), "lookahead_gate_by_grasp_state": {k: dict(v) for k, v in gate_by_phase.items()}, "lookahead_keys": sorted(la_keys),
            "tail_reference_clearance_mm": st(tail_clear), "tail_other": {k: st([v for kk, v in tail_viol if kk == k]) for k in sorted({kk for kk, _ in tail_viol})},
            "deflection_rate_records": len(dr), "deflection_rate_keys": sorted({k for x in dr for k in x}), "servo_accel_records": sa_n, "servo_accel_keys": sorted(sa_keys)}


out["raw_to"] = {"V8": raw_to(T / "V8/runs/V8", "V8"), ARM: raw_to(V / "runs" / ARM, ARM)}
g = out["raw_to"][ARM]["lookahead_gate_counts_all"]
out["lookahead_gate_open_rate"] = {"counts": g, "open_share": (g.get("open", 0) / max(1, sum(g.values()))), "n": int(sum(g.values()))}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: out[k] for k in ("H_totals", "H_pairs_z4_rule")}, default=float)[:1800])
print(out["lookahead_gate_open_rate"], out["per_run_to_solve_median"]["ratio_median"], out.get("loadavg_during_round"))
for a in ("V8", ARM):
    x = out["raw_to"][a]; print(a, x["n_chunks_all"], x["to_solve_ms"], "budget", x["time_budget_hit_count_all"], "iterhit", x["max_iterations_hit_count_all"])
print(chain)
