"""T41a control step 2 — re-solve recorded B ep1967 chunks offline with obstacle_margin 0 and 0.010 (verifier, CPU only).

usage: T41_SERVER_DIR=<T41a server dir> T41_MANIFEST=<manifest> qp_resolve_ctl.py RUN_DIR SEGMENT_FIRST_NPZ OUT_JSON [N_SEQ=2]

Inputs: the record of one run (server chunk npz seq 1..N, client frames.jsonl). Per seq the scene is rebuilt exactly as replay_t41.chunk_eval
(recorded fields, contact policy, target points, AG3S support planes re-fitted from filtered_cloud); variant m0 = that scene,
variant m10 = the same + obstacle_margin 0.010 (support excluded) + support_planes. One TrajOptChunkRefiner per variant, chained over
seq 1..N without reset (server-like: OSQP object / warm start / previous chunk carried), config = FIA 'replay' (server log settings, time
budget off = 1e9 ms, max_iterations = recorded sqp_iterations of that chunk), previous_physical_chunk = the recorded refined chunk of
seq-1 (seq 1: none). QpSolver.solve is wrapped: every QP (P, q, A, l, u, row_blocks) and its OSQP status / iterations / eps are kept.
Reported per seq: per SQP iteration, l/u entries that differ between m0 and m10 (by row block; with the m0 primal A x at the m0
solution = how far from the bound, i.e. active or inactive), q/P/A equal?, OSQP status / iterations; refined m10 - m0 (rad), and each
vs the recorded refined of the matching live run (m0 vs T40 G B1967 / control margin-0 run, m10 vs T41a B1967) when given."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib, dataclasses
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T41/p'))
import numpy as np
import mi_lib as MI  # noqa: E402
import replay_t41 as RT  # noqa: E402
from AG3S.trajopt.experiments import fixed_input_ablation as FIA  # noqa: E402
from AG3S.trajopt import qp as QP  # noqa: E402

RUN = pathlib.Path(sys.argv[1]); FIRST = pathlib.Path(sys.argv[2]); OUT = pathlib.Path(sys.argv[3])
NSEQ = int(sys.argv[4]) if len(sys.argv) > 4 else 2
REFS = json.loads(os.environ.get("T41_REF_RUNS", "{}"))  # {"m0": [seg_first_npz, ...], "m10": [...]}

rows = [json.loads(l) for l in open(RUN / "frames.jsonl")]
plan = {x["seq"]: x for x in rows if x["kind"] == "planning"}
base = FIRST.parent; i0 = int(FIRST.name[6:11])
files = [base / f"chunk_{i0 + k:05d}.npz" for k in range(NSEQ)]

LOG = []
_orig = QP.QpSolver.solve


def logged(self, problem, *, warm_start=True):
    sol = _orig(self, problem, warm_start=warm_start)
    LOG.append({"P": problem.P.copy(), "q": problem.q.copy(), "A": problem.A.copy(), "l": problem.l.copy(), "u": problem.u.copy(),
                "blocks": dict(problem.row_blocks), "status": sol.status, "iterations": sol.iterations, "x": sol.x.copy(),
                "eps_abs": self.config.eps_abs, "eps_rel": self.config.eps_rel, "max_iter": self.config.max_iter,
                "polish": self.config.polish, "adaptive_rho": self.config.adaptive_rho, "warm_start": warm_start})
    return sol


QP.QpSolver.solve = logged


def scenes(z, prow):
    e = RT.chunk_eval(z, prow)  # sets finger path on MI.meter (not used by the refiner)
    summ = e["summ"]
    fieldc = FIA.build_field(z, MI.OUTSIDE)
    g = summ.get("grasp") or {}
    att = g.get("attached") if g.get("attached_in_constraints") else None
    has_target = e["has_target"]
    margin, mask, _, _, _ = FIA.contact_policy(MI.settings, MI.robot.sphere_link_names, has_target=has_target,
                                                has_free=bool(fieldc.target_free_layers), attached=att is not None)
    s0 = FIA.make_scene(fieldc, MI.RADII, MI.settings.esdf_margin, target_points=z["target_points"] if has_target else None,
                        margin=margin, mask=mask)
    planes, _ = RT.planes_of(z)
    s1 = dataclasses.replace(s0, obstacle_margin=0.010, obstacle_margin_support=False, support_planes=planes)
    return s0, s1, e, summ


def block_of(i, blocks):
    for n, (a, b) in blocks.items():
        if a <= i < b:
            return n
    return "?"


runners = {}
QPLOG = {}


def qp_compare(A0, A1):
    out = []
    for j in range(min(len(A0), len(A1))):
        a, b = A0[j], A1[j]
        same_shape = a["A"].shape == b["A"].shape
        e = {"sqp_qp_index": j, "status": [a["status"], b["status"]], "osqp_iterations": [a["iterations"], b["iterations"]],
             "shape_equal": bool(same_shape), "P_equal": bool(same_shape and (a["P"] != b["P"]).nnz == 0),
             "q_equal": bool(np.array_equal(a["q"], b["q"])), "A_equal": bool(same_shape and (a["A"] != b["A"]).nnz == 0),
             "blocks_equal": a["blocks"] == b["blocks"], "x_max_abs_diff": float(np.abs(a["x"] - b["x"]).max()) if a["x"].shape == b["x"].shape else None}
        if same_shape:
            dl = np.flatnonzero(a["l"] != b["l"]); du = np.flatnonzero(a["u"] != b["u"])
            rows_ = np.union1d(dl, du)
            e["n_rows_l_differ"] = int(dl.size); e["n_rows_u_differ"] = int(du.size)
            by = {}
            Ax0 = a["A"] @ a["x"]
            for r in rows_:
                bn = block_of(int(r), a["blocks"]); d = by.setdefault(bn, {"n": 0, "bound_delta": [], "m0_slack_to_lower": [], "m0_slack_to_upper": []})
                d["n"] += 1
                d["bound_delta"].append(float((b["l"][r] - a["l"][r]) if r in dl else (b["u"][r] - a["u"][r])))
                d["m0_slack_to_lower"].append(float(Ax0[r] - np.clip(a["l"][r], -1e20, 1e20)))
                d["m0_slack_to_upper"].append(float(np.clip(a["u"][r], -1e20, 1e20) - Ax0[r]))
            for bn, d in by.items():
                for kk in ("bound_delta", "m0_slack_to_lower", "m0_slack_to_upper"):
                    v = np.asarray(d[kk]); d[kk] = {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}
                # an inequality row is 'active' at the m0 solution if A x is within eps of a bound that changed
                tol = a["eps_abs"]
                d["n_m0_active_within_eps_abs"] = None
            n_act = 0
            for r in rows_:
                lo = Ax0[r] - a["l"][r]; hi = a["u"][r] - Ax0[r]
                if min(abs(lo), abs(hi)) <= 10 * a["eps_abs"]:
                    n_act += 1
            e["differing_rows_by_block"] = by
            e["n_differing_rows_m0_within_10eps_of_a_bound"] = int(n_act)
            e["m0_min_slack_over_differing_rows"] = float(min(min(Ax0[r] - a["l"][r], a["u"][r] - Ax0[r]) for r in rows_)) if rows_.size else None
            e["row_blocks"] = {kk: list(vv) for kk, vv in a["blocks"].items()}
        out.append(e)
    return out

res = {"definitions": __doc__, "run": str(RUN), "segment_first": str(FIRST), "seqs": {}}
prev = None
for k, f in enumerate(files):
    z = np.load(f, allow_pickle=True); seq = int(z["t_step"]); p = plan[seq]
    s0, s1, e, summ = scenes(z, p)
    q_now = FIA.q_now_from_qpos(p["qpos"], MI.qadr)
    to = summ.get("to") or {}
    iters = int(to.get("sqp_iterations") or 1)
    cfg = FIA.to_config(MI.settings, collision=True, cost=MI.settings.cost, sqp={"time_budget_ms": FIA.NO_CLOCK_MS, "max_iterations": max(1, iters)})
    ref_chunk = np.asarray(z["reference_chunk"], np.float64)
    out = {"seq": seq, "recorded_sqp_iterations": iters, "recorded_qp_iterations": (to.get("metrics") or {}).get("qp_iterations"),
           "variants": {}, "qp_logs": {}}
    trajs = {}
    for name, sc in (("m0", s0), ("m10", s1)):
        if name not in runners:
            runners[name] = FIA.Instrumented(MI.robot, MI.layout, cfg, {"value": None}); runners[name].refiner.reset()
        r = runners[name]
        r.refiner.optimizer.config = r.refiner.optimizer.config.with_overrides({"sqp": {"max_iterations": cfg.sqp.max_iterations, "time_budget_ms": cfg.sqp.time_budget_ms}})
        r.holder["value"] = (sc, q_now, summ.get("status") == "ok")
        r.captured = []
        LOG.clear()
        r.refiner.refine(ref_chunk, {"t_step": seq, "previous_physical_chunk": prev})
        result = r.refiner.last_result
        trajs[name] = np.asarray(result.trajectory)
        out["qp_logs"][name] = [dict(x) for x in LOG]
        QPLOG.setdefault(seq, {})[name] = [dict(x) for x in LOG]
        out["variants"][name] = {"status": result.status.value, "sqp_iterations": result.metrics.get("sqp_iterations"),
                                 "qp_iterations": result.metrics.get("qp_iterations"), "returned": result.metrics.get("returned"),
                                 "clearance_m": result.metrics.get("clearance_m"), "max_violation_pair": str(result.metrics.get("max_violation_pair"))[:400],
                                 "qp": [{"status": x["status"], "iterations": x["iterations"], "eps_abs": x["eps_abs"], "eps_rel": x["eps_rel"],
                                         "max_iter": x["max_iter"], "polish": x["polish"], "adaptive_rho": x["adaptive_rho"], "warm_start": x["warm_start"],
                                         "n_var": int(x["P"].shape[0]), "n_rows": int(x["A"].shape[0])} for x in LOG]}
    rec_ref = MI.layout.chunk_to_trajectory(np.asarray(z["refined_chunk"], np.float64))[:, :trajs["m0"].shape[1]]
    out["refined_m10_minus_m0_max_abs_rad"] = float(np.abs(trajs["m10"] - trajs["m0"]).max())
    out["refined_m10_minus_m0_max_abs_rad_per_step"] = np.abs(trajs["m10"] - trajs["m0"]).max(axis=0).tolist()
    out["m0_vs_this_record_refined_max_abs_rad"] = float(np.abs(trajs["m0"] - rec_ref).max())
    out["m10_vs_this_record_refined_max_abs_rad"] = float(np.abs(trajs["m10"] - rec_ref).max())
    for nm, lst in REFS.items():
        if k < len(lst):
            zz = np.load(lst[k], allow_pickle=True)
            rr = MI.layout.chunk_to_trajectory(np.asarray(zz["refined_chunk"], np.float64))[:, :trajs["m0"].shape[1]]
            out[f"{nm}_vs_record_{pathlib.Path(lst[k]).parts[-4]}_max_abs_rad"] = float(np.abs(trajs[nm] - rr).max())
    # QP-by-QP comparison (same SQP iteration index j)
    out.pop("qp_logs")
    out["qp_compare"] = qp_compare(QPLOG[seq]["m0"], QPLOG[seq]["m10"])
    res["seqs"][seq] = out
    prev = np.asarray(z["refined_chunk"], np.float64)
    z.close()
OUT.write_text(json.dumps(MI.jl(res), indent=1))
print(json.dumps({k: {kk: v[kk] for kk in ("refined_m10_minus_m0_max_abs_rad", "m0_vs_this_record_refined_max_abs_rad", "m10_vs_this_record_refined_max_abs_rad")} for k, v in res["seqs"].items()}))
