"""T43 TP — OSQP settings vs iterations / time / distance to a tight reference solution, on the
captured V11AB subproblems (qp_capture.py). Fresh setup per solve + the captured warm start."""
import pickle, sys, time, json
import numpy as np, scipy.sparse as sp, osqp
caps = []
for f in sys.argv[2:]:
    caps += pickle.load(open(f, "rb"))
INF = 1e20
SET = {
    "e3": dict(eps_abs=1e-3, eps_rel=1e-3),
    "e4": dict(eps_abs=1e-4, eps_rel=1e-4),
    "e3_polish": dict(eps_abs=1e-3, eps_rel=1e-3, polishing=True),
    "e4_polish": dict(eps_abs=1e-4, eps_rel=1e-4, polishing=True),
    "e4_it500": dict(eps_abs=1e-4, eps_rel=1e-4, max_iter=500),
    "e4_it1000": dict(eps_abs=1e-4, eps_rel=1e-4, max_iter=1000),
    "e4_rho1": dict(eps_abs=1e-4, eps_rel=1e-4, rho=1.0),
    "e4_rho001": dict(eps_abs=1e-4, eps_rel=1e-4, rho=0.01),
    "e4_sig": dict(eps_abs=1e-4, eps_rel=1e-4, sigma=1e-4),
    "e4_tol2": dict(eps_abs=1e-4, eps_rel=1e-4, adaptive_rho_tolerance=2.0, adaptive_rho_interval=25),
    "e4_alpha18": dict(eps_abs=1e-4, eps_rel=1e-4, alpha=1.8),
    "e4_scal25": dict(eps_abs=1e-4, eps_rel=1e-4, scaling=25),
    "e3_polish_ri": dict(eps_abs=1e-3, eps_rel=1e-3, polishing=True, polish_refine_iter=10, delta=1e-7),
}
def solve(c, st, ref=False):
    m = osqp.OSQP()
    kw = dict(eps_abs=1e-3, eps_rel=1e-3, max_iter=4000, polishing=False, adaptive_rho=True, verbose=False, warm_starting=True)
    kw.update(st)
    if ref:
        kw.update(eps_abs=1e-8, eps_rel=1e-8, max_iter=200000, polishing=True)
    m.setup(P=sp.triu(c["P"], format="csc"), q=c["q"], A=c["A"], l=np.clip(c["l"], -INF, INF), u=np.clip(c["u"], -INF, INF), **kw)
    if c["ws"] is not None:
        m.warm_start(x=c["ws"][0], y=c["ws"][1])
    t0 = time.perf_counter(); r = m.solve(); dt = (time.perf_counter() - t0) * 1e3
    return r, dt
out = {k: {"it": [], "ms": [], "dq": [], "fail": 0} for k in SET}
for i, c in enumerate(caps):
    r0, _ = solve(c, {}, ref=True)
    xs = r0.x[:c["n_q"]]
    for k, st in SET.items():
        r, dt = solve(c, st)
        ok = str(r.info.status) in ("solved", "solved inaccurate") or (st.get("max_iter") and np.all(np.isfinite(r.x)))
        out[k]["it"].append(r.info.iter); out[k]["ms"].append(dt)
        out[k].setdefault("polish", []).append(int(getattr(r.info, "status_polish", 0)))
        if not ok:
            out[k]["fail"] += 1; out[k]["dq"].append(np.nan)
        else:
            out[k]["dq"].append(float(np.abs(r.x[:c["n_q"]] - xs).max()))
q = lambda a, p: float(np.nanquantile(a, p))
print(f"n={len(caps)}")
for k, v in out.items():
    print(f"{k:16s} iters med {np.median(v['it']):6.0f} p90 {q(v['it'],.9):6.0f} max {max(v['it']):5d} | ms sum {sum(v['ms']):7.1f} p90 {q(v['ms'],.9):6.1f} | |dQ|inf rad med {q(v['dq'],.5):.2e} p90 {q(v['dq'],.9):.2e} max {np.nanmax(v['dq']):.2e} | fail {v['fail']} polish_ok {sum(1 for p in v['polish'] if p == 1)}")
json.dump(out, open(sys.argv[1], "w"))
