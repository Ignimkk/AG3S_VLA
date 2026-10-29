"""T43 TP — does a small quadratic weight on the slack variables (P_ss = rho_s) make ADMM converge
faster, and how far does the regularized optimum move from the exact one (Q part, rad)?"""
import pickle, sys, time, collections
import numpy as np, scipy.sparse as sp, osqp
INF = 1e20
caps = []
for f in sys.argv[1:]:
    caps += pickle.load(open(f, "rb"))
def solve(c, P, eps, ws=True, **kw):
    m = osqp.OSQP(); base = dict(eps_abs=eps, eps_rel=eps, max_iter=4000, polishing=False, adaptive_rho=True, verbose=False, warm_starting=True); base.update(kw)
    m.setup(P=sp.triu(P, format="csc"), q=c["q"], A=c["A"], l=np.clip(c["l"], -INF, INF), u=np.clip(c["u"], -INF, INF), **base)
    if ws and c["ws"] is not None:
        m.warm_start(x=c["ws"][0], y=c["ws"][1])
    t0 = time.perf_counter(); r = m.solve(); return r, (time.perf_counter() - t0) * 1e3
R = collections.defaultdict(list)
for c in caps:
    n = c["n_q"]; N = c["P"].shape[0]
    xs = solve(c, c["P"], 1e-8, max_iter=200000, polishing=True)[0].x[:n]
    for rs in (0.0, 1e-3, 1e-2, 1e-1, 1.0):
        d = np.zeros(N); d[n:] = rs
        P = (c["P"] + sp.diags(d)).tocsc()
        xr = solve(c, P, 1e-8, max_iter=200000, polishing=True)[0].x[:n] if rs else xs
        for eps in (1e-3, 1e-4):
            r, dt = solve(c, P, eps)
            R[(rs, eps)].append((dt, r.info.iter, np.abs(r.x[:n] - xs).max(), np.abs(xr - xs).max()))
for (rs, eps), v in sorted(R.items()):
    v = np.array(v)
    print(f"slack P {rs:6g} eps {eps:g}: ms sum {v[:,0].sum():7.1f} p90 {np.quantile(v[:,0],.9):6.1f} | iters med {np.median(v[:,1]):5.0f} p90 {np.quantile(v[:,1],.9):5.0f} | dQ vs exact p90 {np.quantile(v[:,2],.9):.1e} max {v[:,2].max():.1e} | reg optimum shift p90 {np.quantile(v[:,3],.9):.1e} max {v[:,3].max():.1e}")
