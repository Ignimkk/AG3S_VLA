"""T43 TP — 'polish first' : eps 1e-3 + polishing; if the polish did not succeed, continue the same
ADMM (warm, same OSQP object) to eps 1e-4. Compared with e4 / e3_polish on the captured problems."""
import pickle, sys, time
import numpy as np, scipy.sparse as sp, osqp
caps = []
for f in sys.argv[1:]:
    caps += pickle.load(open(f, "rb"))
INF = 1e20
def setup(c, **kw):
    m = osqp.OSQP()
    base = dict(eps_abs=1e-3, eps_rel=1e-3, max_iter=4000, polishing=False, adaptive_rho=True, verbose=False, warm_starting=True)
    base.update(kw)
    m.setup(P=sp.triu(c["P"], format="csc"), q=c["q"], A=c["A"], l=np.clip(c["l"], -INF, INF), u=np.clip(c["u"], -INF, INF), **base)
    if c["ws"] is not None:
        m.warm_start(x=c["ws"][0], y=c["ws"][1])
    return m
R = {"e4": [], "hyb": [], "e3p": []}
for c in caps:
    m = setup(c, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000, polishing=True); xs = m.solve().x[:c["n_q"]]
    m = setup(c, eps_abs=1e-4, eps_rel=1e-4); t0 = time.perf_counter(); r = m.solve(); dt = (time.perf_counter()-t0)*1e3
    R["e4"].append((dt, r.info.iter, np.abs(r.x[:c["n_q"]]-xs).max()))
    m = setup(c, polishing=True); t0 = time.perf_counter(); r = m.solve()
    it = r.info.iter; x = r.x
    R["e3p"].append(((time.perf_counter()-t0)*1e3, it, np.abs(x[:c["n_q"]]-xs).max()))
    if r.info.status_polish != 1:
        m.update_settings(eps_abs=1e-4, eps_rel=1e-4, polishing=False)
        m.warm_start(x=r.x, y=r.y)
        r2 = m.solve(); it += r2.info.iter; x = r2.x
    dt = (time.perf_counter()-t0)*1e3
    R["hyb"].append((dt, it, np.abs(x[:c["n_q"]]-xs).max()))
for k, v in R.items():
    v = np.array(v)
    print(f"{k:4s} ms sum {v[:,0].sum():7.1f} p90 {np.quantile(v[:,0],.9):6.1f} max {v[:,0].max():6.1f} | iters p90 {np.quantile(v[:,1],.9):6.0f} | dQ med {np.median(v[:,2]):.1e} p90 {np.quantile(v[:,2],.9):.1e} max {v[:,2].max():.1e}")
