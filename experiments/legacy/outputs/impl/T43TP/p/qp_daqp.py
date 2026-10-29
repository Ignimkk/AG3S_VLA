"""T43 TP — DAQP (dense dual active-set, already in the venv) on the captured subproblems: time and
distance to the exact optimum. Slack columns get P = 1e-6 (DAQP needs P > 0; the shift is < 1e-8 rad,
qp_reg.py)."""
import pickle, sys, time
import numpy as np, scipy.sparse as sp, osqp
from ctypes import c_int
import daqp
INF = 1e20
caps = []
for f in sys.argv[1:]:
    caps += pickle.load(open(f, "rb"))
def exact(c):
    m = osqp.OSQP()
    m.setup(P=sp.triu(c["P"], format="csc"), q=c["q"], A=c["A"], l=np.clip(c["l"], -INF, INF), u=np.clip(c["u"], -INF, INF), eps_abs=1e-9, eps_rel=1e-9, max_iter=200000, polishing=True, verbose=False)
    return m.solve().x
res = []
for c in caps[::3]:
    n = c["n_q"]; N = c["P"].shape[0]
    xs = exact(c)[:n]
    P = (c["P"] + sp.diags(np.r_[np.zeros(n), np.full(N - n, 1e-6)])).toarray()
    A = c["A"].toarray(); l = np.clip(c["l"], -1e30, 1e30); u = np.clip(c["u"], -1e30, 1e30)
    # identity rows as simple bounds are not separated here (general rows only)
    sense = np.zeros(A.shape[0], dtype=c_int)
    t0 = time.perf_counter()
    x, fval, flag, info = daqp.solve(P, c["q"], A, u, l, sense)
    dt = (time.perf_counter() - t0) * 1e3
    res.append((dt, flag, np.abs(np.asarray(x)[:n] - xs).max() if flag > 0 else np.nan, info.get("iterations", -1)))
r = np.array(res)
print("n", len(r), "ms med", np.median(r[:,0]), "p90", np.quantile(r[:,0], .9), "max", r[:,0].max(), "| flags", np.unique(r[:,1], return_counts=True), "| dQ p90", np.nanquantile(r[:,2], .9), "max", np.nanmax(r[:,2]), "| iters med", np.median(r[:,3]))
