"""T43 TP — the first QP of a chunk: what to warm-start it with. Captured (previous chunk's last x, y)
vs x = [current iterate, 0 slack] (from the box rows' midpoint) with the captured y / y = 0 / cold."""
import pickle, sys, time, collections
import numpy as np, scipy.sparse as sp, osqp
INF = 1e20
caps = []
for f in sys.argv[2:]:
    cs = pickle.load(open(f, "rb")); cnt = collections.Counter()
    for c in cs:
        cnt[(c["seq"], c["horizon"])] += 1
        c["pos"] = cnt[(c["seq"], c["horizon"])]
    caps += cs
eps = float(sys.argv[1])
def solve(c, ws, **kw):
    m = osqp.OSQP(); base = dict(eps_abs=eps, eps_rel=eps, max_iter=4000, polishing=False, adaptive_rho=True, verbose=False, warm_starting=True); base.update(kw)
    m.setup(P=sp.triu(c["P"], format="csc"), q=c["q"], A=c["A"], l=np.clip(c["l"], -INF, INF), u=np.clip(c["u"], -INF, INF), **base)
    if ws is not None:
        m.warm_start(x=ws[0], y=ws[1]) if ws[1] is not None else m.warm_start(x=ws[0])
    t0 = time.perf_counter(); r = m.solve(); return r, (time.perf_counter() - t0) * 1e3
R = collections.defaultdict(list)
for c in caps:
    xs = solve(c, None, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000, polishing=True)[0].x[:c["n_q"]]
    n = c["n_q"]; mid = 0.5 * (np.clip(c["l"][:n], -10, 10) + np.clip(c["u"][:n], -10, 10))
    x0 = np.concatenate([mid, np.zeros(c["P"].shape[0] - n)])
    opts = {"captured": c["ws"], "iterate+y": (x0, None if c["ws"] is None else c["ws"][1]),
            "iterate+y0": (x0, np.zeros(c["A"].shape[0])), "cold": None}
    for k, ws in opts.items():
        r, dt = solve(c, ws)
        R[(c["pos"] == 1, k)].append((dt, r.info.iter, np.abs(r.x[:n] - xs).max()))
for (first, k), v in sorted(R.items()):
    v = np.array(v)
    print(f"{'QP#1' if first else 'QP#2+'} {k:11s} n {len(v):3d} ms sum {v[:,0].sum():7.1f} p90 {np.quantile(v[:,0],.9):6.1f} | iters med {np.median(v[:,1]):5.0f} p90 {np.quantile(v[:,1],.9):5.0f} | dQ p90 {np.quantile(v[:,2],.9):.1e} max {v[:,2].max():.1e}")
