"""T43 Q: what does FK (+jac) and an ESDF query cost per configuration (the sub-sample budget)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, time, numpy as np
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/impl/T41b/p'))
import replay_c1807 as R
sh = R.setup()
lin = sh['meter'].lin
import casadi as ca
q = ca.SX.sym('q', int(sh['robot'].nq), 1)
print('n_spheres', lin.n_spheres, 'nq_opt', lin.nq_opt, 'horizon', lin.horizon, 'params', lin.param_names)
T = sh['T']; seq = 28
z = np.load(R.CH / f'chunk_{seq + R.OFFSET:05d}.npz', allow_pickle=False)
field = T.build_field(z, outside_distance=sh['outside'])
q_now = T.q_now_from_qpos(sh['rows'][seq]['qpos'], sh['qadr'])
out = {}
for n in (8, 32, 64, 128):
    fn = lin._fk.map(n)
    full = np.tile(q_now[:, None], (1, n)) + 0.01 * np.random.randn(len(q_now), n)
    args = [full] if lin._p_sym is None else [full, np.tile(lin.param_path()[:, :1], (1, n))]
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(5): c, j = fn(*args); cn = np.asarray(c); jn = np.asarray(j.nonzeros())
    t_fk = (time.perf_counter() - t0) / 5 * 1e3
    pts = np.asarray(c).T.reshape(-1, 3)
    field.distance(pts)
    t0 = time.perf_counter()
    for _ in range(5): field.distance(pts)
    t_d = (time.perf_counter() - t0) / 5 * 1e3
    t0 = time.perf_counter()
    for _ in range(5): field.gradient(pts[: n * 24])
    t_g = (time.perf_counter() - t0) / 5 * 1e3
    out[n] = dict(fk_jac_ms=round(t_fk, 2), esdf_distance_ms=round(t_d, 2), esdf_grad_24rows_ms=round(t_g, 2), n_points=len(pts))
    print(n, out[n], flush=True)
