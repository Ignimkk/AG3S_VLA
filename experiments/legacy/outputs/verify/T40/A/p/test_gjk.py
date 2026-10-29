from common.workspace import workspace_path as _ag3s_workspace_path
import t40a_lib as L, numpy as np, mujoco, collections
from scipy.optimize import lsq_linear
import gjk as G
R = L.load_run(_ag3s_workspace_path('outputs/verify/T39/runs/E0/E0_ep1807_s18071'))
rng = np.random.default_rng(1); ft = np.zeros(6)
def boxbox(a, b):
    ca, cb = L.d.geom_xpos[a], L.d.geom_xpos[b]; Ra, Rb = L.d.geom_xmat[a].reshape(3, 3), L.d.geom_xmat[b].reshape(3, 3)
    A = np.hstack([Ra, -Rb]); r = lsq_linear(A, cb - ca, bounds=(np.r_[-L.m.geom_size[a], -L.m.geom_size[b]], np.r_[L.m.geom_size[a], L.m.geom_size[b]]))
    return float(np.linalg.norm(A @ r.x - (cb - ca)))
others = np.concatenate([L.B_APPLE, L.CR_CRATE, L.A_GRIP, L.F_FRUITS['banana'], L.F_FRUITS['orange']])
err_bb = []; agree = collections.Counter(); dis = collections.Counter(); worst = collections.defaultdict(float)
for it in range(300):
    slot = ['bollard_0', 'divider_0'][it % 2]
    L.place(slot, (rng.uniform(0.4, 0.7), rng.uniform(-0.1, 0.35)), rng.uniform(-3, 3)); L.set_q(R['q'][rng.integers(0, 600)])
    for og in L.OB[slot]:
        sa = G.make_support(L.m, L.d, int(og))
        for g in others:
            sb = G.make_support(L.m, L.d, int(g))
            gd = G.gjk_dist(sa, sb, L.d.geom_xpos[g] - L.d.geom_xpos[og] + 1e-9)
            md = mujoco.mj_geomDistance(L.m, L.d, int(og), int(g), 0.3, ft)
            key = (int(L.m.geom_type[og]), int(L.m.geom_type[g]))
            if key == (6, 6):
                err_bb.append(abs(gd - boxbox(int(og), int(g))))
            if gd < 0.3 or md < 0.3:
                e = abs(max(md, 0) - min(gd, 0.3))
                (agree if e < 1e-4 else dis)[key] += 1; worst[key] = max(worst[key], e)
print('GJK vs exact box-box lsq: n', len(err_bb), 'max abs err m', max(err_bb))
print('GJK vs mj_geomDistance (|max(mj,0) - gjk| < 0.1 mm) agree', dict(agree), 'disagree', dict(dis), 'worst m', dict(worst))
