"""mj_geomDistance consistency survey: same pair at distmax 0.3 vs 0.05 must agree when < 0.05; plus independent
check against the bounded least-squares distance for box-box pairs (exact)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import t40a_lib as L, numpy as np, mujoco, math, json, collections, sys
from scipy.optimize import lsq_linear
mode = sys.argv[1]
if mode == "noccd":
    L.m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
R = L.load_run(_ag3s_workspace_path('outputs/verify/T39/runs/E0/E0_ep1807_s18071'))
rng = np.random.default_rng(0); bad = collections.Counter(); tot = collections.Counter(); bb_err = []
T = lambda g: int(L.m.geom_type[g])
others = np.concatenate([L.B_APPLE, L.CR_CRATE, L.A_GRIP, L.F_FRUITS['banana'], L.F_FRUITS['orange']])
ft = np.zeros(6)
def boxbox(a, b):
    ca, cb = L.d.geom_xpos[a], L.d.geom_xpos[b]; Ra, Rb = L.d.geom_xmat[a].reshape(3, 3), L.d.geom_xmat[b].reshape(3, 3)
    ha, hb = L.m.geom_size[a], L.m.geom_size[b]
    A = np.hstack([Ra, -Rb]); r = lsq_linear(A, cb - ca, bounds=(np.r_[-ha, -hb], np.r_[ha, hb]))
    return float(np.linalg.norm(A @ r.x - (cb - ca)))
for it in range(300):
    slot = ['bollard_0', 'divider_0'][it % 2]
    L.place(slot, (rng.uniform(0.4, 0.7), rng.uniform(-0.1, 0.35)), rng.uniform(-3, 3)); L.set_q(R['q'][rng.integers(0, 600)])
    for og in L.OB[slot]:
        for g in others:
            a = mujoco.mj_geomDistance(L.m, L.d, int(og), int(g), 0.3, ft)
            c = mujoco.mj_geomDistance(L.m, L.d, int(og), int(g), 0.05, ft)
            key = (T(og), T(g)); tot[key] += 1
            if abs(min(a, 0.05) - c) > 1e-6: bad[key] += 1
            if key == (6, 6):
                e = boxbox(int(og), int(g))
                if e > 1e-6 or a > 1e-6:
                    bb_err.append(a - min(e, 0.3))
print(mode, 'tot', dict(tot), 'bad', dict(bad))
bb = np.abs(np.array(bb_err)); print('box-box |mj(0.3) - lsq| max', bb.max(), 'n>1mm', int((bb > 1e-3).sum()), 'of', len(bb))
