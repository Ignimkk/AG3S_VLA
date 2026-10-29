"""T40 A detour room: kinematic upper limit of the left TCP above a given xy (verifier, CPU).

max  z(site left_ee)  over the 7 left-arm joints (left_arm_0..6, qpos 20..26, MJCF joint ranges),
s.t. |xy(site left_ee) - target_xy| <= 5 mm and angle(gripper approach axis, recorded carry approach axis) <= 15 deg.
Everything else (torso, head, right arm, gripper opening) fixed at the recorded carry row used as start.
approach axis = the site_xmat column with the largest |z| component averaged over the recorded carry rows (sign kept).
Multi-start: 12 recorded carry rows (evenly spaced in the window) + the row whose TCP xy is nearest target.
SLSQP (scipy), finite differences on mj_kinematics. Self-collision and collision with the scene are NOT checked
(upper bound of the reachable height, not a collision-free plan).
Reported: tcp_z_max above table; apple_bottom_max_est = tcp_z_max - median(tcp z - apple bottom z) over the carry rows."""
import numpy as np, mujoco
from scipy.optimize import minimize
import t40a_lib as L

ARM = list(range(20, 27))
LO, HI = L.m.jnt_range[20:27, 0], L.m.jnt_range[20:27, 1]
S = L.SITE_LEE


def _fk(q_base, x):
    q = q_base.copy(); q[ARM] = x; L.set_q(q)
    return L.d.site_xpos[S].copy(), L.d.site_xmat[S].reshape(3, 3).copy()


def approach_axis(Qcarry):
    cols = []
    for q in Qcarry:
        L.set_q(q); cols.append(L.d.site_xmat[S].reshape(3, 3).copy())
    M = np.mean(cols, axis=0); k = int(np.argmax(np.abs(M[2, :])))
    a = M[:, k] / np.linalg.norm(M[:, k])
    return k, a


def tcp_minus_apple_bottom(Qcarry):
    off = []
    for q in Qcarry:
        L.set_q(q)
        g = L.B_APPLE[0]; sup = L.gjk.make_support(L.m, L.d, int(g))
        zb = sup(np.array([0, 0, -1.0]))[2]
        off.append(L.d.site_xpos[S][2] - zb)
    return float(np.median(off))


def max_tcp_z(target_xy, Qcarry, ang_deg=15.0, tol_xy=0.005):
    k, a_ref = approach_axis(Qcarry)
    cos_min = np.cos(np.radians(ang_deg))
    idx = list(np.linspace(0, len(Qcarry) - 1, 12).astype(int))
    dxy = []
    for q in Qcarry:
        L.set_q(q); dxy.append(np.linalg.norm(L.d.site_xpos[S][:2] - target_xy))
    idx.append(int(np.argmin(dxy)))
    best = None
    for i in idx:
        qb = Qcarry[i].copy()
        f = lambda x: -_fk(qb, x)[0][2]
        cons = [{"type": "ineq", "fun": lambda x: tol_xy ** 2 - np.sum((_fk(qb, x)[0][:2] - target_xy) ** 2)},
                {"type": "ineq", "fun": lambda x: _fk(qb, x)[1][:, k] @ a_ref - cos_min}]
        r = minimize(f, qb[ARM], method="SLSQP", bounds=list(zip(LO, HI)), constraints=cons,
                     options={"maxiter": 300, "ftol": 1e-9})
        p, Rm = _fk(qb, r.x)
        ok = np.linalg.norm(p[:2] - target_xy) <= tol_xy + 1e-4 and Rm[:, k] @ a_ref >= cos_min - 1e-4
        if ok and (best is None or p[2] > best["tcp_z"]):
            best = {"tcp_z": float(p[2]), "q_arm": r.x.tolist(), "start_row": int(i),
                    "xy_err_m": float(np.linalg.norm(p[:2] - target_xy)), "axis_cos": float(Rm[:, k] @ a_ref)}
    return {"axis_col": k, "axis_ref": a_ref.tolist(), "best": best,
            "tcp_z_max_above_table_m": None if best is None else best["tcp_z"] - L.TABLE_Z}
