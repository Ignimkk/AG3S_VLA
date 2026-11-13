import json, numpy as np, glob, sys
from AG3S.adapters.panda import model as pm, action as pa
from AG3S.ag3s.robot_models.urdf_sphere_chain import UrdfSphereChain
kin = pa.PandaKinematics(UrdfSphereChain(pm.load_panda_urdf(), pm.PANDA_ARM_JOINTS))
np.set_printoptions(precision=3, suppress=True, linewidth=220)
import os
B = os.environ.get("AUDIT_BUNDLE", "3_raw_i19").rstrip("/") + "/"   # the folder with the raw episodes

def ik_lm(kin, p_tgt, R_tgt, q_seed, iters=400):
    """reference solver: Levenberg-Marquardt with adaptive damping + bound clipping; returns best (lowest error) iterate"""
    q = np.clip(np.asarray(q_seed, float).copy(), kin.lower, kin.upper)
    lam = 1e-3
    def err(q):
        p, R = kin.fk(q); return np.concatenate([p_tgt - p, pa.matrix_to_rotvec(R_tgt @ R.T)])
    e = err(q); best = (np.linalg.norm(e), q.copy())
    for it in range(iters):
        if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4: break
        J = kin.jacobian(q)
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
        n = np.abs(dq).max()
        if n > 0.2: dq *= 0.2 / n
        qn = np.clip(q + dq, kin.lower, kin.upper); en = err(qn)
        if np.linalg.norm(en) < np.linalg.norm(e):
            q, e = qn, en; lam = max(lam / 3, 1e-7)
            if np.linalg.norm(e) < best[0]: best = (np.linalg.norm(e), q.copy())
        else:
            lam = min(lam * 5, 1e2)
    p, R = kin.fk(best[1])
    return best[1], {"iters": it + 1, "pos_err_m": float(np.linalg.norm(p_tgt - p)), "rot_err_rad": float(np.linalg.norm(pa.matrix_to_rotvec(R_tgt @ R.T)))}

def chunk(kin, A, q_now, solver):
    (ap, bp), (ar, br) = pa.RESPONSE["pos"], pa.RESPONSE["rot"]
    yp = np.zeros(3); yr = np.zeros(3); p, R = kin.fk(q_now); q = q_now.copy(); out = []
    for k in range(len(A)):
        a = np.clip(A[k], -1, 1); yp = ap * yp + bp * 0.05 * a[:3]; yr = ar * yr + br * 0.5 * a[3:6]
        p = p + yp; R = pa.rotvec_to_matrix(yr) @ R
        q, rec = solver(kin, p, R, q); out.append((q.copy(), rec))
    return out

d = "goal-LI-i19_t0"
calls = [json.loads(l) for l in open(glob.glob(B + d + "/*_calls.jsonl")[0], encoding="utf-8")]
g = np.load(glob.glob(B + d + "/*_geom.npz")[0], allow_pickle=True); rs = g["robot_state/head"]; oi = list(g["obs_index"])
dls = lambda kin, p, R, q: kin.ik(p, R, q)
for s in (26, 27, 34, 45):
    q0 = rs[oi.index(calls[s]["obs_index"]), :7]
    print(f"== seq {s}  q4={q0[3]:.3f}")
    for name, a in (("+z", [0,0,1,0,0,0]), ("-z", [0,0,-1,0,0,0]), ("+x", [1,0,0,0,0,0]), ("-x", [-1,0,0,0,0,0]), ("+y", [0,1,0,0,0,0]), ("-y", [0,-1,0,0,0,0])):
        A = np.tile(np.array(a, float), (10, 1))
        r1 = chunk(kin, A, q0, dls); r2 = chunk(kin, A, q0, ik_lm)
        e1 = [round(1000 * r[1]["pos_err_m"], 1) for r in r1]; e2 = [round(1000 * r[1]["pos_err_m"], 1) for r in r2]
        j1 = max(np.abs(r1[k][0] - (r1[k-1][0] if k else q0)).max() for k in range(10)); j2 = max(np.abs(r2[k][0] - (r2[k-1][0] if k else q0)).max() for k in range(10))
        print(f"  {name}: server DLS pos_err_mm rows {e1}  max joint jump/row {j1:.2f} rad")
        print(f"      LM reference pos_err_mm rows {e2}  max joint jump/row {j2:.2f} rad  iters max {max(r[1]['iters'] for r in r2)}")
