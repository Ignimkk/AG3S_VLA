import json, numpy as np, glob, sys
from AG3S.adapters.panda import model as pm, action as pa
from AG3S.ag3s.robot_models.urdf_sphere_chain import UrdfSphereChain
kin = pa.PandaKinematics(UrdfSphereChain(pm.load_panda_urdf(), pm.PANDA_ARM_JOINTS))
import os
B = os.environ.get("AUDIT_BUNDLE", "3_raw_i19").rstrip("/") + "/"   # the folder with the raw episodes
np.set_printoptions(precision=3, suppress=True, linewidth=220)
for d in sys.argv[1:]:
    calls = [json.loads(l) for l in open(glob.glob(B + d + "/*_calls.jsonl")[0], encoding="utf-8")]
    cs = [json.loads(l) for l in open(B + d + "/chunk_summary.jsonl", encoding="utf-8")]
    g = np.load(glob.glob(B + d + "/*_geom.npz")[0], allow_pickle=True)
    rs = g["robot_state/head"]; oi = list(g["obs_index"])
    print("=====", d)
    print(f"{'seq':>3} {'latch':>9} {'qp':>5} {'ikerr':>7} | pred-vs-actual EE err after 5 steps (mm): pos, |actual move| mm, |pred move| mm, joint err max rad")
    for s in range(len(calls) - 1):
        fb = cs[s + 1].get("exec_feedback") or {}
        arm = fb.get("applied_arm")
        if not arm: continue
        arm = np.asarray(arm, float)
        k0 = oi.index(calls[s]["obs_index"]); k1 = oi.index(calls[s + 1]["obs_index"])
        n = k1 - k0
        q0 = rs[k0, :7]; q1 = rs[k1, :7]
        p0, _ = kin.fk(q0); p1, R1 = kin.fk(q1); pp, Rp = kin.fk(arm[-1])
        cv = calls[s].get("conversion") or {}
        qp = (cv.get("ee_action_qp") or {}).get("used")
        print(f"{s:>3} {str((calls[s].get('grasp') or {}).get('state')):>9} {str(qp):>5} {1000*cv.get('ik_pos_err_m_max', float('nan')):7.1f} | "
              f"err {1000*np.linalg.norm(pp - p1):6.1f}  act {1000*np.linalg.norm(p1 - p0):6.1f}  pred {1000*np.linalg.norm(pp - p0):6.1f}  dq {np.abs(arm[-1] - q1).max():.3f}  n={n} rows={len(arm)}  dvec(pred-act) {1000*(pp-p1)}")
