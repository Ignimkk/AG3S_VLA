import json, numpy as np, glob, sys
from AG3S.adapters.panda import model as pm, action as pa
from AG3S.ag3s.robot_models.urdf_sphere_chain import UrdfSphereChain
kin = pa.PandaKinematics(UrdfSphereChain(pm.load_panda_urdf(), pm.PANDA_ARM_JOINTS))
np.set_printoptions(precision=3, suppress=True, linewidth=220)
import os
B = os.environ.get("AUDIT_BUNDLE", "3_raw_i19").rstrip("/") + "/"   # the folder with the raw episodes
d = sys.argv[1]; seqs = [int(x) for x in sys.argv[2:]]
calls = [json.loads(l) for l in open(glob.glob(B + d + "/*_calls.jsonl")[0], encoding="utf-8")]
g = np.load(glob.glob(B + d + "/*_geom.npz")[0], allow_pickle=True)
rs = g["robot_state/head"]; oi = list(g["obs_index"]); Twb = g["T_world_base"]
for s in seqs:
    k = oi.index(calls[s]["obs_index"]); q = rs[k, :7]
    p, R = kin.fk(q)
    J = kin.jacobian(q); sv = np.linalg.svd(J, compute_uv=False)
    print(f"== seq {s} q={q} p={p} ee_z={R[:,2]}  J singular values {sv}")
    # actual EE step just before (world frame ~ base frame? print T_world_base R)
    if s == seqs[0]: print("R_world_base=\n", Twb[k][:3,:3])
    for name, a in (("+z", [0,0,1,0,0,0]), ("-z", [0,0,-1,0,0,0]), ("+x", [1,0,0,0,0,0]), ("-x", [-1,0,0,0,0,0]), ("+y", [0,1,0,0,0,0]), ("-y", [0,-1,0,0,0,0]), ("zero", [0]*6)):
        A = np.tile(np.array(a + [-1.0]), (10, 1))
        conv = pa.delta_chunk_to_joint_chunk(kin, A, q, Twb[k], prev_step_world=None)
        it = [r["iters"] for r in conv.ik]; pe = [round(1000*r["pos_err_m"], 2) for r in conv.ik]; re = [round(r["rot_err_rad"], 4) for r in conv.ik]
        print(f"   {name:>4}: iters {it}  pos_err_mm {pe}  rot_err {re}")
