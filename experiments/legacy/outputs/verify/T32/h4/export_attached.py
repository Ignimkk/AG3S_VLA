"""T32 H4 — export the attached (held-apple) query points in the base frame at chunk capture state.

usage (cwd = outputs/verify/T32/h4/runroot, snapshot benchmark):
  MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}/outputs/verify/T31diag/snap ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python \
      ../export_attached.py E3b_ep1807_r1 16 20 30 36
Same reconstruction as T31diag D3 (reconstruct_attached: attach_from_target on the recorded target_points at
the attach chunk, parent ee_finger_l1). Base-frame points at chunk k = T_base_parent(q of planning row seq k+1,
recorded finger joints of chunk k) @ parent-frame points.  Only written when the recorded chunk has
esdf_stats.n_attached_points > 0 (the server passed attached points to the builder at that chunk).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T31diag/d4'))
import fig_common as FC
from AG3S.ag3s.constraints.attached import _link_pose_numeric
import AG3S.trajopt.experiments.fixed_input_ablation as T

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T32/h4/real'))
run = sys.argv[1]; chunks = [int(x) for x in sys.argv[2:]]
R = FC.Run(run)
robot = R.sh["robot"]
res = {}
for k in chunks:
    t = 8 * k
    c = R.by_t[t]
    z = np.load(FC.T30 / f"{R.cond}_rec" / c["server_chunk"], allow_pickle=False)
    s = json.loads(str(z["summary_json"]))
    n_att = int((s.get("esdf_stats") or {}).get("n_attached_points", 0) or 0)
    row = R.rows[c["seq"]]
    if n_att == 0 or R.att is None:
        res[k] = {"n_attached_points_recorded": n_att, "written": False}
        continue
    joints = (s.get("finger_joints") or {}).get("joints") or {}
    robot.set_joint_parameters({kk: v for kk, v in joints.items() if kk in robot.param_joint_names})
    q = T.q_now_from_qpos(row["qpos"], R.sh["qadr"])
    Tbp = _link_pose_numeric(robot, q, "ee_finger_l1")
    robot.set_joint_parameters(None)
    P = np.asarray(R.att.points, np.float64)
    Pb = P @ Tbp[:3, :3].T + Tbp[:3, 3]
    np.save(OUT / f"attached_{run}_chunk{k:05d}.npy", Pb)
    res[k] = {"n_attached_points_recorded": n_att, "n_reconstructed": int(len(Pb)), "written": True,
              "centroid_base_m": Pb.mean(axis=0).tolist(), "server_chunk": c["server_chunk"], "seq": c["seq"]}
(OUT / f"attached_{run}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
