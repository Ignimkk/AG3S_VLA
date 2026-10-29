"""frame alignment + URDF<->MJCF finger pairing by position (not by name)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np, sys
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T45/p'))
import common as C
r = [json.loads(l) for l in open(_ag3s_workspace_path('outputs/verify/T44/CL/runs/C1a/R/C1a_ep3_s31/frames.jsonl'))]
qpos = np.asarray([x for x in r if x["kind"] == "control" and x["t_step"] == 224][0]["qpos"])
qm = C.model("gripper")
sc = C.scene()
print("joint_names", qm.joint_names); print("param", qm.param_joint_names)
cen, rad = C.spheres_world(qm, qpos); links = np.asarray(qm.sphere_link_names)
C.forward(qpos)
out = {}
for L in sorted(set(links)):
    T = qm.link_pose(C.q_from_qpos(sc, qm, qpos), L, C.params_from_qpos(sc, qm, qpos))
    best = []
    for b in ["EE_BODY_R", "EE_BODY_L", "ee_finger_r1", "ee_finger_r2", "ee_finger_l1", "ee_finger_l2"]:
        bid = C.BODY(b)
        hv = np.vstack([C.geom_world_verts(g) for g in C.coll_geoms(bid)])
        sel = cen[links == L]
        d = C.gap(sel, hv, np.zeros(len(hv)))  # centre -> nearest hull vertex
        best.append((float(np.median(d)), b, float(np.linalg.norm(T[:3, 3] - C.D.xpos[bid])),
                     float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ C.D.xmat[bid].reshape(3, 3)) - 1) / 2, -1, 1))))))
    best.sort()
    out[L] = {"n": int((links == L).sum()), "nearest_body_by_sphere_position": best[0][1], "median_centre_to_vertex_mm": best[0][0] * 1e3,
              "frame_origin_offset_mm": best[0][2] * 1e3, "frame_rot_deg": best[0][3], "second": [best[1][1], best[1][0] * 1e3]}
    print(L, out[L])
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T45/align.json'), "w"), indent=1)
