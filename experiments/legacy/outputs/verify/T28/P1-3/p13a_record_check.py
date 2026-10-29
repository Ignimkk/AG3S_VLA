"""Cross-check with live records: recorded server sphere_centres (base frame, step 0) of finger links vs MuJoCo
finger body/collision-mesh positions at the paired planning-row qpos (T16, T14; chunk k <-> seq k+1)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-3'))
from p13a_spheres import body_mesh_world, XML
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
out = {}
for run in ("t16", "t14"):
    R = pathlib.Path(f"{_ag3s_workspace_path()}/outputs/live_test/20260928_{run}")
    rows = {json.loads(l)["seq"]: json.loads(l) for l in open(R / "execute_ep1807/frames.jsonl") if '"planning"' in l}
    res = []
    for k in (0, 9, 12, 16, 22):
        z = np.load(R / f"server_constraints/run_0000/chunk_{k:05d}.npz", allow_pickle=True)
        s = json.loads(str(z["summary_json"])); names = np.array(s["sphere_link_names"])
        pr = rows[k + 1]; d.qpos[:] = pr["qpos"]; mujoco.mj_forward(m, d)
        Rb = d.xmat[bid("base")].reshape(3, 3); pb = d.xpos[bid("base")]
        C = z["sphere_centres"].astype(float) @ Rb.T + pb
        e = {"chunk": k, "t": 8 * k, "grip_l_qpos": pr["qpos"][27]}
        for L in ("ee_finger_l1", "ee_finger_l2", "ee_left"):
            MJ = {"ee_left": "EE_BODY_L"}.get(L, L)
            V = body_mesh_world(m, d, MJ, True)
            c = C[names == L].mean(0)
            e[L] = {"recorded_sphere_centroid_to_mujoco_collision_mesh_centroid_mm": float(1000 * np.linalg.norm(c - V.mean(0))),
                    "n": int((names == L).sum()), "r_mm": sorted({round(1000 * float(x), 2) for x in z["sphere_radii"][names == L]})}
        l1 = C[names == "ee_finger_l1"].mean(0); l2 = C[names == "ee_finger_l2"].mean(0)
        e["recorded_l1_l2_sphere_centroid_separation_mm"] = float(1000 * np.linalg.norm(l1 - l2))
        e["mujoco_l1_l2_collision_mesh_centroid_separation_mm"] = float(1000 * np.linalg.norm(body_mesh_world(m, d, "ee_finger_l1", True).mean(0) - body_mesh_world(m, d, "ee_finger_l2", True).mean(0)))
        res.append(e); print(run, json.dumps(e))
    out[run] = res
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T28/P1-3/p13a_record_check.json'), "w"), indent=1)
