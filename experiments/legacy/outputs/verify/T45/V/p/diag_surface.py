"""For the worst palm rows of the PC solution: where is the nearest field surface point and which cloud class is it."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, argparse, numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T45/V/p'))
import vt45 as V
from scipy.spatial import cKDTree
cases = [("C1a_ep3_s31", 28), ("C1a_ep104_s1041", 15), ("C1a_ep30_s302", 14), ("C1a_ep54_s542", 20)]
ns = argparse.Namespace(hm=False, no_plane_rows=False)
out = []
for name, seq in cases:
    entry = V.load_entry(_ag3s_workspace_path('outputs/verify/T45/V/index_c1a.json'), name)
    sd = V.Side(name, entry, ns, V.PC_ARGS); run = sd.run
    inp, _ = run.inputs(seq); scene = sd.scene(inp)
    z = inp["z"]
    own = np.load(f"{_ag3s_workspace_path()}/outputs/verify/T45/V/raw/cmp/{name}.cmp.npz")[f"s{seq}_PC_own"]
    traj = run.layout.chunk_to_trajectory(own)[:, :V.K]
    lin = run.meter(V.K); lin.set_joint_parameter_path(run.finger_path(inp["summary"], inp["ref_chunk"], V.K))
    centres = lin.sphere_states(traj, inp["q_now"])[0]
    to = np.asarray(lin._esdf_clearance(centres, scene)[..., 0])
    pi = V.group_indices(run)["palm"]
    t = to[:, pi]; k, j = np.unravel_index(np.argmin(t), t.shape)
    p = centres[k, pi[j]][None]
    f = scene.esdf
    d = float(np.asarray(f.distance(p))[0]); g = np.asarray(f.gradient(p))[0]
    q = p[0] - d * g / max(np.linalg.norm(g), 1e-9)
    pts = np.asarray(z["filtered_cloud"], float); sm = np.asarray(z["support_mask"], bool)
    ti = np.zeros(len(pts), bool); ti[np.asarray(z["target_point_indices"], int)] = True
    raw = np.asarray(z["raw_cloud"], float) if "raw_cloud" in z.files else None
    cls = np.where(ti, "target", np.where(sm, "support", "other"))
    tree = cKDTree(pts); dd, ii = tree.query(q, k=5)
    rec = {"run": name, "seq": seq, "step": int(k), "palm_sphere": int(j), "centre": p[0].tolist(), "d_main_mm": d * 1e3,
           "nearest_surface_pt": q.tolist(), "nearest_cloud_classes": [(str(cls[i]), float(x * 1e3)) for x, i in zip(dd, ii)],
           "target_centroid": np.asarray(z["target_centroid"]).tolist(), "plane": None if inp["plane"] is None else inp["plane"].tolist(),
           "surface_z_minus_plane_mm": None if inp["plane"] is None else float((q @ inp["plane"][:3] - inp["plane"][3]) * 1e3),
           "grasp_state": inp["summary"]["grasp"]["state"]}
    if raw is not None:
        rt = cKDTree(raw); rd, _ = rt.query(q); rec["nearest_raw_cloud_mm"] = float(rd * 1e3)
    for nm in ("esdf_free_distance",):
        pass
    out.append(rec); print(json.dumps(rec), flush=True)
    z.close()
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T45/V/diag_surface.json'), "w"), indent=1)
