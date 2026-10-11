"""Diagnose the palm rows that violate in the PC solution: which surface is the nearest (target / table / other / field layers)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, argparse, numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T45/V/p'))
import vt45 as V
from AG3S.trajopt.experiments import ta_offline as TA

cases = [("C1a_ep0_s2", 13), ("C1a_ep104_s1041", 15), ("C1a_ep123_s1231", 11), ("C1a_ep12_s121", 13), ("C1a_ep12_s121", 15),
         ("C1a_ep30_s302", 14), ("C1a_ep3_s31", 28), ("C1a_ep54_s542", 20)]
ns = argparse.Namespace(hm=False, no_plane_rows=False)
out = []
cache = {}
for name, seq in cases:
    entry = V.load_entry(_ag3s_workspace_path('outputs/verify/T45/V/index_c1a.json'), name)
    if name not in cache:
        cache[name] = V.Side(name, entry, ns, V.PC_ARGS)
    sd = cache[name]
    run = sd.run
    inp, why = run.inputs(seq)
    scene = sd.scene(inp)
    npz = np.load(f"${AG3S_PROJECT_ROOT}/outputs/verify/T45/V/raw/cmp/{name}.cmp.npz")
    own = npz[f"s{seq}_PC_own"]
    traj = run.layout.chunk_to_trajectory(own)[:, :V.K]
    cfg = run.base_cfg
    lin = run.meter(V.K)
    s = inp["summary"]
    lin.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], V.K))
    lin.set_attached(getattr(inp["scene"], "attached_points", None), getattr(inp["scene"], "attached_parent_link", None))
    centres = lin.sphere_states(traj, inp["q_now"])[0]
    to = np.asarray(lin._esdf_clearance(centres, scene)[..., 0])
    S = len(run.names)
    geom = V.geom_of(inp)
    pi = V.group_indices(run)["palm"]
    r = np.asarray(lin.robot_radii)[pi]
    P = centres[:, pi]
    flat = P.reshape(-1, 3); rr = np.tile(r[None], (V.K, 1)).reshape(-1)
    d_main = np.asarray(scene.esdf.distance(flat)) - rr
    d_free = np.asarray(scene.esdf.target_free_distance(flat)) - rr
    gt = geom["target_tree"].query(flat)[0] - rr
    go = geom["other_tree"].query(flat)[0] - rr
    pl = inp["plane"]; gtab = flat @ pl[:3] - pl[3] - rr
    tov = to[:, pi].reshape(-1)
    bad = tov < 0
    rec = {"run": name, "seq": seq, "n_palm_rows_to_lt0": int(bad.sum()), "of": int(len(bad)), "to_min_mm": float(tov.min() * 1e3),
           "rows": []}
    for i in np.argsort(tov)[:6]:
        rec["rows"].append({"step": int(i // len(pi)), "sphere": int(i % len(pi)), "r_mm": float(rr[i] * 1e3), "to_mm": float(tov[i] * 1e3),
                            "d_free_minus_r_mm": float(d_free[i] * 1e3), "d_main_minus_r_mm": float(d_main[i] * 1e3),
                            "gap_target_pts_mm": float(gt[i] * 1e3), "gap_other_pts_mm": float(go[i] * 1e3), "gap_table_mm": float(gtab[i] * 1e3),
                            "centre": flat[i].tolist()})
    rec["bad_rows_summary"] = None if not bad.any() else {
        "gap_target_pts_mm": [float(x * 1e3) for x in (gt[bad].min(), np.median(gt[bad]), gt[bad].max())],
        "gap_other_pts_mm": [float(x * 1e3) for x in (go[bad].min(), np.median(go[bad]), go[bad].max())],
        "gap_table_mm": [float(x * 1e3) for x in (gtab[bad].min(), np.median(gtab[bad]), gtab[bad].max())],
        "d_free_minus_r_mm": [float(x * 1e3) for x in (d_free[bad].min(), np.median(d_free[bad]), d_free[bad].max())],
        "d_main_minus_r_mm": [float(x * 1e3) for x in (d_main[bad].min(), np.median(d_main[bad]), d_main[bad].max())],
        "radii_mm": [float(x * 1e3) for x in (rr[bad].min(), np.median(rr[bad]), rr[bad].max())]}
    rec["esdf_stats_exclusion"] = (s.get("target_exclusion") or {})
    out.append(rec)
    inp["z"].close()
    print(name, seq, rec["n_palm_rows_to_lt0"], rec["to_min_mm"], rec["bad_rows_summary"], flush=True)
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T45/V/diag_hold.json'), "w"), indent=1, default=str)
