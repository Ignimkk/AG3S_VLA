"""T44 V-I0 (verifier): contact event -> server chunk -> removed seeds / permission rows, V11AB H ep1807 s18071.
Pre-T44 record: no target_free_removed_ijk / target_permission keys. Proxy for 'removed' = fine voxels with
esdf_free_distance - esdf_distance > voxel (same grid, same origin) — a field difference, NOT the seed list."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, ast, numpy as np, pathlib
R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2/vi0'))
rep = json.load(open(R / "V11AB_H_V11AB_ep1807_s18071.json"))
cdir = pathlib.Path(_ag3s_workspace_path('')) / rep["server_chunk_alignment"]["block"]["chunk_dir"]
off = rep["server_chunk_alignment"]["block"]["offset"]
ev = rep["contact_chunks"]
first = {c: (v[0] if v else None) for c, v in ev.items()}
k0 = first["hand-target"]["server_chunk_k"]
rows = []
for k in range(k0 - 3, k0 + 2):
    z = np.load(cdir / f"chunk_{k + off:05d}.npz", allow_pickle=True)
    s = json.loads(str(z["summary_json"]))
    tf = s["esdf_stats"].get("target_free")
    tf = ast.literal_eval(tf) if isinstance(tf, str) else tf
    names = np.asarray(s["sphere_link_names"])
    auth = s.get("exclusion", {}).get("authorized_links") or []
    is_auth = np.isin(names, auth)
    cl = z["clearance"].astype(np.float64)  # (steps, spheres) mm? m?
    dm, df = z["esdf_distance"].astype(np.float32), z["esdf_free_distance"].astype(np.float32)
    same = np.allclose(z["esdf_origin"], z["esdf_free_origin"]) and float(z["esdf_voxel_size"]) == float(z["esdf_free_voxel_size"])
    vs = float(z["esdf_free_voxel_size"])
    carved = int(((df - dm) > vs).sum()) if same else None
    carved_surf = int(((np.abs(dm) <= vs) & ((df - dm) > vs)).sum()) if same else None
    cls_here = {c: [e for e in v if e["server_chunk_k"] == k] for c, v in ev.items()}
    rows.append({
        "server_chunk_k": k, "file": f"chunk_{k + off:05d}.npz", "seq_local": k + 1, "t_step_npz": int(z["t_step"]),
        "contacts_executing_this_chunk": {c: {"t_first": e[0]["t_first"], "t_last": e[0]["t_last"], "min_dist_mm": round(e[0]["min_dist_m"] * 1e3, 2),
                                              "max_force_N": round(e[0]["max_force_N"], 1)} for c, e in cls_here.items() if e},
        "manipulated": {"id": (s.get("manipulated") or {}).get("id"), "state": (s.get("manipulated") or {}).get("state")},
        "exclusion_active": s.get("exclusion", {}).get("active"), "grasp_active": s.get("exclusion", {}).get("grasp_active"),
        "authorized_links": auth,
        "n_rows_spheres": int(len(names)), "n_rows_authorized": int(is_auth.sum()),
        "target_free_mode": (tf or {}).get("mode"), "ball_radius_mm": ((tf or {}).get("ball") or {}).get("radius_mm"),
        "n_ball_seeds_excluded": (tf or {}).get("n_ball_seeds_excluded"),
        "n_target_points": (tf or {}).get("n_target_points"),
        "carved_voxels_proxy_dfree_minus_dmain_gt_vs": carved,
        "carved_surface_voxels_proxy_absdmain_le_vs": carved_surf,
        "clearance_min_authorized": round(float(cl[:, is_auth].min()), 3) if is_auth.any() else None,
        "clearance_min_nonauthorized": round(float(cl[:, ~is_auth].min()), 3) if (~is_auth).any() else None,
        "has_new_T44_keys": {k_: (k_ in z.files) for k_ in ("target_free_removed_ijk", "target_free_nontarget_kept_ijk", "target_free_ambiguous_ijk")},
        "summary_has_target_permission": "target_permission" in s, "summary_has_target_exclusion": "target_exclusion" in s,
        "verdict_reasons": (s.get("verdict") or {}).get("reasons"),
    })
out = {"definitions": __doc__, "run": rep["run_dir"], "first_contact_per_class": first, "seq_rule": rep.get("seq_rule"),
       "clearance_unit": "m (npz clearance, steps x spheres)", "rows": rows}
json.dump(out, open(R / "vi0_table.json", "w"), indent=1, default=str)
for r in rows: print(json.dumps(r, default=str)[:900])
