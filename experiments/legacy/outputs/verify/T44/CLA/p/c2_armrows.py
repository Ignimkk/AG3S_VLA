"""T44dA verifier: C2 chunks whose worst TO row (max_violation_pair) is an arm link (not palm/finger): GT geometry of that row's sphere.
For each: link, recorded clearance (d - r - margin), tier, obstacle label, sphere point; GT signed gap (convex hull of the fruit collision meshes,
planning-row object_poses) of the sphere (minus its radius) to the target and to the nearest other fruit; height of the sphere bottom above the
support plane; recomputed main-field distance d_main - r at that point (field.py, recorded grids). Also counts, over ALL C2 chunks and
reference rows 0..15, the unauthorized arm-link rows with d_main - r < 10 mm (obstacle margin) and their nearest-object class by GT gap.
-> CLA/c2_armrows.json   (run from ${AG3S_PROJECT_ROOT}-t44cl)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import os, sys, json, glob, pathlib
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/CLA/p'))
import numpy as np
import fkmodel as F, field as FD
from extract_chunks import hull_sd, FRUITS
CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL')); C = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CLA'))
sc, qm = F.model("arms"); links = np.asarray(qm.sphere_link_names); rad = np.asarray(qm.radii)
armrow = np.array(["arm" in l for l in links])
worst = []; agg = {"n_chunks": 0, "rows_lt10": 0, "by_class": {}, "by_link": {}}
for f in sorted(glob.glob(str(CL / "analysis" / "C2_*.json"))):
    a = json.load(open(f)); name = a["name"]; side = a["used_arm"]; tgt = a["target_fruit"]; others = [b for b in FRUITS if b != tgt]
    rows = [json.loads(l) for l in open(pathlib.Path(_ag3s_workspace_path('')) / a["run_dir"] / "frames.jsonl")]
    P = {r["seq"]: r for r in rows if r["kind"] == "planning"}
    for c in a["server_chunks"]:
        z = np.load(c["server_file"], allow_pickle=True); s = json.loads(str(z["summary_json"])); p = P.get(c["seq"])
        if p is None: continue
        mvp = (s.get("to") or {}).get("metrics", {}).get("max_violation_pair"); mvp = eval(mvp) if isinstance(mvp, str) else mvp
        tf = s.get("esdf_stats", {}).get("target_free"); tf = eval(tf) if isinstance(tf, str) else (tf or {})
        sz = ((tf or {}).get("ball") or {}).get("support_plane_z_m"); op = p["object_poses"]
        # rows near the hand only: chunks whose reference brings the active fingers within 120 mm of the target
        Q = F.chunk_q(sc, qm, p["qpos"], np.asarray(z["reference_chunk"], np.float64)[:16]); pr = F.params_from_qpos(sc, qm, p["qpos"])
        S = np.stack([F.spheres(qm, q, pr) for q in Q])
        side_arm = np.array([f"link_{side}_arm" in l for l in links])
        X = S[:, side_arm].reshape(-1, 3); r = np.tile(rad[side_arm], 16); L = np.tile(links[side_arm], 16)
        dm = np.asarray(FD.build(z, coarse=False).distance(X)) - r
        near = dm < 0.010
        agg["n_chunks"] += 1
        if near.any():
            gT = hull_sd(op[tgt], tgt, X[near]) - r[near]; gO = np.min([hull_sd(op[b], b, X[near]) - r[near] for b in others], axis=0)
            hb = X[near, 2] - r[near] - (sz if sz is not None else np.nan)
            cls = np.where((gT < gO) & (gT < hb), "target", np.where(gO < hb, "fruit_other", "table/other"))
            for k in np.unique(cls): agg["by_class"][str(k)] = agg["by_class"].get(str(k), 0) + int((cls == k).sum())
            for k in np.unique(L[near]): agg["by_link"][str(k)] = agg["by_link"].get(str(k), 0) + int((L[near] == k).sum())
            agg["rows_lt10"] += int(near.sum())
        if mvp and "arm" in str(mvp.get("link")):
            pt = np.asarray(mvp["point_m"]); q = int(mvp["query"]); rr = float(rad[q]) if q < len(rad) else np.nan
            worst.append({"name": name, "seq": c["seq"], "hold": bool(c.get("hold")), "link": mvp["link"], "clearance_mm": 1e3 * mvp["clearance_m"], "tier": mvp.get("tier"),
                          "obstacle": mvp.get("obstacle"), "radius_mm": 1e3 * rr, "d_main_minus_r_mm": float(1e3 * (FD.build(z, coarse=False).distance(pt[None])[0] - rr)),
                          "gt_gap_target_mm": float(1e3 * (hull_sd(op[tgt], tgt, pt[None])[0] - rr)),
                          "gt_gap_other_mm": float(1e3 * min(hull_sd(op[b], b, pt[None])[0] - rr for b in others)),
                          "h_bottom_mm": float(1e3 * (pt[2] - rr - sz)) if sz is not None else None})
json.dump({"worst_arm_rows": worst, "arm_rows_lt10mm": agg}, open(C / "c2_armrows.json", "w"), indent=1)
print(json.dumps(agg)); [print(w) for w in worst]
