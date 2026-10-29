"""T40 A: carry-height statistics for all 44 runs of the 22 both-success pairs (verifier, CPU).
apple bottom z = lowest vertex of apple_geom (mesh) in world, via support(-z); height = z - table_surface_z.
Window [lift_t, place_t] (search.window). Segments inside the window:
  near_start : apple xy within 70 mm of its t = 0 xy
  over_crate : apple xy inside the crate collision AABB in xy (walls + handles, t = 0)
  traverse   : the rest
Per run: min / median / max apple-bottom height per segment, and the crate top height (max z of crate geoms, t = 0).
(extract_paths.py's apple_bottom_above_table used an ellipsoid formula on the mesh geom size; this file supersedes it.)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
import numpy as np, mujoco
import t40a_lib as L, search as S

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/A/raw'))
allp = json.load(open(RAW / "carry_paths_22pairs.json"))["pairs"]
out = {"definitions": __doc__, "runs": {}}
for ent in allp:
    tag = f"{ent['episode']}_{ent['seed']}"
    for c in ("E0", "E3b"):
        R = L.load_run(ent["runs"][c]["run_dir"])
        w = S.window(R, ent["runs"][c]["grasp_t"], ent["runs"][c]["place_t"])
        L.set_q(R["q0"])
        lo = np.full(3, 9.0); hi = np.full(3, -9.0)
        for g in L.CR_CRATE:
            sup = L.gjk.make_support(L.m, L.d, int(g))
            for ax in range(3):
                e = np.zeros(3); e[ax] = 1
                hi[ax] = max(hi[ax], sup(e)[ax]); lo[ax] = min(lo[ax], sup(-e)[ax])
        a0 = R["q0"][38:40]
        seg = {"near_start": [], "over_crate": [], "traverse": []}
        for t, q in zip(R["t"], R["q"]):
            if not (w[0] <= t <= w[2]):
                continue
            L.set_q(q)
            zb = L.gjk.make_support(L.m, L.d, int(L.B_APPLE[0]))(np.array([0, 0, -1.0]))[2] - L.TABLE_Z
            xy = q[38:40]
            if np.linalg.norm(xy - a0) < 0.07:
                seg["near_start"].append(zb)
            elif lo[0] <= xy[0] <= hi[0] and lo[1] <= xy[1] <= hi[1]:
                seg["over_crate"].append(zb)
            else:
                seg["traverse"].append(zb)
        out["runs"][f"{c}_{tag}"] = {"window": dict(zip(("lift_t", "grasp_t", "place_t"), w)), "crate_top_above_table_m": hi[2] - L.TABLE_Z,
                                     **{k: (None if not v else {"n": len(v), "min_m": float(min(v)), "median_m": float(np.median(v)), "max_m": float(max(v))}) for k, v in seg.items()}}
        r = out["runs"][f"{c}_{tag}"]
        print(c, tag, {k: (r[k] and (r[k]["n"], round(1e3 * r[k]["min_m"]), round(1e3 * r[k]["max_m"]))) for k in seg}, round(1e3 * r["crate_top_above_table_m"]), flush=True)
tr = [r["traverse"]["min_m"] for r in out["runs"].values() if r["traverse"]]
out["summary"] = {"n_runs": len(out["runs"]), "n_runs_with_traverse_rows": len(tr),
                  "traverse_min_apple_bottom_m": {"min": min(tr), "median": float(np.median(tr)), "max": max(tr)}}
print(out["summary"])
(RAW / "carry_stats.json").write_text(json.dumps(out, indent=1))
