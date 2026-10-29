"""T43 Y — the binding held row of attached chunks: recorded ESDF value at the held sphere centre vs the TRUE crate distance.
(verifier, CPU only)

usage: esdf_held_y.py OUT_JSON KEY [KEY ...]     (KEY as in raw/server_chunks.json; reads that file + the chunk npz)

Per attached chunk whose worst TO row is a held query sphere (link 'attached:*'):
  point_m      = worst row point (base frame) == held sphere centre at refined row `step` (server_y.py check, 0.0 mm).
  d_fine_mm    = recorded esdf_distance (fine tier, voxel esdf_voxel_size; esdf_origin = centre of voxel (0,0,0),
                 frame_record.py:157) trilinear at point_m; NaN if outside the fine grid. d_coarse_mm likewise (esdf_coarse_*).
  radius_mm    = that held sphere's radius (body-cover set of the chunk).
  esdf_gap_mm  = d(tier of the worst row) - radius;  implied_margin_mm = esdf_gap_mm - TO clearance_mm (linearize: clearance = d - r - margin).
  true_sd_mm   = server_y.py held_query.crate_sd_mm_worst_step[that sphere] (sphere vs MuJoCo crate boxes, < 0 = overlap)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
from scipy.ndimage import map_coordinates

SC = _ag3s_workspace_path('outputs/verify/T43/Y/raw/server_chunks.json')


def samp(grid, origin, vox, p):
    idx = (np.asarray(p, float) - np.asarray(origin, float)) / float(vox)
    if np.any(idx < 0) or np.any(idx > np.asarray(grid.shape) - 1):
        return float("nan")
    return float(map_coordinates(np.asarray(grid, np.float32), idx.reshape(3, 1), order=1)[0])


def main():
    S = json.load(open(SC))["runs"]; outp = pathlib.Path(sys.argv[1]); res = {"definitions": __doc__, "runs": {}}
    for key in sys.argv[2:]:
        r = S[key]; first = pathlib.Path(r["server_segment"]["first"]); i0 = int(first.stem.split("_")[1]); rows = []
        for n, c in enumerate(r["chunks"]):
            w = c["worst"]
            if not (c["attached"] and isinstance(w.get("link"), str) and w["link"].startswith("attached:")):
                continue
            z = np.load(first.parent / f"chunk_{i0 + n:05d}.npz", allow_pickle=True); assert int(z["t_step"]) == c["seq"]
            s = json.loads(str(z["summary_json"])); import ast
            mv = s["to"]["metrics"]["max_violation_pair"]; mv = ast.literal_eval(mv) if isinstance(mv, str) else mv
            qi = int(w["query"]) - int(z["sphere_radii"].shape[0])
            rad = (c.get("held_query") or {}).get("radii_mm", [None] * (qi + 1))[qi]
            df = 1e3 * samp(z["esdf_distance"], z["esdf_origin"], z["esdf_voxel_size"], mv["point_m"])
            dc = 1e3 * samp(z["esdf_coarse_distance"], z["esdf_coarse_origin"], z["esdf_coarse_voxel_size"], mv["point_m"])
            d = df if w.get("tier") == "fine" else dc
            gap = None if rad is None else d - rad
            sdk = (c.get("held_query") or {}).get("crate_sd_mm_worst_step")
            rows.append({"t": c["t"], "seq": c["seq"], "hold": c["hold"], "sphere": qi, "radius_mm": rad, "tier": w.get("tier"), "obstacle": w.get("obstacle"),
                         "to_clearance_mm": w["clearance_mm"], "d_fine_mm": df, "d_coarse_mm": dc, "esdf_gap_mm": gap,
                         "implied_margin_mm": None if gap is None else gap - w["clearance_mm"],
                         "true_sd_mm": None if not sdk else sdk[qi], "true_sd_min_all_spheres_mm": None if not sdk else min(sdk)})
        res["runs"][key] = rows
        h = [x for x in rows if x["hold"]]
        print(key, "held-worst chunks", len(rows), "HOLD", len(h), "implied margin", sorted({round(x["implied_margin_mm"], 1) for x in rows if x["implied_margin_mm"] == x["implied_margin_mm"]})[:8], flush=True)
    outp.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
