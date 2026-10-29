"""T43 Y — what the worst TO row of every HOLD chunk is near, in the TRUE scene (verifier, CPU only).

usage: hold_rows_y.py OUT_JSON KEY [KEY ...]

Per HOLD chunk (raw/server_chunks.json hold flag): worst row (to.metrics.max_violation_pair) point_m (base frame; = query sphere centre at
refined row `step`), query radius r (robot: sphere_radii[q]; held: the chunk's body-cover radius), MuJoCo base -> world at the planning-row qpos.
  esdf_mm            recorded ESDF at point_m (tier of the row; esdf_held_y.py sampling), gap = esdf - r, implied margin = gap - TO clearance.
  apple_surface_mm   |point - apple centre| - r - 33.3 mm (apple collision mesh half-extent 31.1 / 33.3 / 33.3 mm; planning-row apple pose) — approximate.
  crate_sd_mm        point-to-OBB signed distance to the nearest crate collision box - r (planning-row crate pose).
  table_mm           point z - r - 0.82 m (table top plane z, T43 P2 scenes).
  nearest_true       robot rows: argmin of (apple_surface_mm, crate_sd_mm, table_mm); held rows: argmin of (crate_sd_mm, table_mm) (the held
                     sphere encloses the apple while it is in the hand, so the apple is not a candidate for held rows)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, json, pathlib, sys, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/Y/p'))
import hurdle_metrics as HM  # noqa: E402
from esdf_held_y import samp  # noqa: E402
from server_y import obb_sd  # noqa: E402

SC = _ag3s_workspace_path('outputs/verify/T43/Y/raw/server_chunks.json')
AN = _ag3s_workspace_path('outputs/verify/T43/W/analysis/analysis.json')
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x


def main():
    S = json.load(open(SC))["runs"]; A = json.load(open(AN))["runs"]; res = {"definitions": __doc__, "runs": {}}
    for key in sys.argv[2:]:
        r = S[key]; HM.set_arm(r["side"]); HM._M.clear(); M = HM.model(); m, d = M["m"], M["d"]
        base = M["bid"]["base"]; cb = M["bid"]["crate"]
        crate_g = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == cb and (m.geom_contype[g] or m.geom_conaffinity[g])]
        plan = {p["seq"]: p for p in (json.loads(l) for l in open(pathlib.Path(A[key]["run_dir"]) / "frames.jsonl")) if p["kind"] == "planning"}
        first = pathlib.Path(r["server_segment"]["first"]); i0 = int(first.stem.split("_")[1]); rows = []
        for n, c in enumerate(r["chunks"]):
            if not c["hold"]:
                continue
            z = np.load(first.parent / f"chunk_{i0 + n:05d}.npz", allow_pickle=True); s = json.loads(str(z["summary_json"]))
            mv = lit(s["to"]["metrics"]["max_violation_pair"]); q = int(mv["query"]); nr = int(z["sphere_radii"].shape[0])
            rad = float(z["sphere_radii"][q]) if q < nr else 1e-3 * float(c["held_query"]["radii_mm"][q - nr])
            p = plan[c["seq"]]; d.qpos[:] = p["qpos"]; mujoco.mj_kinematics(m, d)
            w = d.xpos[base] + d.xmat[base].reshape(3, 3) @ np.asarray(mv["point_m"], float)
            tier = mv.get("tier")
            e = (samp(z["esdf_distance"], z["esdf_origin"], z["esdf_voxel_size"], mv["point_m"]) if tier == "fine"
                 else samp(z["esdf_coarse_distance"], z["esdf_coarse_origin"], z["esdf_coarse_voxel_size"], mv["point_m"]))
            ap = np.asarray(p["object_poses"]["apple"]["pos"], float)
            dist = {"apple_surface_mm": 1e3 * (np.linalg.norm(w - ap) - rad - 0.0333),
                    "crate_sd_mm": 1e3 * (min(obb_sd(w, d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]) for g in crate_g) - rad),
                    "table_mm": 1e3 * (w[2] - rad - 0.82)}
            rows.append({"t": c["t"], "link": mv.get("link"), "query": q, "kind": "held" if q >= nr else "robot", "radius_mm": 1e3 * rad, "obstacle": mv.get("obstacle"),
                         "tier": tier, "to_clearance_mm": 1e3 * mv["clearance_m"], "esdf_mm": 1e3 * e, "esdf_gap_mm": 1e3 * (e - rad),
                         "implied_margin_mm": 1e3 * (e - rad - mv["clearance_m"]), **dist, "nearest_true": min(dist, key=dist.get) if q < nr else min(("crate_sd_mm", "table_mm"), key=dist.get)})
        res["runs"][key] = rows
        from collections import Counter
        print(key, len(rows), Counter((x["kind"], x["obstacle"], x["nearest_true"]) for x in rows), flush=True)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
