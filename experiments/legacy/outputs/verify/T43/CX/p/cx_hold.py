"""T43 CX — every HOLD chunk of the V11AB / V11ABC H runs: binding row, recorded ESDF, true-scene distances, cloud points inside the sphere,
and what the SQP returned (verifier, CPU only, numbers only).
usage: T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v11 cx_hold.py OUT_JSON

HOLD chunk = planning row with gate.action 'hold' (V11/analysis/analysis_<ARM>.json hold_t; server chunk by the run's server_segment).
Per HOLD chunk, worst TO row = to.metrics.max_violation_pair (point_m = query sphere centre at refined row `step`, base frame):
  radius_mm      robot query: sphere_radii[q]; held query (q >= n_robot): grasp.attached.held body_cover spheres if present else held spheres [q - n_robot].
  esdf_mm        recorded ESDF at point_m, tier of the row (trilinear; esdf_origin = centre of voxel 0), gap = esdf - r, implied_margin = gap - TO clearance.
  world point    base -> world at the planning-row qpos (hurdle_metrics.base_to_world).
  bar_gap_mm     analytic distance point -> hurdle collision geoms (hurdle_metrics.point_to_hurdle, obstacle manager of the run's config/profile) - r.
  apple_centre_mm  |point - apple centre (planning-row object_poses)|; apple_gap_mm = that - r - 33.3 mm (apple half-extent).
  table_gap_mm   point z - r - 0.82.
  cloud          filtered_cloud points (base frame) with |p - point_m| < r: n, and per point the nearest true surface among
                 bar (analytic), apple (|p - apple| - 33.3 mm), table (z - 0.82), robot (any robot collision geom: mj_geomDistance of a 2 mm probe is not used;
                 instead the point's distance to the active-arm finger / palm body origins < 60 mm counts as 'hand'); counts per class, and min bar distance of those points.
  sqp            to.metrics.returned, returned_index, initial_violation_m, final clearance, candidates (iteration, merit, violation_m, accepted, step_size_rad),
                 lookahead {gate, window_clearance_m, tail_clearance_m}, refine_dev_rad (max |refined - reference| rows 0-7, 14 arm joints),
                 servo_accel {n_over, overshoot_max_rad, boundary_velocity_ratio}, deflection_rate {cost}."""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/Y/p'))
import hurdle_metrics as HM  # noqa: E402
from esdf_held_y import samp  # noqa: E402

V11 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V11'))
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x
J14 = list(range(0, 7)) + list(range(8, 15))
APPLE_R = 0.0333


def main():
    res = {"definitions": __doc__, "arms": {}}
    for arm in ("V11AB", "V11ABC"):
        R = json.load(open(V11 / f"analysis/analysis_{arm}.json"))["runs"]; out = []
        for key, A in sorted(R.items()):
            if not A.get("hold_t") or A["stage"] != "H":
                continue
            ep = int(A["episode"]); side = "right" if ep == 1834 else "left"; HM.set_arm(side); HM._M.clear(); M = HM.model(); m, d = M["m"], M["d"]
            rd = pathlib.Path(A["run_dir"]); st = rd.parent / f"{rd.name}.start"
            cfg, prof = next(l.split("=", 1)[1].split(":", 1) for l in st.read_text().splitlines() if l.startswith("obstacle="))
            mg = HM.manager(cfg, prof); shapes = HM.hurdle_shapes(mg)
            plan = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]; plan = [p for p in plan if p["kind"] == "planning"]
            first = pathlib.Path(A["server_segment"]["first"]); i0 = int(first.stem.split("_")[1])
            fl = "l" if side == "left" else "r"; sd = "L" if side == "left" else "R"
            hand_b = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in (f"ee_finger_{fl}1", f"ee_finger_{fl}2", f"EE_BODY_{sd}")]
            for n, p in enumerate(plan):
                if (p.get("gate") or {}).get("action") != "hold":
                    continue
                z = np.load(first.parent / f"chunk_{i0 + n:05d}.npz", allow_pickle=True); assert int(z["t_step"]) == p["seq"]
                s = json.loads(str(z["summary_json"])); to = s["to"]; mt = to["metrics"]; mv = lit(mt["max_violation_pair"])
                q = int(mv["query"]); nr = int(z["sphere_radii"].shape[0])
                held = lit(((s.get("grasp") or {}).get("attached") or {}).get("held")) if (s.get("grasp") or {}).get("attached") else None
                if q < nr:
                    r = float(z["sphere_radii"][q]); kind = "robot"
                else:
                    bc = (held or {}).get("body_cover") if isinstance((held or {}).get("body_cover"), dict) else None
                    sph = (bc or {}).get("spheres") or (held or {}).get("spheres") or []
                    r = 1e-3 * float(sph[q - nr]["radius_mm"]) if q - nr < len(sph) else float("nan"); kind = "held"
                P = np.asarray(mv["point_m"], float); tier = mv.get("tier")
                e = (samp(z["esdf_distance"], z["esdf_origin"], z["esdf_voxel_size"], P) if tier == "fine"
                     else samp(z["esdf_coarse_distance"], z["esdf_coarse_origin"], z["esdf_coarse_voxel_size"], P))
                qp = np.asarray(p["qpos"], float); W = HM.base_to_world(qp, P)
                d.qpos[:] = qp; mujoco.mj_kinematics(m, d); base = M["bid"]["base"]; Rb = d.xmat[base].reshape(3, 3); ob = d.xpos[base].copy()
                hand = [d.xpos[b].copy() for b in hand_b]
                ap = np.asarray(p["object_poses"]["apple"]["pos"], float)
                cl = np.asarray(z["filtered_cloud"], float); inside = cl[np.linalg.norm(cl - P, axis=1) < r]
                cnt = {"bar": 0, "apple": 0, "table": 0, "hand": 0}; bar_min = None
                for c in inside:
                    w = ob + Rb @ c
                    dd = {"bar": HM.point_to_hurdle(w, shapes), "apple": np.linalg.norm(w - ap) - APPLE_R, "table": w[2] - 0.82,
                          "hand": min(np.linalg.norm(w - h) for h in hand) - 0.06}
                    cnt[min(dd, key=dd.get)] += 1; bar_min = dd["bar"] if bar_min is None else min(bar_min, dd["bar"])
                ref = np.asarray(z["reference_chunk"], float); fin = np.asarray(z["refined_chunk"], float)
                cands = [lit(c) for c in (lit(mt.get("candidates")) or [])]
                sa = lit(mt.get("servo_accel")) if mt.get("servo_accel") is not None else None
                la = lit(mt.get("lookahead")) if mt.get("lookahead") is not None else None
                dr = lit(mt.get("deflection_rate")) if mt.get("deflection_rate") is not None else None
                out.append({"key": key, "seq": p["seq"], "t": p["t_step"], "kind": kind, "link": mv.get("link"), "query": q, "step": mv.get("step"), "tier": tier,
                            "to_clearance_mm": 1e3 * mv["clearance_m"], "radius_mm": 1e3 * r, "esdf_mm": 1e3 * e, "esdf_gap_mm": 1e3 * (e - r),
                            "implied_margin_mm": 1e3 * (e - r - mv["clearance_m"]), "point_world_m": W.tolist(),
                            "bar_gap_mm": 1e3 * (HM.point_to_hurdle(W, shapes) - r), "apple_centre_mm": 1e3 * float(np.linalg.norm(W - ap)),
                            "apple_gap_mm": 1e3 * float(np.linalg.norm(W - ap) - r - APPLE_R), "table_gap_mm": 1e3 * (W[2] - r - 0.82),
                            "cloud_inside_n": int(len(inside)), "cloud_inside_by_nearest": cnt, "cloud_inside_bar_min_mm": None if bar_min is None else 1e3 * bar_min,
                            "collision_by_why": lit(s["verdict"]["classification"].get("collision_by_why")), "why": (lit(s["verdict"]["classification"].get("collision_worst")) or {}).get("why"),
                            "follow_to": s["verdict"].get("follow_to"), "grasp_state": s["grasp"].get("state"),
                            "sqp": {"returned": mt.get("returned"), "returned_index": mt.get("returned_index"), "initial_violation_mm": None if mt.get("initial_violation_m") is None else 1e3 * mt["initial_violation_m"],
                                    "initial_merit": mt.get("initial_merit"), "candidates": cands, "status": to.get("status"),
                                    "lookahead": None if not la else {k: la.get(k) for k in ("gate", "window_clearance_m", "tail_clearance_m", "tail_reference_clearance_m")},
                                    "refine_dev_rad": float(np.abs(fin[:8][:, J14] - ref[:8][:, J14]).max()),
                                    "servo_accel": None if not sa else {k: sa.get(k) for k in ("n_over", "overshoot_max_rad", "boundary_velocity_ratio")},
                                    "deflection_rate_cost": None if not dr else dr.get("cost"), "continuity": mt.get("continuity_reason")}})
                x = out[-1]; print(arm, key.split("/")[-1], x["t"], x["link"], round(x["to_clearance_mm"], 1), "r", round(x["radius_mm"], 1), "esdf", round(x["esdf_mm"], 1),
                                   "bar_gap", round(x["bar_gap_mm"], 1), "apple_c", round(x["apple_centre_mm"], 1), "cloud", x["cloud_inside_n"], x["cloud_inside_by_nearest"],
                                   "ret", x["sqp"]["returned"], x["sqp"]["returned_index"], "dev", round(x["sqp"]["refine_dev_rad"], 3), flush=True)
        res["arms"][arm] = out
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
