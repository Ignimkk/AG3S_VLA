"""T30 extra per-chunk fields on top of p/<COND>_analysis.json (analyze_e3_base.py = T28 analyze_e3.py with T30 paths).
usage: analyze_e3_extra.py COND   -> p/<COND>_analysis_extra.json

Per chunk (paired server chunk):
  association / anchor : server summary manipulated.association.outcome (+shift/ratio) and manipulated.anchor
                         {source, kept_frames, geometry_source, since_frame}; decision reason = summary.admissibility.decision.reason
  switched             : manipulated.switched_from not None, and 'after_switch' = any earlier chunk of the run had it
  frozen               : summary.exclusion.grasp_active
  ball vs GT apple     : recorded esdf_stats.target_free.ball {centre_m (base), radius_mm, z_min_m}; GT apple = planning row
                         object_poses apple pos -> base frame (base pose of that row's qpos). centre_inside = |c-a| <= r,
                         whole_inside = |c-a| + 34 mm <= r (z_min not applied). no ball -> both False, ball None.
  stamps               : local 'observation' row with the same t_step: stamp_mode, camera_skew_sec, render_spread_sec
  verdict              : server summary verdict.classification {n_violating, n_allowed, n_unresolved, n_collision,
                         collision_by_why, allowed_by_tier} and the reason kinds
  violating rows       : recorded refined chunk re-evaluated with the T24 tool (as off1/f1_counterfactual.py, new rule =
                         working tree): every (step, query) with esdf clearance < 0 on the executed 8-step window; for rows of
                         authorized links (target_free mask) and others separately: link, tier (field.answer_tier with the mask),
                         clearance, GT nearest object (mj_geomDistance probe). Attached chunks: robot-sphere rows only
                         (held-object query points are not recorded) and flagged.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, json, pathlib, sys
import numpy as np

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P  # noqa: E402
import AG3S.trajopt.experiments.fixed_input_ablation as T  # noqa: E402
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap  # noqa: E402

COND = sys.argv[1]
REC = pathlib.Path(f"{_ag3s_workspace_path()}/outputs/verify/T30/{COND}_rec")
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30/p'))
ANA = json.loads((V / (sys.argv[2] if len(sys.argv) > 2 else f"{COND}_analysis.json")).read_text())
OUTNAME = f"{COND}_analysis_extra.json" if len(sys.argv) <= 2 else f"{COND}_analysis_extra_partial.json"
R_APPLE = 0.034


def lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def setup(port, run_dir):
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.types import ChunkLayout
    args = T.build_parser().parse_args(["--constraints", str(REC / f"server_{port}/server_constraints/run_0000"), "--frames", str(run_dir / "frames.jsonl"),
                                        "--server-log", str(REC / f"serve_safe_{port}.log"), "--out", "/dev/null"])
    settings = T.resolve_settings(args, T.parse_server_log(args.server_log), json.loads((run_dir / "manifest.json").read_text()))
    robot, qadr, _ = T.build_robot(settings)
    layout = ChunkLayout.rby1(robot.joint_names)
    _, _, _, maxd, unk = T.contact_policy(settings, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
    planning = T.to_config(settings, collision=True, cost=dict(settings.cost), sqp={})
    meter = T.Meter(robot, layout, planning.horizon.planned, settings.hand,
                    build_limits(robot, layout, dt=planning.horizon.dt, config=planning.limits))
    return dict(settings=settings, robot=robot, qadr=qadr, layout=layout, lin=meter.lin, outside=maxd if unk == "free" else -maxd,
                fmap=FingerJointMap(robot))


def world_to_base(m, d, pw):
    import mujoco
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    return d.xmat[bid].reshape(3, 3).T @ (np.asarray(pw, float) - d.xpos[bid])


def main():
    m, d, probe, objg, _ = P.build()
    shared = {}
    out = {"condition": COND, "definitions": __doc__, "runs": {}}
    for rname, r in ANA["runs"].items():
        run_dir = REC / rname
        rows = [json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines() if l.strip()]
        plan = {x["seq"]: x for x in rows if x["kind"] == "planning"}
        obs = {x["t_step"]: x for x in rows if x["kind"] == "observation"}
        ex_chunks = []
        seen_switch = False
        for c in r["chunks"]:
            e = {"seq": c["seq"], "t": c["t"]}
            o = obs.get(c["t"])
            e["stamps"] = None if o is None else {"stamp_mode": o.get("stamp_mode"), "camera_skew_sec": o.get("camera_skew_sec"),
                                                  "render_spread_sec": o.get("render_spread_sec")}
            sc = c.get("server_chunk")
            if not sc:
                ex_chunks.append(e); continue
            port = sc.split("/")[0].split("_")[1]
            if port not in shared:
                shared[port] = setup(port, run_dir)
            sh = shared[port]; lin = sh["lin"]
            z = np.load(REC / sc, allow_pickle=False)
            s = json.loads(str(z["summary_json"]))
            man = s.get("manipulated") or {}
            assoc = man.get("association") or {}
            anc = man.get("anchor") or {}
            dec = ((s.get("admissibility") or {}).get("decision") or {})
            switched = man.get("switched_from") is not None
            seen_switch = seen_switch or (dec.get("mode") == "switch")
            e["association"] = {"outcome": assoc.get("outcome"), "shift_mm": assoc.get("shift_mm"), "shift_over_radius": assoc.get("shift_over_radius"),
                                "extent_ratio": assoc.get("extent_ratio"), "coverage": assoc.get("coverage"), "failed": assoc.get("failed")} if assoc else None
            e["anchor"] = {k: anc.get(k) for k in ("source", "since_frame", "kept_frames", "geometry_source", "geometry_frame", "radius_mm", "extents_mm")} if anc else None
            e["decision"] = {k: dec.get(k) for k in ("mode", "reason", "rank")} if dec else None
            e["manipulated_switched_from"] = man.get("switched_from")
            e["after_switch"] = seen_switch
            e["frozen"] = bool((s.get("exclusion") or {}).get("grasp_active"))
            pr = plan[c["seq"]]
            P.set_q(m, d, pr["qpos"])
            apple_b = world_to_base(m, d, pr["object_poses"]["apple"]["pos"])
            tf = lit((s.get("esdf_stats") or {}).get("target_free"))
            ball = (tf or {}).get("ball") if isinstance(tf, dict) else None
            if ball:
                da = float(np.linalg.norm(apple_b - np.asarray(ball["centre_m"])))
                rr = ball["radius_mm"] / 1000.0
                e["ball_vs_gt_apple"] = {"radius_mm": ball["radius_mm"], "apple_centre_dist_mm": 1000 * da, "centre_inside": bool(da <= rr),
                                         "whole_inside": bool(da + R_APPLE <= rr), "z_min_m": ball.get("z_min_m")}
            else:
                e["ball_vs_gt_apple"] = None
            cls = (s.get("verdict") or {}).get("classification") or {}
            e["verdict"] = {"kinds": [x.get("kind") for x in (s.get("verdict") or {}).get("reasons", [])],
                            **{k: cls.get(k) for k in ("n_violating", "n_allowed", "n_unresolved", "n_collision", "collision_by_why", "allowed_by_tier")},
                            "collision_worst": {k: (cls.get("collision_worst") or {}).get(k) for k in ("link", "tier", "label", "clearance_m", "why")} if cls.get("collision_worst") else None}
            # violating rows on the executed refined window
            attached = int((s.get("esdf_stats") or {}).get("n_attached_points", 0) or 0) > 0
            q_now = T.q_now_from_qpos(pr["qpos"], sh["qadr"])
            field = T.build_field(z, outside_distance=sh["outside"])
            tp = z["target_points"] if "target_points" in z.files else None
            margin, mask, _, _, _ = T.contact_policy(sh["settings"], sh["robot"].sphere_link_names,
                                                     has_target=bool(s.get("has_target")) and tp is not None,
                                                     has_free=bool(field.target_free_layers), attached=attached)
            scene = T.make_scene(field, lin.robot_radii, sh["settings"].esdf_margin, target_points=tp, margin=margin, mask=mask)
            refc = np.asarray(z["reference_chunk"], float)
            Pp, _ = sh["fmap"].path((s.get("finger_joints") or {}).get("opening_norm") or {}, {"left": refc[:, 7], "right": refc[:, 15]},
                                    lin.horizon, default=sh["robot"].param_vector())
            lin.set_joint_parameter_path(Pp)
            traj = sh["layout"].chunk_to_trajectory(np.asarray(z["refined_chunk"], float))[:, :lin.horizon]
            centres = lin.sphere_states(traj, q_now)[0]
            cl = lin.esdf_clearance(traj, q_now, scene)[..., 0]
            maskv = np.zeros(cl.shape[1], bool) if mask is None else np.asarray(mask, bool)
            names = [str(n) for n in sh["robot"].sphere_link_names]
            vrows = []
            for k, q in zip(*np.nonzero(cl < 0)):
                pt = centres[k, q]
                tier = lin._esdf_tier(pt.reshape(1, 3), scene, bool(maskv[q]))
                P.set_q(m, d, pr["qpos"])
                body, cat, per = P.nearest(m, d, probe, objg, P.base_to_world(m, d, pt))
                vrows.append({"step": int(k), "query": int(q), "link": names[q], "authorized": bool(maskv[q]), "tier": tier,
                              "clearance_mm": float(1000 * cl[k, q]), "gt_nearest": body, "gt_surface_mm": float(1000 * per[body]),
                              "apple_surface_mm": float(1000 * per["apple"])})
            P.set_q(m, d, pr["qpos"])
            lin.set_joint_parameter_path(None)
            e["violating_rows"] = {"attached_robot_rows_only": attached, "n": len(vrows), "n_authorized": sum(1 for v in vrows if v["authorized"]),
                                   "rows": vrows}
            ex_chunks.append(e)
        out["runs"][rname] = ex_chunks
        print(rname, len(ex_chunks), flush=True)
    out["n_invalid_geomdistance"] = P.N_INVALID[0]
    (V / OUTNAME).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
