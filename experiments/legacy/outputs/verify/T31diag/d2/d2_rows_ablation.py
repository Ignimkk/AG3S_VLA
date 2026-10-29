"""T31-diag D2 — which rows/terms drive refined-reference in the grasp -> place window (E3 ep1807 runs).

usage: MUJOCO_GL=osmesa PYTHONPATH=outputs/verify/T31diag/snap .venv-ag3s/bin/python d2_rows_ablation.py RUN [RUN ...]

Code: benchmark.* imported from the T31diag snapshot (outputs/verify/T31diag/snap, = T30 working tree, md5-checked).
Inputs restored with AG3S.trajopt.experiments.fixed_input_ablation (T24 tool) functions, as T30 off1/f1_counterfactual.py:
  settings  : parse_server_log(serve_safe_<port>.log) + manifest (phase approach, manipulators left, links gripper, 122 spheres,
              esdf_margin 0.0, w_track 1 w_smooth 0.05 w_continuity 0.5)
  field     : build_field(chunk npz) — coarse 20 mm + fine 5 mm (+ esdf_free if recorded)
  q_now     : paired local planning row (T30 p/<COND>_analysis.json 'server_chunk')
  fingers   : FingerJointMap.path(summary.finger_joints.opening_norm, reference cols 7/15) -> linearizer.set_joint_parameter_path
ATTACHED POINTS (not recorded — reconstructed here; the T24 tool skips attached chunks):
  at the chunk whose summary.grasp.attach == True: robot.set_joint_parameters(summary.finger_joints.joints), q = that chunk's q_now,
  geometry = TargetGeometry-like (points = recorded target_points of the attach chunk [or of the chunk before, whichever
  reproduces summary.grasp.attached.n_points], bounding_geometry = geometry.fit_sphere(points)),
  attached.attach_from_target(..., parent_link='ee_finger_l1') (defaults: voxel 10 mm, trim 20 mm / 3 neighbours).
  Held-object bounding sphere (manipulated_spheres, used by the linearizer's is-object test) = that sphere through FK at q_now.
  manipulated_link_margin = ClearancePolicy.margin_matrix(TARGET) (contact_policy with has_target, attached=False).
NOT reproduced: label layer -> destination margin (contact.destination_margin 20 mm on held points whose nearest label is
  'destination') is absent; fidelity numbers say how much that matters per chunk.
Fidelity: reconstructed worst_row(refined) vs recorded to.metrics.clearance_m / max_violation_pair (step, query);
          reconstructed max(0, -worst_row(reference)) vs recorded initial_violation_m.
Rows (reference = TO initial iterate, refined = recorded output), executed window 8 steps:
  worst row + all rows with clearance < 0: link ('attached:ee_finger_l1[i]' for held points), tier (field.answer_tier), attached flag,
  clearance mm, GT nearest body (mj_geomDistance probe, P1-2 helper; for held points also the nearest body excluding the apple).
Limits: reference overshoot with the TO limits (Meter.overshoot = trajopt.limits.limit_report, URDF - position_margin) and,
  separately, per joint vs MuJoCo jnt_range of model_transport.xml (rad beyond range; 0 = inside).
Ablation (per chunk, fresh solver, clock off, max_iterations = recorded sqp_iterations unless named):
  replay         : recorded cost, collision on, limits as recorded config, previous chunk = recorded previous refined_chunk
  no_collision   : collision.enabled False
  no_limits      : limits.enforce_position/velocity/acceleration False
  no_continuity  : w_continuity 0
  no_smooth      : w_smooth 0
  replay_3iter   : as replay with max_iterations 3 (the configured maximum) — not a recorded condition
  deviation = max over arm joints and executed steps of |traj - reference| (deg); fingertip = mean of left finger spheres (mm).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, dataclasses, json, pathlib, sys, time, types
import numpy as np
import mujoco

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P  # noqa
import AG3S.trajopt.experiments.fixed_input_ablation as T  # noqa
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap  # noqa
from AG3S.ag3s.constraints.attached import attach_from_target  # noqa
from AG3S.ag3s.stages.geometry import fit_sphere  # noqa
from AG3S.trajopt.linearize import SceneSnapshot  # noqa

T30 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d2'))
D1 = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d1/d1_oscillation.json')).read_text())
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')


def pair_of(m):
    p = m.get("max_violation_pair")
    return ast.literal_eval(p) if isinstance(p, str) else p


def lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def setup(cond, port, run_dir):
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.types import ChunkLayout
    rec = T30 / f"{cond}_rec"
    argv = ["--constraints", str(rec / f"server_{port}/server_constraints/run_0000"), "--frames", str(run_dir / "frames.jsonl"),
            "--server-log", str(rec / f"serve_safe_{port}.log"), "--out", "/dev/null"]
    args = T.build_parser().parse_args(argv)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    log = T.parse_server_log(args.server_log)
    settings = T.resolve_settings(args, log, manifest)
    robot, qadr, xml = T.build_robot(settings)
    layout = ChunkLayout.rby1(robot.joint_names)
    _, _, _, maxd, unk = T.contact_policy(settings, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
    outside = maxd if unk == "free" else -maxd
    planning = T.to_config(settings, collision=True, cost=dict(settings.cost), sqp={})
    limits = build_limits(robot, layout, dt=planning.horizon.dt, config=planning.limits)
    meter = T.Meter(robot, layout, planning.horizon.planned, settings.hand, limits)
    return dict(settings=settings, robot=robot, qadr=qadr, layout=layout, meter=meter, outside=outside, limits=limits,
                fmap=FingerJointMap(robot), log=log)


def variant_cfg(sh, name, rec_iters):
    st = sh["settings"]
    cost = dict(st.cost)
    sqp = {"time_budget_ms": T.NO_CLOCK_MS, "max_iterations": max(1, int(rec_iters))}
    collision = True
    extra = {}
    if name == "no_collision":
        collision = False
    elif name == "no_limits":
        extra = {"limits": {"enforce_position": False, "enforce_velocity": False, "enforce_acceleration": False}}
    elif name == "no_continuity":
        cost["w_continuity"] = 0.0
    elif name == "no_smooth":
        cost["w_smooth"] = 0.0
    elif name == "replay_3iter":
        sqp["max_iterations"] = 3
    cfg = T.to_config(st, collision=collision, cost=cost, sqp=sqp)
    if extra:
        cfg = cfg.with_overrides(extra)
    return cfg


VARIANTS = ("replay", "no_collision", "no_limits", "no_continuity", "no_smooth", "replay_3iter")


def main():
    m, d, probe, objg, _ = P.build()
    jr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): m.jnt_range[i].copy() for i in range(m.njnt)}
    runs = sys.argv[1:]
    for run in runs:
        cond = run.split("_")[0]
        A = json.loads((T30 / "p" / f"{cond}_analysis.json").read_text())["runs"][run]
        d1 = D1["runs"][run]
        g, rel = d1["grasp_t"], d1["release_t"]
        run_dir = T30 / f"{cond}_rec" / run
        rows = {int(x["seq"]): x for x in (json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines() if l.strip()) if x.get("kind") == "planning"}
        chunks = [c for c in A["chunks"] if c.get("server_chunk")]
        port = chunks[0]["server_chunk"].split("/")[0].split("_")[1]
        sh = setup(cond, port, run_dir)
        robot, lin, layout, st = sh["robot"], sh["meter"].lin, sh["layout"], sh["settings"]
        names = [str(n) for n in robot.sphere_link_names]
        S = len(names)
        jnames = [str(robot.joint_names[i]) for i in layout.q_indices]
        # --- reconstruct attached points
        att = None; att_info = None
        for i, c in enumerate(chunks):
            z = np.load(T30 / f"{cond}_rec" / c["server_chunk"], allow_pickle=False)
            s = json.loads(str(z["summary_json"]))
            gr = s.get("grasp") or {}
            if gr.get("attach"):
                want = (gr.get("attached") or {}).get("n_points")
                q_a = T.q_now_from_qpos(rows[c["seq"]]["qpos"], sh["qadr"])
                joints = (s.get("finger_joints") or {}).get("joints") or {}
                tries = []
                for src, cc in (("attach_chunk", c), ("previous_chunk", chunks[i - 1] if i else None)):
                    if cc is None:
                        continue
                    zz = np.load(T30 / f"{cond}_rec" / cc["server_chunk"], allow_pickle=False)
                    if "target_points" not in zz.files:
                        continue
                    tp = np.asarray(zz["target_points"], np.float64)
                    robot.set_joint_parameters({k: v for k, v in joints.items() if k in robot.param_joint_names})
                    geom = types.SimpleNamespace(id=0, points=tp, centroid=tp.mean(axis=0), bounding_geometry=fit_sphere(tp, semantic_role="target"))
                    a = attach_from_target(geom, robot_model=robot, robot_state=q_a, parent_link="ee_finger_l1", label="manipulated")
                    robot.set_joint_parameters(None)
                    tries.append((src, a, len(a.points), float(geom.bounding_geometry.dimensions[0])))
                ok = [t for t in tries if t[2] == want]
                pick = ok[0] if ok else tries[0]
                att = pick[1]
                att_info = {"attach_chunk_t": c["t"], "attach_seq": c["seq"], "recorded_n_points": want,
                            "tries": [{"source": t[0], "n_points": t[2], "bounding_radius_mm": 1000 * t[3]} for t in tries],
                            "used": pick[0], "n_points_match": bool(ok), "bounding_radius_mm": 1000 * pick[3],
                            "n_primitives": len(att.primitives), "parent_link": att.parent_link}
                break
        print(run, "attached", att_info, flush=True)
        out = {"run": run, "definitions": __doc__, "attached_reconstruction": att_info, "window": [g, rel], "chunks": []}
        runners = {}
        prev_refined = None
        t0 = time.time()
        for c in chunks:
            f = T30 / f"{cond}_rec" / c["server_chunk"]
            z = np.load(f, allow_pickle=False)
            if not (g <= c["t"] <= rel):
                prev_refined = np.asarray(z["refined_chunk"], np.float64)
                continue
            s = json.loads(str(z["summary_json"]))
            row = rows[c["seq"]]
            q_now = T.q_now_from_qpos(row["qpos"], sh["qadr"])
            to = s.get("to") or {}; tm = to.get("metrics") or {}; rp = pair_of(tm) or {}
            stt = s.get("esdf_stats") or {}
            attached = int(stt.get("n_attached_points", 0) or 0) > 0
            e = {"t": c["t"], "seq": c["seq"], "file": str(f.relative_to(T30)), "attached": attached,
                 "executed": c.get("n_executed"), "n_control": c.get("n_control"), "phase": c.get("phase"),
                 "gate": c.get("gate"),
                 "recorded": {"clearance_m": tm.get("clearance_m"), "worst": rp, "initial_violation_m": tm.get("initial_violation_m"),
                              "initial_limit_overshoot": lit(tm.get("initial_limit_overshoot")), "limit_overshoot": lit(tm.get("limit_overshoot")),
                              "returned": tm.get("returned"), "candidates": lit(tm.get("candidates")), "sqp_iterations": to.get("sqp_iterations"),
                              "time_budget_hit": to.get("time_budget_hit"), "status": to.get("status"),
                              "plan_join_discontinuity_rad": tm.get("plan_join_discontinuity_rad"), "timing_ms": lit(tm.get("timing_ms")),
                              "verdict_kinds": [r.get("kind") for r in (s.get("verdict") or {}).get("reasons", [])],
                              "classification": {k: (s.get("verdict") or {}).get("classification", {}).get(k) for k in ("n_violating", "n_allowed", "n_unresolved", "n_collision", "collision_by_why", "allowed_by_tier")},
                              "exec_continuity": s.get("exec_continuity")}}
            field = T.build_field(z, outside_distance=sh["outside"])
            tp = z["target_points"] if "target_points" in z.files else None
            margin, _, _, _, _ = T.contact_policy(st, robot.sphere_link_names, has_target=True, has_free=False, attached=False)
            fj = s.get("finger_joints") or {}
            meas = fj.get("opening_norm") or {}
            refc = np.asarray(z["reference_chunk"], np.float64)
            Pp, _ = sh["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, lin.horizon, default=robot.param_vector())
            if attached and att is not None:
                robot.set_joint_parameters({k: v for k, v in (fj.get("joints") or {}).items() if k in robot.param_joint_names})
                Tp = np.asarray(robot.link_pose(q_now, "ee_finger_l1"), np.float64)
                robot.set_joint_parameters(None)
                bs = att.primitives[0]
                man_c = (Tp[:3, :3] @ bs.center + Tp[:3, 3]).reshape(1, 3)
                man_r = np.array([float(bs.dimensions[0])])
                scene = SceneSnapshot(candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0), candidate_active=np.zeros(0, bool),
                                      d_safe=np.zeros((S, 0)), plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0),
                                      plane_active=np.zeros(0, bool), robot_radii=lin.robot_radii, candidate_ids=np.zeros(0, np.int64),
                                      esdf=field, esdf_margin=float(st.esdf_margin), manipulated_spheres=man_c,
                                      manipulated_sphere_radii=man_r, manipulated_link_margin=margin,
                                      attached_points=np.asarray(att.points, np.float64), attached_parent_link="ee_finger_l1")
                lin.set_attached(att.points, "ee_finger_l1")
            else:
                scene = T.make_scene(field, lin.robot_radii, st.esdf_margin, target_points=tp, margin=margin, mask=None)
                lin.set_attached(None, None)
            lin.set_joint_parameter_path(Pp)
            H = lin.horizon
            trajs = {"reference": layout.chunk_to_trajectory(refc)[:, :H],
                     "refined": layout.chunk_to_trajectory(np.asarray(z["refined_chunk"], np.float64))[:, :H]}
            P.set_q(m, d, row["qpos"])
            for tn, traj in trajs.items():
                cval, pair = lin.worst_row(traj, q_now, scene)
                cl = lin.esdf_clearance(traj, q_now, scene)[..., 0]
                centres = lin.sphere_states(traj, q_now)[0]
                tiers = np.asarray(field.answer_tier(centres.reshape(-1, 3))).reshape(cl.shape)
                neg = np.argwhere(cl < 0)
                rowsv = []
                for (k, q) in sorted(neg.tolist(), key=lambda kq: cl[kq[0], kq[1]])[:12]:
                    pw = P.base_to_world(m, d, centres[k, q])
                    body, cat, per = P.nearest(m, d, probe, objg, pw)
                    P.set_q(m, d, row["qpos"])
                    nx = {b: v for b, v in per.items() if b != "apple"}
                    rowsv.append({"step": k, "query": q, "link": lin._query_name(q) if hasattr(lin, "_query_name") else (names[q] if q < S else f"attached[{q - S}]"),
                                  "attached": bool(q >= S), "tier": str(tiers[k, q]), "clearance_mm": 1000 * float(cl[k, q]),
                                  "gt_nearest": body, "gt_nearest_mm": 1000 * per[body], "gt_nearest_excl_apple": min(nx, key=nx.get),
                                  "gt_nearest_excl_apple_mm": 1000 * min(nx.values())})
                negq = cl < 0
                by_link = {}
                for q in np.nonzero(negq.any(axis=0))[0]:
                    key = "attached" if q >= S else names[q]
                    by_link[key] = by_link.get(key, 0) + int(negq[:, q].sum())
                by_tier = {}
                for tt in tiers[negq]:
                    by_tier[str(tt)] = by_tier.get(str(tt), 0) + 1
                v = {"worst_clearance_mm": 1000 * float(cval), "worst": pair and {k2: pair.get(k2) for k2 in ("step", "query", "link", "tier", "obstacle")},
                     "n_neg_rows": int(negq.sum()), "neg_by_link": by_link, "neg_by_tier": by_tier, "neg_rows_top": rowsv}
                if pair and pair.get("point_m") is not None:
                    pw = P.base_to_world(m, d, pair["point_m"])
                    body, cat, per = P.nearest(m, d, probe, objg, pw)
                    P.set_q(m, d, row["qpos"])
                    nx = {b: vv for b, vv in per.items() if b != "apple"}
                    v["worst"].update({"attached": bool(int(pair["query"]) >= S), "gt_nearest": body, "gt_nearest_mm": 1000 * per[body],
                                       "gt_nearest_excl_apple": min(nx, key=nx.get), "gt_nearest_excl_apple_mm": 1000 * min(nx.values())})
                # tiers of all hand / held queries (fine vs coarse) — D3
                hand = [i for i, n in enumerate(names) if n in ("ee_left", "ee_finger_l1", "ee_finger_l2")] + list(range(S, cl.shape[1]))
                ht = tiers[:, hand]
                v["tier_counts_left_hand_and_held"] = {str(k2): int((ht == k2).sum()) for k2 in np.unique(ht)}
                if cl.shape[1] > S:
                    at = tiers[:, S:]
                    v["tier_counts_held_points"] = {str(k2): int((at == k2).sum()) for k2 in np.unique(at)}
                e[tn] = v
            e["fidelity"] = {"refined_worst_abs_diff_mm": (abs(e["refined"]["worst_clearance_mm"] - 1000 * tm["clearance_m"]) if tm.get("clearance_m") is not None else None),
                             "refined_same_row": bool(e["refined"]["worst"] and rp and e["refined"]["worst"]["step"] == rp.get("step") and e["refined"]["worst"]["query"] == rp.get("query")),
                             "reference_initial_violation_abs_diff_mm": (abs(max(0.0, -e["reference"]["worst_clearance_mm"]) - 1000 * tm["initial_violation_m"]) if tm.get("initial_violation_m") is not None else None)}
            # limits
            e["reference_overshoot_TO_limits"] = sh["meter"].overshoot(trajs["reference"], q_now)
            mj = {}
            for jj, jn in enumerate(jnames):
                lo, hi = jr[jn]
                x = trajs["reference"][jj]
                over = float(max(0.0, (x - hi).max(), (lo - x).max()))
                if over > 0:
                    mj[jn] = over
            e["reference_beyond_mujoco_range_rad"] = mj
            tol = {}
            lo_to, hi_to = np.asarray(sh["limits"].lower, float), np.asarray(sh["limits"].upper, float)
            for jj, jn in enumerate(jnames):
                x = trajs["reference"][jj]
                over = float(max(0.0, (x - hi_to[jj]).max(), (lo_to[jj] - x).max()))
                if over > 0:
                    tol[jn] = {"over_rad": over, "to_lower": float(lo_to[jj]), "to_upper": float(hi_to[jj]), "mj_range": [float(jr[jn][0]), float(jr[jn][1])],
                               "ref_min": float(x.min()), "ref_max": float(x.max())}
            e["reference_beyond_TO_position_limits"] = tol
            dev = np.degrees(np.abs(trajs["refined"] - trajs["reference"]))
            e["recorded_dev_deg"] = {"max": float(dev.max()), "per_joint": {jn: float(dev[jj].max()) for jj, jn in enumerate(jnames)}}
            # ablation
            rec_it = int(to.get("sqp_iterations") or to.get("iterations") or 1)
            certified = s.get("status") == "ok"
            ab = {}
            ref_tip, ref_rot = sh["meter"].tip_and_rot(trajs["reference"], q_now)
            for vn in VARIANTS:
                cfg = variant_cfg(sh, vn, rec_it)
                key = (vn, T._cfg_key(cfg))
                if key not in runners:
                    runners[key] = T.Instrumented(robot, layout, cfg, {"value": None})
                rn = runners[key]
                rn.holder["value"] = (scene, q_now, certified)
                rlin = rn.refiner.optimizer.linearizer
                rn.refiner.reset()
                rlin.set_joint_parameter_path(Pp)
                rn.captured = []
                try:
                    rn.refiner.refine(refc, {"t_step": c["seq"], "previous_physical_chunk": prev_refined})
                    res = rn.refiner.last_result
                    if res is None:
                        ab[vn] = {"error": str(rn.refiner.last_failure)}; continue
                    tr = res.trajectory
                    cmpd = sh["meter"].compare(tr, trajs["reference"], q_now, ref_tip, ref_rot)
                    mm = res.metrics
                    ab[vn] = {"dev_max_deg": cmpd["joint_dev_max_deg"], "dev_per_joint_deg": cmpd["joint_dev_per_joint_deg"],
                              "fingertip_disp_mm_max": cmpd["fingertip_disp_mm_max"], "status": res.status.value,
                              "returned": mm.get("returned"), "sqp_iterations": mm.get("sqp_iterations"),
                              "final_violation_mm": 1000 * float(res.max_violation), "candidates": mm.get("candidates"),
                              "worst_row": mm.get("max_violation_pair"),
                              "vs_recorded_refined_max_deg": float(np.degrees(np.abs(tr - trajs["refined"])).max())}
                except Exception as ex:  # noqa
                    ab[vn] = {"error": f"{type(ex).__name__}: {ex}"}
            e["ablation"] = ab
            lin.set_joint_parameter_path(None); lin.set_attached(None, None)
            out["chunks"].append(e)
            prev_refined = np.asarray(z["refined_chunk"], np.float64)
            print(run, c["t"], "att", attached, "fid", None if e["fidelity"]["refined_worst_abs_diff_mm"] is None else round(e["fidelity"]["refined_worst_abs_diff_mm"], 2), e["fidelity"]["refined_same_row"],
                  "ref worst", e["reference"]["worst"] and (e["reference"]["worst"]["link"], e["reference"]["worst"]["tier"], round(e["reference"]["worst_clearance_mm"], 1), e["reference"]["worst"].get("gt_nearest")),
                  "limTO", {k: round(v["over_rad"], 3) for k, v in tol.items()},
                  "dev", round(e["recorded_dev_deg"]["max"], 2), {k: round(v.get("dev_max_deg", -1), 2) for k, v in ab.items()}, f"{time.time() - t0:.0f}s", flush=True)
            (OUT / f"d2_{run}.json").write_text(json.dumps(out, indent=1, default=float))
        out["n_invalid_geomdistance"] = P.N_INVALID[0]
        (OUT / f"d2_{run}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
