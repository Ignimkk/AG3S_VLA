"""T35-diag B — which rows / terms drive refined - reference in the 10 chunks before and after the first close command
(4 failed T34 runs; T34 records, working tree = T34 code, md5-checked).

usage: MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} <py> b_ablation.py RUN [RUN ...]   (RUN = E3b_ep1800_r1 ...)

Window: planning rows (local seq) s_close-10 .. s_close+10, s_close = chunk of the first control row with applied left
gripper < 0.5 (a/a_runs.json t_close). Only chunks paired to a server npz (T34 p/<COND>_analysis.json 'server_chunk').
Inputs restored with AG3S.trajopt.experiments.fixed_input_ablation (T24 tool) functions, as T31diag d2_rows_ablation.py:
  settings  : parse_server_log(serve_safe_<port>.log) + manifest
  field     : build_field(chunk npz) — coarse + fine + target-free layer when recorded (outside = esdf max_distance)
  q_now     : paired local planning row qpos
  contact   : T.contact_policy(has_target, has_free, attached) -> manipulated_link_margin + target_free_mask; scene = T.make_scene
  fingers   : FingerJointMap.path(summary.finger_joints.opening_norm, reference cols 7/15) -> set_joint_parameter_path
HELD chunks (summary.grasp.attached_in_constraints and grasp.attached.held): held spheres = recorded
  held.spheres_parent_frame {centre_m, radius_mm} on held.frame_link -> scene.attached_points / attached_parent_link /
  attached_radii (T32 H2 path of scene_from_constraint_set); manipulated_spheres = those spheres through FK at q_now;
  manipulated_link_margin = contact_policy(has_target=True, attached=False) margin; target_free_mask None.
  (T24 tool skips attached chunks; this is the T31diag/T33 extension with the T34 held-sphere record.)
Rows (reference = TO initial iterate, refined = recorded output), executed window = planned horizon:
  worst row + rows with clearance < 0: link ('attached[i]' for held spheres), tier (field.answer_tier), clearance mm,
  GT nearest body (mj_geomDistance probe, P1-2 helper) and nearest excluding apple.
Fidelity: reconstructed worst_row(refined) vs recorded to.metrics.clearance_m / max_violation_pair (step, query).
Limits: reference and refined overshoot vs TO limits (Meter.overshoot) and per joint beyond MuJoCo jnt_range of
  model_transport.xml (rad; 0 = inside); recorded to.metrics.initial_limit_overshoot / limit_overshoot.
Ablation (per chunk, fresh solver, clock off, max_iterations = recorded sqp_iterations):
  replay / no_collision / no_limits / no_continuity / no_smooth  (+ replay_3iter)
  dev = max over arm joints and planned steps of |traj - reference| (deg), per joint; fingertip displacement (mm);
  returned path; vs_recorded_refined_max_deg (replay fidelity).
Recorded refined vs reference: Meter.compare -> joint dev and fingertip (mean of left finger spheres) displacement per step (mm).
Join: |reference_chunk[0] - previous chunk's recorded refined_chunk[8]| per column (deg) — the row the previous chunk would
  have commanded next (8 rows executed per chunk).
Recorded per chunk: returned, returned_index, candidates, sqp_iterations, verdict kinds; per-joint recorded dev (deg).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, json, pathlib, sys, time
import numpy as np
import mujoco

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P  # noqa
import AG3S.trajopt.experiments.fixed_input_ablation as T  # noqa
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap  # noqa
from AG3S.trajopt.linearize import SceneSnapshot  # noqa

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T34'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T35diag/b'))
ARUNS = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T35diag/a/a_runs.json')).read_text())["runs"]
PORT = {"E3a": 8204, "E3b": 8206}
HALF = 10


def lit(v):
    try:
        return ast.literal_eval(v) if isinstance(v, str) else v
    except Exception:
        return v


def setup(cond, run_dir):
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.types import ChunkLayout
    rec = V / f"{cond}_rec"
    port = PORT[cond]
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
                fmap=FingerJointMap(robot), log=log,
                settings_record={k: str(getattr(settings, k)) for k in ("phase", "manipulators", "links", "esdf_margin", "cost",
                                                                        "target_field_policy", "plan_horizon", "hand", "require_certified")
                                 if hasattr(settings, k)})


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


def beyond(traj, jnames, lo, hi):
    out = {}
    for jj, jn in enumerate(jnames):
        x = traj[jj]
        over = float(max(0.0, (x - hi[jj]).max(), (lo[jj] - x).max()))
        if over > 0:
            out[jn] = {"over_rad": over, "over_deg": float(np.degrees(over)), "lo": float(lo[jj]), "hi": float(hi[jj]),
                       "min": float(x.min()), "max": float(x.max())}
    return out


def main():
    m, d, probe, objg, _ = P.build()
    jr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): m.jnt_range[i].copy() for i in range(m.njnt)}
    OUT.mkdir(parents=True, exist_ok=True)
    for run in sys.argv[1:]:
        cond = run.split("_")[0]
        A = json.loads((V / "p" / f"{cond}_analysis.json").read_text())["runs"][run]
        ar = ARUNS[f"{cond}_rec/{run}"]
        t_close = ar["t_close"]
        s_close = next(e["seq"] for e in ar["timeline"] if e["t"] == t_close)
        run_dir = V / f"{cond}_rec" / run
        rows = {int(x["seq"]): x for x in (json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines() if l.strip())
                if x.get("kind") == "planning"}
        chunks = [c for c in A["chunks"] if c.get("server_chunk")]
        sh = setup(cond, run_dir)
        robot, lin, layout, st = sh["robot"], sh["meter"].lin, sh["layout"], sh["settings"]
        names = [str(n) for n in robot.sphere_link_names]
        S = len(names)
        jnames = [str(robot.joint_names[i]) for i in layout.q_indices]
        lo_to, hi_to = np.asarray(sh["limits"].lower, float), np.asarray(sh["limits"].upper, float)
        lo_mj = np.array([jr[j][0] for j in jnames]); hi_mj = np.array([jr[j][1] for j in jnames])
        out = {"run": run, "definitions": __doc__, "t_close": t_close, "s_close": s_close, "window_seq": [s_close - HALF, s_close + HALF],
               "settings": sh["settings_record"], "joint_names": jnames,
               "to_limits": {"lower": lo_to.tolist(), "upper": hi_to.tolist()}, "mj_range": {"lower": lo_mj.tolist(), "upper": hi_mj.tolist()},
               "chunks": []}
        runners = {}
        prev_refined = None
        t0 = time.time()
        for c in chunks:
            f = V / f"{cond}_rec" / c["server_chunk"]
            z = np.load(f, allow_pickle=False)
            if not (s_close - HALF <= c["seq"] <= s_close + HALF):
                prev_refined = np.asarray(z["refined_chunk"], np.float64)
                continue
            s = json.loads(str(z["summary_json"]))
            row = rows[c["seq"]]
            q_now = T.q_now_from_qpos(row["qpos"], sh["qadr"])
            to = s.get("to") or {}; tm = to.get("metrics") or {}; rp = lit(tm.get("max_violation_pair")) or {}
            gr = s.get("grasp") or {}
            held = (gr.get("attached") or {}).get("held") if gr.get("attached_in_constraints") else None
            e = {"t": c["t"], "seq": c["seq"], "file": str(f.relative_to(V)), "held": bool(held),
                 "rel_seq": c["seq"] - s_close, "executed": c.get("n_executed"), "n_control": c.get("n_control"),
                 "grasp_state": gr.get("state"),
                 "recorded": {"clearance_m": tm.get("clearance_m"), "worst": {k: rp.get(k) for k in ("step", "query", "link", "tier", "clearance_m")} if rp else None,
                              "initial_violation_m": tm.get("initial_violation_m"),
                              "initial_limit_overshoot": lit(tm.get("initial_limit_overshoot")), "limit_overshoot": lit(tm.get("limit_overshoot")),
                              "returned": tm.get("returned"), "returned_index": tm.get("returned_index"), "candidates": lit(tm.get("candidates")),
                              "sqp_iterations": tm.get("sqp_iterations"), "time_budget_hit": tm.get("time_budget_hit"),
                              "initial_eligible": tm.get("initial_eligible"), "status": to.get("status"),
                              "verdict_kinds": [r.get("kind") for r in (s.get("verdict") or {}).get("reasons", [])]}}
            field = T.build_field(z, outside_distance=sh["outside"])
            tp = z["target_points"] if "target_points" in z.files else None
            has_target = bool(s.get("has_target")) and tp is not None
            fj = s.get("finger_joints") or {}
            meas = fj.get("opening_norm") or {}
            refc = np.asarray(z["reference_chunk"], np.float64)
            Pp, _ = sh["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, lin.horizon, default=robot.param_vector())
            if held:
                link = held["frame_link"]
                cl_ = np.array([x["centre_m"] for x in held["spheres_parent_frame"]], float)
                rr_ = np.array([x["radius_mm"] / 1000.0 for x in held["spheres_parent_frame"]], float)
                Tq = np.asarray(robot.link_pose(q_now, link), float)
                man_c = (Tq[:3, :3] @ cl_.T).T + Tq[:3, 3]
                margin, _, _, _, _ = T.contact_policy(st, robot.sphere_link_names, has_target=True, has_free=False, attached=False)
                scene = SceneSnapshot(candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0), candidate_active=np.zeros(0, bool),
                                      d_safe=np.zeros((S, 0)), plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0),
                                      plane_active=np.zeros(0, bool), robot_radii=lin.robot_radii, candidate_ids=np.zeros(0, np.int64),
                                      esdf=field, esdf_margin=float(st.esdf_margin), manipulated_spheres=man_c,
                                      manipulated_sphere_radii=rr_, manipulated_link_margin=margin,
                                      attached_points=cl_, attached_parent_link=link, attached_radii=rr_)
                lin.set_attached(cl_, link, rr_)
                e["held_spheres"] = {"frame_link": link, "n": len(rr_), "radii_mm": (1000 * rr_).tolist()}
            else:
                margin, mask, _, _, _ = T.contact_policy(st, robot.sphere_link_names, has_target=has_target,
                                                         has_free=bool(field.target_free_layers), attached=False)
                scene = T.make_scene(field, lin.robot_radii, st.esdf_margin, target_points=tp, margin=margin, mask=mask)
                lin.set_attached(None, None)
                e["target_free_mask"] = mask is not None
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
                for (k, q) in sorted(neg.tolist(), key=lambda kq: cl[kq[0], kq[1]])[:8]:
                    pw = P.base_to_world(m, d, centres[k, q])
                    body, cat, per = P.nearest(m, d, probe, objg, pw)
                    P.set_q(m, d, row["qpos"])
                    nx = {b: v for b, v in per.items() if b != "apple"}
                    rowsv.append({"step": k, "query": q, "link": names[q] if q < S else f"attached[{q - S}]",
                                  "tier": str(tiers[k, q]), "clearance_mm": 1000 * float(cl[k, q]),
                                  "gt_nearest": body, "gt_nearest_mm": 1000 * per[body], "gt_nearest_excl_apple": min(nx, key=nx.get),
                                  "gt_nearest_excl_apple_mm": 1000 * min(nx.values())})
                negq = cl < 0
                by_link = {}
                for q in np.nonzero(negq.any(axis=0))[0]:
                    key = f"attached[{q - S}]" if q >= S else names[q]
                    by_link[key] = by_link.get(key, 0) + int(negq[:, q].sum())
                v = {"worst_clearance_mm": 1000 * float(cval), "n_neg_rows": int(negq.sum()), "neg_by_link": by_link,
                     "neg_by_tier": {str(k2): int((tiers[negq] == k2).sum()) for k2 in np.unique(tiers[negq])}, "neg_rows_top": rowsv}
                if pair:
                    q = int(pair.get("query"))
                    wv = {"step": pair.get("step"), "query": q, "link": names[q] if q < S else f"attached[{q - S}]", "tier": pair.get("tier")}
                    pt = pair.get("point_m")
                    if pt is None:
                        pt = centres[int(pair["step"]), q]
                    pw = P.base_to_world(m, d, pt)
                    body, cat, per = P.nearest(m, d, probe, objg, pw)
                    P.set_q(m, d, row["qpos"])
                    nx = {b: vv for b, vv in per.items() if b != "apple"}
                    wv.update({"gt_nearest": body, "gt_nearest_mm": 1000 * per[body], "gt_nearest_excl_apple": min(nx, key=nx.get),
                               "gt_nearest_excl_apple_mm": 1000 * min(nx.values()), "gt_apple_mm": 1000 * per["apple"]})
                    v["worst"] = wv
                v["limit_TO"] = sh["meter"].overshoot(traj, q_now)
                v["beyond_TO_position"] = beyond(traj, jnames, lo_to, hi_to)
                v["beyond_MJCF_range"] = beyond(traj, jnames, lo_mj, hi_mj)
                e[tn] = v
            rcl = tm.get("clearance_m")
            e["fidelity"] = {"refined_worst_abs_diff_mm": (abs(e["refined"]["worst_clearance_mm"] - 1000 * rcl) if rcl is not None else None),
                             "refined_same_row": bool(e["refined"].get("worst") and rp and e["refined"]["worst"]["step"] == rp.get("step")
                                                      and e["refined"]["worst"]["query"] == rp.get("query")),
                             "reference_initial_violation_abs_diff_mm": (abs(max(0.0, -e["reference"]["worst_clearance_mm"]) - 1000 * tm["initial_violation_m"])
                                                                         if tm.get("initial_violation_m") is not None else None)}
            dev = np.degrees(np.abs(trajs["refined"] - trajs["reference"]))
            ref_tip0, ref_rot0 = sh["meter"].tip_and_rot(trajs["reference"], q_now)
            cmp_rec = sh["meter"].compare(trajs["refined"], trajs["reference"], q_now, ref_tip0, ref_rot0)
            e["recorded_refined_vs_reference"] = {k2: cmp_rec.get(k2) for k2 in ("joint_dev_max_deg", "fingertip_disp_mm_max", "fingertip_disp_mm_per_step")}
            if prev_refined is not None:
                jd = np.degrees(np.abs(refc[0] - prev_refined[8]))
                e["join_ref0_vs_prev_refined8_deg"] = {"max_arm": float(jd[[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]].max()),
                                                       "per_col": jd.tolist()}
            e["recorded_dev_deg"] = {"max": float(dev.max()), "per_joint": {jn: float(dev[jj].max()) for jj, jn in enumerate(jnames)},
                                     "rows0_7_max": float(dev[:, :8].max())}
            rec_it = int(tm.get("sqp_iterations") or to.get("sqp_iterations") or 1)
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
                rn.refiner.reset()
                rn.refiner.optimizer.linearizer.set_joint_parameter_path(Pp)
                rn.captured = []
                try:
                    rn.refiner.refine(refc, {"t_step": c["seq"], "previous_physical_chunk": prev_refined})
                    res = rn.refiner.last_result
                    if res is None:
                        ab[vn] = {"error": str(rn.refiner.last_failure)}
                        continue
                    tr = res.trajectory
                    cmpd = sh["meter"].compare(tr, trajs["reference"], q_now, ref_tip, ref_rot)
                    mm = res.metrics
                    ab[vn] = {"dev_max_deg": cmpd["joint_dev_max_deg"], "dev_per_joint_deg": cmpd["joint_dev_per_joint_deg"],
                              "fingertip_disp_mm_max": cmpd["fingertip_disp_mm_max"], "status": res.status.value,
                              "returned": mm.get("returned"), "returned_index": mm.get("returned_index"), "sqp_iterations": mm.get("sqp_iterations"),
                              "final_violation_mm": 1000 * float(res.max_violation), "initial_violation_mm": 1000 * float(mm.get("initial_violation_m") or 0.0),
                              "initial_limit_overshoot": mm.get("initial_limit_overshoot"), "limit_overshoot": mm.get("limit_overshoot"),
                              "worst_row": mm.get("max_violation_pair"),
                              "vs_recorded_refined_max_deg": float(np.degrees(np.abs(tr - trajs["refined"])).max())}
                except Exception as ex:  # noqa
                    ab[vn] = {"error": f"{type(ex).__name__}: {ex}"}
            e["ablation"] = ab
            lin.set_joint_parameter_path(None); lin.set_attached(None, None)
            out["chunks"].append(e)
            prev_refined = np.asarray(z["refined_chunk"], np.float64)
            print(run, c["t"], "rel", e["rel_seq"], "held", e["held"], "fid", None if e["fidelity"]["refined_worst_abs_diff_mm"] is None else round(e["fidelity"]["refined_worst_abs_diff_mm"], 2),
                  e["fidelity"]["refined_same_row"], "ret", tm.get("returned"),
                  "ref worst", e["reference"].get("worst") and (e["reference"]["worst"]["link"], e["reference"]["worst"]["tier"], round(e["reference"]["worst_clearance_mm"], 1), e["reference"]["worst"]["gt_nearest"]),
                  "limMJ", {k: round(v["over_deg"], 2) for k, v in e["reference"]["beyond_MJCF_range"].items()},
                  "dev", round(e["recorded_dev_deg"]["max"], 2), {k: round(v.get("dev_max_deg", -1), 2) for k, v in ab.items()}, f"{time.time() - t0:.0f}s", flush=True)
            (OUT / f"b_{run}.json").write_text(json.dumps(out, indent=1, default=float))
        out["n_invalid_geomdistance"] = P.N_INVALID[0]
        (OUT / f"b_{run}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
