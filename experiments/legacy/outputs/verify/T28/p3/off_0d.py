"""T28 Phase 3 offline 0d — T16 chunks replayed: which real object does the worst constraint row sit in, with the T29 finger joints.

Inputs restored with AG3S.trajopt.experiments.fixed_input_ablation (working tree): settings from T16 serve_safe.log
(links gripper, capsule_radius_scale 0.2, esdf_margin 10 mm, exclude-authorized), recorded field (coarse/fine/target-free layers
from chunk_k.npz), contact policy (ClearancePolicy x phase approach, manipulators left), q_now from local planning row seq k+1.
prepare_chunk() checks reconstruction fidelity with the model's default (closed, pre-T29) fingers.
Trajectory = the recorded refined chunk (what the verdict was computed on), planned window 8 steps.
Variants (worst row = linearizer.worst_row on the TO scene incl. contact policy):
  T16model_oldfinger : T16 sphere model, finger params = 0 (pre-T29)            -- reproduces the recorded verdict row
  T16model_T29finger : T16 sphere model, finger param path = FingerJointMap.path(measured opening of row seq k+1 (qpos, client
                       convention), commanded = refined chunk col 7 rows 0..7, horizon 8)  (SafePolicy T29 envelope)
  E3model_T29finger  : E3 sphere model (scale 0.05, spacing 0.4, max 32), esdf_margin 0, same finger path
Worst point = the worst row's query sphere centre (base) -> MuJoCo world (base pose from that row's qpos); nearest GT object by
mj_geomDistance from a 0.1 mm probe (P1-2 helper); 'inside apple' = signed distance to the apple collision geom < 0.
The FIELD is the recorded T16 field (built with the T16 self-filter: inflation 0.05, fingers closed) — not rebuilt (no depth recorded).
HOLD chunks = P1-2 (p12_records.json) T16 chunks with executed False.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P
import AG3S.trajopt.experiments.fixed_input_ablation as T
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap, openings_from_mujoco

RUN = pathlib.Path(_ag3s_workspace_path('outputs/live_test/20260928_t16'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/p3'))
p12 = json.load(open(_ag3s_workspace_path('outputs/verify/T28/P1-2/p12_records.json')))
hold_chunks = {e["chunk"] for e in p12["runs"]["t16"]["chunks"] if e.get("has_local_row") and not e["executed"]}
p12_by_chunk = {e["chunk"]: e for e in p12["runs"]["t16"]["chunks"]}

argv = ["--constraints", str(RUN / "server_constraints/run_0000"), "--frames", str(RUN / "execute_ep1807/frames.jsonl"),
        "--server-log", str(RUN / "serve_safe.log"), "--out", "/dev/null"]


def setup(extra):
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.types import ChunkLayout
    args = T.build_parser().parse_args(argv + extra)
    manifest = json.loads((RUN / "execute_ep1807/manifest.json").read_text())
    log = T.parse_server_log(args.server_log)
    settings = T.resolve_settings(args, log, manifest)
    robot, qadr, xml = T.build_robot(settings)
    layout = ChunkLayout.rby1(robot.joint_names)
    _, _, _, maxd, unk = T.contact_policy(settings, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
    outside = maxd if unk == "free" else -maxd
    rows = [json.loads(l) for l in (RUN / "execute_ep1807/frames.jsonl").read_text().splitlines() if l.strip()]
    plan_rows = {int(r["seq"]): r for r in rows if r.get("kind") == "planning"}
    planning = T.to_config(settings, collision=True, cost=dict(settings.cost), sqp={})
    limits = build_limits(robot, layout, dt=planning.horizon.dt, config=planning.limits)
    meter = T.Meter(robot, layout, planning.horizon.planned, settings.hand, limits)
    sh = T.Shared(args=args, settings=settings, robot=robot, qadr=qadr, layout=layout, meter=meter, plan_rows=plan_rows,
                  variant_configs={}, active_rows_config=None, outside_distance=outside, recorded_iters={})
    return sh


def main():
    m, d, probe, obj, fing = P.build()
    import mujoco
    sh16 = setup([])
    she3 = setup(["--sphere-spacing", "0.4", "--max-spheres-per-capsule", "32", "--capsule-radius-scale", "0.05", "--esdf-margin", "0"])
    fm16, fme3 = FingerJointMap(sh16.robot), FingerJointMap(she3.robot)
    files = sorted((RUN / "server_constraints/run_0000").glob("chunk_*.npz"), key=T._chunk_index)
    out = []
    for f in files:
        rec, inp = T.prepare_chunk(sh16, f)
        k = rec["chunk_index"]
        e = {"chunk": k, "t_step": 8 * k, "hold": k in hold_chunks, "skipped_reason": rec.get("skipped_reason"),
             "fidelity": {kk: rec.get("fidelity", {}).get(kk) for kk in ("record_clearance", "to_view", "within_tolerance")},
             "p12_worst_nearest_category": (p12_by_chunk.get(k) or {}).get("worst_nearest_category"),
             "p12_worst_link": ((p12_by_chunk.get(k) or {}).get("worst_row") or {}).get("link")}
        pr = sh16.plan_rows.get(k + 1)
        if inp is None and pr is not None and rec.get("attached"):
            # attached chunk: robot-sphere rows only (attached query points are not recorded) — flagged
            z = np.load(f)
            field = T.build_field(z, outside_distance=sh16.outside_distance)
            q_now = T.q_now_from_qpos(pr["qpos"], sh16.qadr)
            scene = T.make_scene(field, sh16.meter.lin.robot_radii, sh16.settings.esdf_margin)
            planned = sh16.meter.lin.horizon
            inp = T.ChunkInputs(k=k, seq=k + 1, z=z, summary=json.loads(str(z["summary_json"])), q_now=q_now, field=field,
                                scene=scene, reference_chunk=None, reference=None,
                                refined_traj=sh16.layout.chunk_to_trajectory(np.asarray(z["refined_chunk"], np.float64))[:, :planned],
                                certified=False, ref_tip=None, ref_rot=None)
            e["robot_rows_only_attached_not_recorded"] = True
        if inp is None or pr is None:
            out.append(e); continue
        P.set_q(m, d, pr["qpos"])
        opn = openings_from_mujoco(m, d)
        rc = np.asarray(np.load(f)["refined_chunk"], float)
        cmd = {"left": np.clip(rc[:8, 7], 0, 1).tolist(), "right": np.clip(rc[:8, 15], 0, 1).tolist()}
        e["measured_opening"] = opn
        variants = {}
        for vname, sh, fmap, new in (("T16model_oldfinger", sh16, fm16, False), ("T16model_T29finger", sh16, fm16, True),
                                     ("E3model_T29finger", she3, fme3, True)):
            lin = sh.meter.lin
            if new:
                Ppath, env = fmap.path(opn, cmd, lin.horizon)
                lin.set_joint_parameter_path(Ppath)
            else:
                lin.set_joint_parameter_path(np.zeros((len(lin.param_names), lin.horizon)))
            if sh is sh16:
                scene = inp.scene
            else:
                z = np.load(f)
                s = json.loads(str(z["summary_json"]))
                tp = z["target_points"] if "target_points" in z.files else None
                has_t = bool(s.get("has_target")) and tp is not None
                margin, mask, _, _, _ = T.contact_policy(sh.settings, sh.robot.sphere_link_names, has_target=has_t,
                                                         has_free=bool(inp.field.target_free_layers),
                                                         attached=bool(e.get("robot_rows_only_attached_not_recorded")))
                scene = T.make_scene(inp.field, lin.robot_radii, sh.settings.esdf_margin, target_points=tp, margin=margin, mask=mask)
            c, pair = lin.worst_row(inp.refined_traj, inp.q_now, scene)
            v = {"clearance_mm": float(1000 * c), "pair": {kk: pair.get(kk) for kk in ("block", "step", "query", "link", "tier")} if pair else None}
            if pair and pair.get("point_m") is not None:
                P.set_q(m, d, pr["qpos"])
                pw = P.base_to_world(m, d, pair["point_m"])
                body, cat, per = P.nearest(m, d, probe, obj, pw)
                P.set_q(m, d, pr["qpos"])
                v.update({"point_world": pw.tolist(), "nearest_body": body, "nearest_category": cat,
                          "nearest_surface_mm": 1000 * per[body], "apple_surface_mm": 1000 * per["apple"],
                          "inside_apple": bool(per["apple"] < 0), "sphere_radius_mm": float(1000 * lin.robot_radii[pair["query"]])})
            variants[vname] = v
            lin.set_joint_parameter_path(None)
        e["variants"] = variants
        out.append(e)
        print(k, "HOLD" if e["hold"] else "exec", {vn: (vv.get("pair") or {}).get("link") and ((vv["pair"]["link"], vv.get("nearest_body"), round(vv.get("apple_surface_mm", 0), 1), round(vv["clearance_mm"], 1))) for vn, vv in variants.items()}, flush=True)
    summ = {}
    for sel_name, sel in (("hold_chunks", lambda e: e["hold"]), ("all_chunks", lambda e: True)):
        L = [e for e in out if sel(e) and e.get("variants")]
        s = {"n": len(L), "n_skipped": sum(1 for e in out if sel(e) and not e.get("variants"))}
        for vn in ("T16model_oldfinger", "T16model_T29finger", "E3model_T29finger"):
            V = [e["variants"][vn] for e in L]
            s[vn] = {"n_worst_point_inside_apple": sum(1 for v in V if v.get("inside_apple")),
                     "nearest_body_counts": {b: sum(1 for v in V if v.get("nearest_body") == b) for b in sorted({v.get("nearest_body") for v in V if v.get("nearest_body")})},
                     "worst_link_counts": {b: sum(1 for v in V if (v.get("pair") or {}).get("link") == b) for b in sorted({(v.get("pair") or {}).get("link") for v in V if v.get("pair")})},
                     "worst_tier_counts": {b: sum(1 for v in V if (v.get("pair") or {}).get("tier") == b) for b in sorted({str((v.get("pair") or {}).get("tier")) for v in V if v.get("pair")})},
                     "n_clearance_negative": sum(1 for v in V if v["clearance_mm"] < 0),
                     "clearance_mm_min": min((v["clearance_mm"] for v in V), default=None)}
        s["p12_recorded_worst_nearest_category_counts"] = {c: sum(1 for e in L if e["p12_worst_nearest_category"] == c) for c in sorted({str(e["p12_worst_nearest_category"]) for e in L})}
        s["fidelity_within_tolerance"] = sum(1 for e in L if e["fidelity"].get("within_tolerance"))
        s["n_attached_robot_rows_only"] = sum(1 for e in L if e.get("robot_rows_only_attached_not_recorded"))
        summ[sel_name] = s
    res = {"definitions": __doc__, "settings_T16": sh16.settings.to_json(), "settings_E3": she3.settings.to_json(),
           "hold_chunks": sorted(hold_chunks), "summary": summ, "chunks": out}
    (OUT / "off_0d.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
