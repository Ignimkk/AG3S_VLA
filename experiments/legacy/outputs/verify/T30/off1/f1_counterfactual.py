"""T30 #1 — F1 (+F1b) counterfactual on the pre-T30 E3 server records (T28 Phase 3 E3a/E3b).

Inputs restored with AG3S.trajopt.experiments.fixed_input_ablation (T24 tool, working tree = T30):
  settings        : T.parse_server_log(<cond>_rec/serve_safe_<port>.log) + run manifest (safe_phase approach, manipulators left)
  robot model     : T.build_robot (links gripper, spacing 0.4, max 32, capsule_radius_scale 0.05) — 122 constraint spheres
  field           : T.build_field(chunk npz) — coarse + fine + target-free (esdf_free_*) layers as recorded
  contact policy  : T.contact_policy (ClearancePolicy x phase approach, manipulators left, exclude_authorized) — margin + mask
  q_now           : local planning row paired to that server chunk (T28 p3/<COND>_analysis.json 'server_chunk', exact actions match)
  finger joints   : FingerJointMap.path(measured = summary.finger_joints.opening_norm, commanded = reference_chunk cols 7/15,
                    horizon 8, default = model param vector) — same call as SafePolicy._apply_finger_state (T29)
Rules:
  new  : T.make_scene(...) as is  -> SceneSnapshot.target_free_exempt = None = target_free_mask (T30 F1)
  old  : dataclasses.replace(new_scene, target_free_exempt=np.zeros(S, bool))  (pre-T30 rule, as the tests do)
Trajectories: reference = reference_chunk[:, :8] (TO initial iterate), refined = recorded refined_chunk[:, :8] (TO output).
Per trajectory x rule: linearizer.worst_row (value + row identity incl. tier) and the full (8, S) esdf_clearance.
  violated            : worst clearance < 0 (same sign test as the recorded to.metrics.clearance_m < 0 / initial_violation_m > 0)
  target-margin row   : a row whose margin is the manipulated-object margin, recomputed here with the linearizer's own test
                        (|d_eff - d_object| <= voxel, d_object = KD-tree distance to recorded target_points, row in the
                        manipulated_link_margin set with margin != esdf_margin). Under the new rule exempt rows are removed first.
                        The recorded label layer is not in the npz, so this stands in for the recorded obstacle == 'target'.
  target_free.target  : worst row with tier == 'target_free' and target-margin row True
  GT object           : worst row query sphere centre (base) -> MuJoCo world (base pose from the paired planning row qpos) ->
                        nearest non-robot collision geom by mj_geomDistance from a 0.1 mm probe (T28 P1-2 helper)
Fidelity (old rule, refined): reconstructed worst_row clearance vs recorded to.metrics.clearance_m, and the same (step, query) row.
Attached chunks (esdf_stats.n_attached_points > 0) are skipped (held-object query points are not recorded).
usage: f1_counterfactual.py COND   (COND = E3a | E3b)
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, dataclasses, json, pathlib, sys, time
import numpy as np

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P  # noqa: E402
import AG3S.trajopt.experiments.fixed_input_ablation as T  # noqa: E402
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap  # noqa: E402

COND = sys.argv[1]
T28 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28'))
REC = T28 / f"{COND}_rec"
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30/off1'))
ANA = json.loads((T28 / "p3" / f"{COND}_analysis.json").read_text())


def pair_of(m):
    p = m.get("max_violation_pair")
    return ast.literal_eval(p) if isinstance(p, str) else p


def setup(port, run_dir):
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.types import ChunkLayout
    argv = ["--constraints", str(REC / f"server_{port}/server_constraints/run_0000"), "--frames", str(run_dir / "frames.jsonl"),
            "--server-log", str(REC / f"serve_safe_{port}.log"), "--out", "/dev/null"]
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
    return dict(settings=settings, robot=robot, qadr=qadr, layout=layout, meter=meter, outside=outside,
                fmap=FingerJointMap(robot))


def object_rows(lin, traj, q_now, scene):
    """(H, S) bool — rows that carry the manipulated-object margin under this scene's rule (linearizer's own test)."""
    centres = lin.sphere_states(traj, q_now)[0]
    flat = centres.reshape(-1, 3)
    d = np.asarray(scene.esdf.distance(flat), np.float64).reshape(centres.shape[:2])
    S = centres.shape[1]
    if scene.target_free_mask is not None:
        mask = np.asarray(scene.target_free_mask, bool)[:S]
        d_free = np.asarray(scene.esdf.target_free_distance(flat), np.float64).reshape(centres.shape[:2])
        d = np.where(mask[None, :], d_free, d)
    if scene.manipulated_link_margin is None or scene.manipulated_points is None or not len(scene.manipulated_points):
        return np.zeros(centres.shape[:2], bool)
    from scipy.spatial import cKDTree
    d_obj = cKDTree(scene.manipulated_points).query(flat)[0].reshape(centres.shape[:2])
    is_obj = np.abs(d - d_obj) <= float(scene.esdf.grid.voxel_size)
    ex = lin._target_free_exempt_rows(scene, S)
    if ex is not None:
        is_obj &= ~ex[None, :]
    per = np.asarray(scene.manipulated_link_margin, np.float64)[:S]
    return is_obj & (per[None, :] != float(scene.esdf_margin))


def main():
    m, d, probe, objg, _ = P.build()
    shared = {}
    out = {"condition": COND, "definitions": __doc__, "chunks": []}
    t0 = time.time()
    for rname, r in ANA["runs"].items():
        run_dir = REC / rname
        rows = {int(x["seq"]): x for x in (json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines() if l.strip()) if x.get("kind") == "planning"}
        for c in r["chunks"]:
            sc = c.get("server_chunk")
            if not sc:
                continue
            port = sc.split("/")[0].split("_")[1]
            if port not in shared:
                shared[port] = setup(port, run_dir)
            sh = shared[port]
            lin, st = sh["meter"].lin, sh["settings"]
            f = REC / sc
            z = np.load(f, allow_pickle=False)
            s = json.loads(str(z["summary_json"]))
            e = {"run": rname, "t": c["t"], "seq": c["seq"], "phase": c["phase"], "file": sc,
                 "latch": (c.get("grasp_latch") or {}).get("state")}
            to = s.get("to") or {}
            tm = to.get("metrics") or {}
            rp = pair_of(tm) or {}
            e["recorded"] = {"clearance_m": tm.get("clearance_m"), "initial_violation_m": tm.get("initial_violation_m"),
                             "tier": rp.get("tier"), "obstacle": rp.get("obstacle"), "link": rp.get("link"),
                             "step": rp.get("step"), "query": rp.get("query")}
            attached = int((s.get("esdf_stats") or {}).get("n_attached_points", 0) or 0) > 0
            if attached:
                e["skipped"] = "attached"
                out["chunks"].append(e)
                continue
            row = rows[c["seq"]]
            q_now = T.q_now_from_qpos(row["qpos"], sh["qadr"])
            field = T.build_field(z, outside_distance=sh["outside"])
            tp = z["target_points"] if "target_points" in z.files else None
            has_t = bool(s.get("has_target")) and tp is not None
            margin, mask, _, _, _ = T.contact_policy(st, sh["robot"].sphere_link_names, has_target=has_t,
                                                     has_free=bool(field.target_free_layers), attached=False)
            new = T.make_scene(field, lin.robot_radii, st.esdf_margin, target_points=tp, margin=margin, mask=mask)
            old = dataclasses.replace(new, target_free_exempt=np.zeros(lin.robot_radii.size, bool))
            refc = np.asarray(z["reference_chunk"], np.float64)
            fj = s.get("finger_joints") or {}
            meas = (fj.get("opening_norm") or {})
            Pp, _env = sh["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, lin.horizon,
                                       default=sh["robot"].param_vector())
            lin.set_joint_parameter_path(Pp)
            e["finger_measured"] = meas
            H = lin.horizon
            trajs = {"reference": sh["layout"].chunk_to_trajectory(refc)[:, :H],
                     "refined": sh["layout"].chunk_to_trajectory(np.asarray(z["refined_chunk"], np.float64))[:, :H]}
            names = [str(n) for n in sh["robot"].sphere_link_names]
            for tn, traj in trajs.items():
                for rn, scene in (("old", old), ("new", new)):
                    cval, pair = lin.worst_row(traj, q_now, scene)
                    cl = lin.esdf_clearance(traj, q_now, scene)[..., 0]
                    isobj = object_rows(lin, traj, q_now, scene)
                    maskv = np.zeros(cl.shape[1], bool) if scene.target_free_mask is None else np.asarray(scene.target_free_mask, bool)
                    neg = cl < 0
                    v = {"worst_clearance_mm": 1000 * float(cval), "violated": bool(cval < 0),
                         "n_neg_rows": int(neg.sum()),
                         "n_neg_rows_target_free_target": int((neg & isobj & maskv[None, :]).sum()),
                         "n_neg_rows_target_free": int((neg & maskv[None, :]).sum())}
                    if pair:
                        k, q = int(pair["step"]), int(pair["query"])
                        v.update({"link": pair["link"], "tier": pair["tier"], "step": k, "query": q,
                                  "target_margin_row": bool(isobj[k, q]), "mask_row": bool(maskv[q])})
                        P.set_q(m, d, row["qpos"])
                        pw = P.base_to_world(m, d, pair["point_m"])
                        body, cat, per = P.nearest(m, d, probe, objg, pw)
                        P.set_q(m, d, row["qpos"])
                        v.update({"gt_nearest": body, "gt_nearest_surface_mm": 1000 * per[body], "apple_surface_mm": 1000 * per["apple"]})
                    e[f"{tn}_{rn}"] = v
            # fidelity (old rule, refined) vs recorded TO view
            if tm.get("clearance_m") is not None:
                e["fidelity_refined_old_abs_diff_mm"] = abs(e["refined_old"]["worst_clearance_mm"] - 1000 * tm["clearance_m"])
                e["fidelity_same_row"] = (e["refined_old"].get("step") == rp.get("step") and e["refined_old"].get("query") == rp.get("query"))
            if tm.get("initial_violation_m") is not None:
                e["fidelity_reference_old_initial_violation_abs_diff_mm"] = abs(max(0.0, -e["reference_old"]["worst_clearance_mm"]) - 1000 * tm["initial_violation_m"])
            lin.set_joint_parameter_path(None)
            out["chunks"].append(e)
        print(rname, len(out["chunks"]), f"{time.time() - t0:.0f}s", flush=True)
    out["n_invalid_geomdistance"] = P.N_INVALID[0]
    (OUT / f"f1_{COND}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
