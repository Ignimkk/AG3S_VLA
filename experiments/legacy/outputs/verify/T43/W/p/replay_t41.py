"""T41a — TO row classes per phase + TO-predicted vs true-mesh hand clearance at the closest-approach chunk (verifier, CPU only).

usage: T41_SERVER_DIR=<T41a server dir> T41_MANIFEST=<any T41a run manifest.json> \
       replay_t41.py ANALYSIS_JSON OUT_JSON [RUN_KEY ...]
       (ANALYSIS_JSON = analyze_t41.py output; RUN_KEY like E3b/C/E3b_ep1807_s18071; default all E3b runs with a server segment)

Per paired server chunk (analyze_t41 pairing: server t_step == planning seq, refined_chunk == planning actions):
  scene     : as the server built it, rebuilt from the record with fixed_input_ablation (T40 I method): field = recorded coarse /
              fine / target-free grids; contact policy (manipulated_link_margin, target_free_mask) = FIA.contact_policy;
              not held -> manipulated_points = recorded target_points (when has_target);
              held (summary.grasp.attached_in_constraints) -> manipulated_spheres = summary.grasp.attached.held.spheres_parent_frame
              placed with the URDF pose of grasp.attached.parent_link at q_now (static over the window, as SceneSnapshot);
              support_planes = AG3S fit_support_surfaces(filtered_cloud, AG3SConfig().support_surface) — recomputed; the recomputed
              inlier mask is compared with the recorded support_mask (support_mask_equal);
              obstacle_margin = 0.010, obstacle_margin_support False (server flags).
  rows      : robot query rows only = plan steps (meter horizon, 8) x 122 spheres on the recorded REFINED trajectory, finger
              path FingerJointMap.path(summary.finger_joints.opening_norm, reference gripper columns). Held-object query rows
              (attached spheres, class 'held' by definition) are NOT rebuilt; their count = 8 x grasp.attached.n_spheres is reported.
  class     : CollisionLinearizer._last_obstacle_class after esdf_clearance with the margin on (obstacle / target / support).
  clear_off : the same rows with obstacle_margin 0 (= the field clearance the margin floor acts on). 'margin binding' = obstacle
              row with clear_off < 10 mm.
  fidelity  : reconstructed worst_row (scene with margin) vs recorded summary.to.metrics clearance_m / max_violation_pair
              (incl. obstacle_margin_class); recorded worst rows on held-object queries (query >= 122) cannot match.
Phases (by the chunk's first control step t, analysis hurdle lift_t / place_t): before_lift t < lift_t; carry lift_t <= t <= place_t
  (or end); after t > place_t. No grasp -> all before_lift.
Closest approach: control row t* of the run's minimum gripper (EE_BODY_L + fingers, true MuJoCo mesh) to hurdle distance
  (analysis hurdle series A, all rows); chunk = last planning row with t_step <= t*. There, per hand link (URDF ee_left,
  ee_finger_l1, ee_finger_l2; MuJoCo bodies by T40 I geometric mapping):
  TO_view_mm      : min over the 8 refined plan steps of the TO clearance (margin on) and of clear_off;
  TO_view_class   : class of that row;
  sphere_true_mm  : analytic distance URDF query spheres -> true hurdle primitives (all hurdle geoms) on the refined plan;
  mesh_true_mm    : MuJoCo mesh distance (pair_distance, fixed GJK client code) of the link's collision geoms -> hurdle geoms,
                    on the refined plan (fingers as at the planning row) and on the executed control rows of that chunk."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib, dataclasses, datetime
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T41/p'))
import numpy as np
import mi_lib as MI  # noqa: E402  (setup = server's constraint model; asserts the t41a snapshot)
import hurdle_metrics as HM  # noqa: E402
from AG3S.trajopt.experiments import fixed_input_ablation as FIA  # noqa: E402
from AG3S.trajopt import linearize as LZ  # noqa: E402
from AG3S.ag3s.stages.support_surface import fit_support_surfaces  # noqa: E402
from AG3S.ag3s.types import PointCloud  # noqa: E402
from AG3S.ag3s.config import AG3SConfig  # noqa: E402

assert LZ.__file__.startswith("${AG3S_PROJECT_ROOT}-t41a/"), LZ.__file__
SSCFG = AG3SConfig().support_surface
OM = 0.010
CLS = list(LZ.OBSTACLE_CLASSES)
HAND = ("ee_left", "ee_finger_l1", "ee_finger_l2")
NAMES, RADII, H = MI.NAMES, MI.RADII, MI.H
HANDMASK = np.isin(NAMES, HAND)


def planes_of(z):
    fc = np.asarray(z["filtered_cloud"], np.float64)
    surf, mask = fit_support_surfaces(PointCloud(points=fc), SSCFG, timestamp=0.0)
    P = np.asarray([[*np.asarray(s.normal, float), float(s.offset)] for s in surf], float).reshape(-1, 4)
    eq = bool(mask.shape == z["support_mask"].shape and (mask == z["support_mask"]).all())
    return P, eq


def chunk_eval(z, prow):
    summ = json.loads(str(z["summary_json"]))
    q = np.array(prow["qpos"]); q_now = FIA.q_now_from_qpos(q, MI.qadr)
    ref_c = np.asarray(z["reference_chunk"], float); fin_c = np.asarray(z["refined_chunk"], float)
    fj = summ.get("finger_joints") or {}
    P, _ = MI.fmap.path(fj.get("opening_norm") or {}, {"left": ref_c[:, 7], "right": ref_c[:, 15]}, H, default=MI.DEFAULT_P)
    MI.meter.lin.set_joint_parameter_path(P)
    MI.robot.set_joint_parameters(dict(zip(MI.PARAM_NAMES, P[:, 0])))
    traj = MI.layout.chunk_to_trajectory(fin_c)[:, :H]
    fieldc = FIA.build_field(z, MI.OUTSIDE)
    g = summ.get("grasp") or {}
    att = g.get("attached") if g.get("attached_in_constraints") else None
    held_sp = (att or {}).get("held", {}).get("spheres_parent_frame") if att else None
    has_target = bool(summ.get("has_target")) and "target_points" in z.files and len(z["target_points"]) > 0
    margin, mask, _, _, _ = FIA.contact_policy(MI.settings, MI.robot.sphere_link_names, has_target=has_target,
                                                has_free=bool(fieldc.target_free_layers), attached=att is not None)
    scene = FIA.make_scene(fieldc, RADII, MI.settings.esdf_margin, target_points=z["target_points"] if has_target else None,
                           margin=margin, mask=mask)
    extra = {}
    if att is not None and held_sp:
        T = np.asarray(MI.robot.link_pose(q_now, att.get("parent_link") or "ee_left", P[:, 0]), float)
        C = np.array([T[:3, :3] @ np.asarray(s["centre_m"], float) + T[:3, 3] for s in held_sp])
        extra = dict(manipulated_points=None, manipulated_spheres=C,
                     manipulated_sphere_radii=np.array([s["radius_mm"] / 1000.0 for s in held_sp]))
    elif has_target:
        extra = dict(manipulated_points=np.asarray(z["target_points"], np.float64))
    planes, mask_eq = planes_of(z)
    s_off = dataclasses.replace(scene, **extra)
    s_on = dataclasses.replace(s_off, obstacle_margin=OM, obstacle_margin_support=False, support_planes=planes)
    c_off = MI.meter.lin.esdf_clearance(traj, q_now, s_off)[..., 0]
    c_on = MI.meter.lin.esdf_clearance(traj, q_now, s_on)[..., 0]
    cls = np.asarray(MI.meter.lin._last_obstacle_class)[:, :len(NAMES)].copy()
    wc, wp = MI.meter.lin.worst_row(traj, q_now, s_on)
    met = ((summ.get("to") or {}).get("metrics") or {})
    rp = FIA._parse_pair(met.get("max_violation_pair"))
    fid = {"recorded_clearance_m": met.get("clearance_m"), "reconstructed_clearance_m": float(wc),
           "recorded_pair": {k: (rp or {}).get(k) for k in ("step", "query", "link", "tier", "obstacle", "obstacle_margin_class")},
           "reconstructed_pair": {k: (wp or {}).get(k) for k in ("step", "query", "link", "tier", "obstacle_margin_class")}}
    if met.get("clearance_m") is not None:
        fid["abs_diff_mm"] = abs(float(wc) - float(met["clearance_m"])) * 1000
    fid["same_row"] = bool(rp and wp and int(rp.get("step", -1)) == int(wp["step"]) and int(rp.get("query", -1)) == int(wp["query"]))
    fid["recorded_worst_is_held_query"] = bool(rp and int(rp.get("query", -1)) >= len(NAMES))
    return dict(summ=summ, q=q, q_now=q_now, traj=traj, P=P, c_on=c_on, c_off=c_off, cls=cls, fid=fid, support_mask_equal=mask_eq,
                n_planes=int(len(planes)), held=att is not None, n_held_spheres=int((att or {}).get("n_spheres") or 0),
                has_target=has_target)


def counts(e):
    cls, off = e["cls"], e["c_off"]
    out = {"rows": int(cls.size), "held_query_rows_not_rebuilt": 8 * e["n_held_spheres"]}
    for k, nm in enumerate(CLS):
        if nm == "held":
            continue
        out[nm] = int((cls == k).sum()); out[nm + "_hand"] = int((cls[:, HANDMASK] == k).sum())
    ob = cls == CLS.index("obstacle")
    out["obstacle_lt10mm"] = int((ob & (off < OM)).sum()); out["obstacle_lt10mm_hand"] = int((ob[:, HANDMASK] & (off[:, HANDMASK] < OM)).sum())
    out["obstacle_lt0mm"] = int((ob & (off < 0)).sum())
    return out


def closest(e, mg, hshapes, ctrl_rows):
    res = {}
    Cpath = MI.spheres_path(e["traj"], e["q_now"], e["P"])  # (H+1?, S, 3) base frame
    Cpath = Cpath[-e["traj"].shape[1]:]
    q = e["q"]
    ft = np.zeros(6); Hg = sorted(mg.collision_geom_ids)

    def mesh_link(qp):
        MI.mj_set(qp)
        return {ln: MI.mm(min(mg.pair_distance(int(o), g, 0.30, ft) for g, l in MI.GEOM_LINK.items() if l == ln for o in Hg)) for ln in HAND}
    plan_mesh = [mesh_link(MI.qpos_with_traj(q, e["traj"][:, k])) for k in range(e["traj"].shape[1])]
    exe_mesh = [dict(t=int(c["t_step"]), **mesh_link(np.array(c["qpos"]))) for c in ctrl_rows]
    Cw = np.stack([MI.to_world(Cpath[k], q) for k in range(len(Cpath))])
    for ln in HAND:
        sl = NAMES == ln
        on, off, cl = e["c_on"][:, sl], e["c_off"][:, sl], e["cls"][:, sl]
        i = np.unravel_index(int(np.argmin(on)), on.shape)
        st = np.array([[HM.point_to_hurdle(Cw[k, j], hshapes) for j in np.where(sl)[0]] for k in range(len(Cw))]) - RADII[sl][None]
        res[ln] = {"TO_view_on_min_mm": float(on.min() * 1000), "TO_view_on_step": int(i[0]), "TO_view_class": CLS[int(cl[i])],
                   "TO_view_off_min_mm": float(off.min() * 1000),
                   "TO_view_on_min_per_step_mm": (on.min(axis=1) * 1000).tolist(),
                   "sphere_true_hurdle_min_mm": float(st.min() * 1000), "sphere_true_hurdle_min_per_step_mm": (st.min(axis=1) * 1000).tolist(),
                   "mesh_true_hurdle_min_mm_refined_plan": min(p[ln] for p in plan_mesh),
                   "mesh_true_hurdle_per_step_mm_refined_plan": [p[ln] for p in plan_mesh],
                   "mesh_true_hurdle_min_mm_executed_rows": min((x[ln] for x in exe_mesh), default=None)}
    res["executed_rows_mesh_mm"] = exe_mesh
    return res


def main():
    A = json.load(open(sys.argv[1]))["runs"]; out_p = pathlib.Path(sys.argv[2])
    keys = sys.argv[3:] or [k for k, r in A.items() if k.startswith("E3b/") and r.get("server_segment")]
    MI.build_mapping()
    res = json.loads(out_p.read_text()) if out_p.exists() else {"definitions": __doc__, "runs": {}}
    for key in keys:
        r = A[key]; sg = r["server_segment"]; h = r.get("hurdle") or {}
        rd = pathlib.Path(r["run_dir"])
        rows = [json.loads(l) for l in open(rd / "frames.jsonl")]
        plan = {x["seq"]: x for x in rows if x["kind"] == "planning"}
        ctrl = [x for x in rows if x["kind"] == "control"]
        base = pathlib.Path(sg["first"]).parent; i0 = int(sg["first"][-9:-4]); i1 = int(sg["last"][-9:-4])
        lift, place = h.get("lift_t"), h.get("place_t")
        if not h and r.get("grasp_t") is not None:  # no hurdle config (B ep1828): same lift rule as hurdle_metrics.run_hurdle
            tt = np.array([c["t_step"] for c in ctrl]); zz = np.array([c["qpos"][40] for c in ctrl])
            z0 = plan[min(plan)]["object_poses"]["apple"]["pos"][2]
            k = min(int(np.searchsorted(tt, r["grasp_t"])), len(tt) - 1)
            while k > 0 and zz[k - 1] >= z0 + 0.002:
                k -= 1
            lift, place = int(tt[k]), r.get("place_t")
        ser = h.get("series") or {}
        tstar = None
        if ser.get("A"):
            tstar = int(ser["t"][int(np.argmin(ser["A"]))])
        mg = HM.manager(h["config"], h["profile"]) if h else None
        if mg is not None:
            MI.mj_set(np.array(plan[min(plan)]["qpos"]))
            hshapes = HM.hurdle_shapes(mg)
        per_chunk = []; phase = {}
        for i in range(i0, i1 + 1):
            z = np.load(base / f"chunk_{i:05d}.npz", allow_pickle=True)
            seq = int(z["t_step"]); p = plan.get(seq)
            if p is None:
                continue
            e = chunk_eval(z, p); t = int(p["t_step"])
            ph = "before_lift" if (lift is None or t < lift) else ("carry" if (place is None or t <= place) else "after")
            c = counts(e)
            rec = {"seq": seq, "t": t, "phase": ph, "held": e["held"], "has_target": e["has_target"], "n_planes": e["n_planes"],
                   "support_mask_equal": e["support_mask_equal"], "counts": c, "fidelity": e["fid"],
                   "gate": (p.get("gate") or {}).get("action"), "hold_kinds": (p.get("gate") or {}).get("hold_kinds")}
            if tstar is not None and mg is not None and seq == max(s for s in plan if plan[s]["t_step"] <= tstar):
                rec["closest_approach"] = {"t_star": tstar, **closest(e, mg, hshapes, [x for x in ctrl if x["chunk_seq"] == seq])}
            per_chunk.append(rec)
            agg = phase.setdefault(ph, {"chunks": 0})
            agg["chunks"] += 1
            for k, v in c.items():
                agg[k] = agg.get(k, 0) + v
            z.close()
        fids = [x["fidelity"] for x in per_chunk]
        res["runs"][key] = {"run_dir": str(rd), "lift_t": lift, "place_t": place, "t_star_min_gripper_hurdle": tstar,
                            "by_phase": phase, "chunks": per_chunk,
                            "fidelity_summary": {"n": len(fids), "same_row": sum(f["same_row"] for f in fids),
                                                 "recorded_worst_held_query": sum(f["recorded_worst_is_held_query"] for f in fids),
                                                 "abs_diff_mm_max_non_held": max([f.get("abs_diff_mm", 0) for f in fids if not f["recorded_worst_is_held_query"]], default=None),
                                                 "abs_diff_mm_le_0p01_non_held": sum(1 for f in fids if not f["recorded_worst_is_held_query"] and f.get("abs_diff_mm", 1) <= 0.01),
                                                 "class_match_where_recorded": sum(1 for f in fids if f["recorded_pair"].get("obstacle_margin_class") is not None and f["same_row"] and f["recorded_pair"]["obstacle_margin_class"] == f["reconstructed_pair"].get("obstacle_margin_class")),
                                                 "n_recorded_class": sum(1 for f in fids if f["recorded_pair"].get("obstacle_margin_class") is not None),
                                                 "support_mask_equal": sum(x["support_mask_equal"] for x in per_chunk)}}
        print(key, json.dumps(res["runs"][key]["fidelity_summary"]), json.dumps(phase), flush=True)
        res["generated"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        out_p.write_text(json.dumps(MI.jl(res), indent=1))


if __name__ == "__main__":
    main()
