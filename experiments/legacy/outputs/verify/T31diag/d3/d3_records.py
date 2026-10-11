"""T31-diag D3 — records side of the crate-representation audit.

usage: MUJOCO_GL=osmesa PYTHONPATH=outputs/verify/T31diag/snap .venv-ag3s/bin/python d3_records.py

Part A (all server chunks of the 4 E3 ep1807 runs + E3a r2): per chunk summary fields — n_candidates, esdf_stats.label_names,
  n_static_shapes, destination {source, n_points, extents_mm}, exclusion.mechanism, esdf_stats.target_free.ball (the target-free
  exclusion ball: centre -> nearest GT object centre), esdf_stats.n_attached_points, attached_sign_correction per tier.
Part B (window chunks grasp_t..release_t, same reconstruction as d2_rows_ablation.py incl. attached points):
  B1 held apple left in the field: GT apple centre (planning-row qpos, MuJoCo apple body) -> base; field value at that point per tier
     (fine / coarse; negative = inside a surface); and the number of voxels per tier with |d| <= voxel/2 or d < 0 within
     GT apple bounding radius (apple_geom rbound 56.4 mm) of the apple centre AND farther than 2 voxels from every other GT body
     (crate, table, other fruit; mj_geomDistance probe) — 'apple-only surface voxels'.
  B2 which tier answers the left-hand queries (ee_left, ee_finger_l1/l2 spheres) and held points on the REFERENCE trajectory
     (8 steps): field.answer_tier; 'coarse' split into coarse_in_window (inside the fine window: coarse value < fine value)
     and coarse_outside_window. Counted for all such queries and for those whose GT surface distance to the crate < 50 mm.
     For near-crate queries answered inside the window: median (d_coarse - d_fine) mm.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, json, pathlib, sys, types, collections
import numpy as np
import mujoco

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T31diag/d2'))
import p12_records as P  # noqa
import d2_rows_ablation as D2  # noqa
import AG3S.trajopt.experiments.fixed_input_ablation as T  # noqa
from AG3S.ag3s.constraints.attached import attach_from_target  # noqa
from AG3S.ag3s.stages.geometry import fit_sphere  # noqa

T30 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d3'))
D1 = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d1/d1_oscillation.json')).read_text())
RUNS = ["E3a_ep1807_r1", "E3a_ep1807_r2", "E3a_ep1807_r3", "E3b_ep1807_r1", "E3b_ep1807_r2"]
WINDOW_RUNS = ["E3a_ep1807_r1", "E3a_ep1807_r3", "E3b_ep1807_r1", "E3b_ep1807_r2"]
lit = D2.lit


def reconstruct_attached(cond, chunks, rows, sh):
    robot = sh["robot"]
    for i, c in enumerate(chunks):
        z = np.load(T30 / f"{cond}_rec" / c["server_chunk"], allow_pickle=False)
        s = json.loads(str(z["summary_json"]))
        gr = s.get("grasp") or {}
        if not gr.get("attach"):
            continue
        want = (gr.get("attached") or {}).get("n_points")
        q_a = T.q_now_from_qpos(rows[c["seq"]]["qpos"], sh["qadr"])
        joints = (s.get("finger_joints") or {}).get("joints") or {}
        for cc in (c, chunks[i - 1] if i else None):
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
            if len(a.points) == want:
                return a
    return None


def main():
    m, d, probe, objg, _ = P.build()
    apple_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    res = {"definitions": __doc__, "A": {}, "B": {}}
    for run in RUNS:
        cond = run.split("_")[0]
        A = json.loads((T30 / "p" / f"{cond}_analysis.json").read_text())["runs"][run]
        recs = []
        for c in A["chunks"]:
            z = np.load(T30 / f"{cond}_rec" / c["server_chunk"], allow_pickle=False)
            s = json.loads(str(z["summary_json"]))
            st = s.get("esdf_stats") or {}
            tf = lit(st.get("target_free")) if st.get("target_free") else None
            ball = tf.get("ball") if isinstance(tf, dict) else None
            dest = s.get("destination") or {}
            recs.append({"t": c["t"], "n_candidates": s.get("n_candidates"), "labels": lit(st.get("label_names")),
                         "n_static_shapes": st.get("n_static_shapes"), "dest_source": dest.get("source"), "dest_n_points": dest.get("n_points"),
                         "dest_extents_mm": dest.get("extents_mm"), "exclusion_mechanism": (s.get("exclusion") or {}).get("mechanism"),
                         "ball": None if not ball else {"radius_mm": ball.get("radius_mm"), "centre_m": ball.get("centre_m")},
                         "ball_nearest_centre": (c.get("ball") or {}).get("centre_nearest"),
                         "n_attached_points": st.get("n_attached_points"), "sign_correction": lit(st.get("attached_sign_correction")),
                         "sphere_n": int(z["sphere_radii"].shape[0]), "has_free_layer": "esdf_free_distance" in z.files})
        res["A"][run] = {
            "n_chunks": len(recs),
            "n_candidates_counts": dict(collections.Counter(str(r["n_candidates"]) for r in recs)),
            "label_names_counts": dict(collections.Counter(str(r["labels"]) for r in recs)),
            "n_static_shapes_counts": dict(collections.Counter(str(r["n_static_shapes"]) for r in recs)),
            "destination_source_counts": dict(collections.Counter(str(r["dest_source"]) for r in recs)),
            "exclusion_mechanism_counts": dict(collections.Counter(str(r["exclusion_mechanism"]) for r in recs)),
            "n_chunks_with_target_free_ball": sum(1 for r in recs if r["ball"]),
            "target_free_ball_nearest_centre_counts": dict(collections.Counter(str(r["ball_nearest_centre"]) for r in recs if r["ball"])),
            "target_free_ball_radius_mm_range": [min([r["ball"]["radius_mm"] for r in recs if r["ball"]] or [None] if False else [r["ball"]["radius_mm"] for r in recs if r["ball"]] or [0]),
                                                 max([r["ball"]["radius_mm"] for r in recs if r["ball"]] or [0])],
            "n_chunks_free_layer_recorded": sum(r["has_free_layer"] for r in recs),
            "constraint_spheres_per_chunk": sorted(set(r["sphere_n"] for r in recs)),
            "n_attached_points_counts": dict(collections.Counter(str(r["n_attached_points"]) for r in recs)),
            "destination_n_points_range": [min([r["dest_n_points"] for r in recs if r["dest_n_points"]] or [0]), max([r["dest_n_points"] for r in recs if r["dest_n_points"]] or [0])],
            "chunks": recs}
        print(run, {k: v for k, v in res["A"][run].items() if k != "chunks"}, flush=True)
    for run in WINDOW_RUNS:
        cond = run.split("_")[0]
        A = json.loads((T30 / "p" / f"{cond}_analysis.json").read_text())["runs"][run]
        d1 = D1["runs"][run]; g, rel = d1["grasp_t"], d1["release_t"]
        run_dir = T30 / f"{cond}_rec" / run
        rows = {int(x["seq"]): x for x in (json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines() if l.strip()) if x.get("kind") == "planning"}
        chunks = [c for c in A["chunks"] if c.get("server_chunk")]
        port = chunks[0]["server_chunk"].split("/")[0].split("_")[1]
        sh = D2.setup(cond, port, run_dir)
        robot, lin, layout = sh["robot"], sh["meter"].lin, sh["layout"]
        names = [str(n) for n in robot.sphere_link_names]; S = len(names)
        hand = [i for i, n in enumerate(names) if n in ("ee_left", "ee_finger_l1", "ee_finger_l2")]
        att = reconstruct_attached(cond, chunks, rows, sh)
        out = []
        for c in chunks:
            if not (g <= c["t"] <= rel):
                continue
            z = np.load(T30 / f"{cond}_rec" / c["server_chunk"], allow_pickle=False)
            s = json.loads(str(z["summary_json"]))
            row = rows[c["seq"]]
            q_now = T.q_now_from_qpos(row["qpos"], sh["qadr"])
            field = T.build_field(z, outside_distance=sh["outside"])
            coarse, fine = field.layers[0], field.layers[1]
            P.set_q(m, d, row["qpos"])
            Rb = d.xmat[base_bid].reshape(3, 3).copy(); tb = d.xpos[base_bid].copy()
            to_base = lambda pw: (np.asarray(pw) - tb) @ Rb
            to_world = lambda pb: np.asarray(pb) @ Rb.T + tb
            ac_w = d.xpos[apple_bid].copy(); ac_b = to_base(ac_w)
            e = {"t": c["t"], "attached": att is not None and int((s.get("esdf_stats") or {}).get("n_attached_points", 0) or 0) > 0}
            e["sign_correction"] = lit((s.get("esdf_stats") or {}).get("attached_sign_correction"))
            # B1
            b1 = {}
            for tn, L in (("fine", fine), ("coarse", coarse)):
                cov = field._covers(L.grid, ac_b[None, :])[0]
                val = float(np.asarray(L.distance(ac_b[None, :]))[0]) if cov else None
                vs = L.grid.voxel_size
                org = np.asarray(L.grid.origin); shp = np.asarray(L.grid.shape)
                lo = np.maximum(np.floor((ac_b - 0.0565 - org) / vs).astype(int), 0); hi = np.minimum(np.ceil((ac_b + 0.0565 - org) / vs).astype(int) + 1, shp)
                n_apple_only = 0; n_surf = 0
                if np.all(hi > lo):
                    sub = np.asarray(L.distance_grid)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                    idx = np.argwhere((np.abs(sub) <= vs / 2) | (sub < 0))
                    for ijk in idx:
                        pb = org + (ijk + lo) * vs
                        if np.linalg.norm(pb - ac_b) > 0.0565:
                            continue
                        n_surf += 1
                        body, cat, per = P.nearest(m, d, probe, objg, to_world(pb))
                        if per["apple"] <= vs and all(v > 2 * vs for b, v in per.items() if b != "apple"):
                            n_apple_only += 1
                    P.set_q(m, d, row["qpos"])
                b1[tn] = {"value_at_gt_apple_centre_mm": None if val is None else 1000 * val, "covers": bool(cov),
                          "n_surface_or_inside_voxels_within_56mm": n_surf, "n_apple_only_voxels": n_apple_only}
            e["held_apple_in_field"] = b1
            # B2
            fj = s.get("finger_joints") or {}
            refc = np.asarray(z["reference_chunk"], np.float64)
            Pp, _ = sh["fmap"].path(fj.get("opening_norm") or {}, {"left": refc[:, 7], "right": refc[:, 15]}, lin.horizon, default=robot.param_vector())
            lin.set_joint_parameter_path(Pp)
            if e["attached"]:
                lin.set_attached(att.points, "ee_finger_l1")
            else:
                lin.set_attached(None, None)
            traj = layout.chunk_to_trajectory(refc)[:, :lin.horizon]
            centres = lin.sphere_states(traj, q_now)[0]
            qidx = hand + list(range(S, centres.shape[1]))
            pts = centres[:, qidx].reshape(-1, 3)
            kinds = np.array(([("sphere") for _ in hand] + ["held"] * (centres.shape[1] - S)) * centres.shape[0])
            tiers = np.asarray(field.answer_tier(pts)).astype("<U24")
            infine = field._covers(fine.grid, pts)
            dc = np.asarray(coarse.distance(pts)); dfv = np.full(len(pts), np.nan)
            if infine.any():
                dfv[infine] = np.asarray(fine.distance(pts[infine]))
            tiers[(tiers == "coarse") & infine] = "coarse_in_window"
            tiers[(tiers == "coarse") & ~infine] = "coarse_outside_window"
            crate_d = np.empty(len(pts))
            crate_geoms = objg["crate"]
            for i, pb in enumerate(pts):
                d.mocap_pos[P.PROBE_MOCAP] = to_world(pb)
                mujoco.mj_kinematics(m, d)
                crate_d[i] = min(P.gdist(m, d, probe, gg) for gg in crate_geoms)
            P.set_q(m, d, row["qpos"])
            near = crate_d < 0.05
            b2 = {"n_queries": int(len(pts)), "tier_counts_all": dict(collections.Counter(tiers.tolist())),
                  "tier_counts_near_crate_lt50mm": dict(collections.Counter(tiers[near].tolist())),
                  "tier_counts_near_crate_held": dict(collections.Counter(tiers[near & (kinds == "held")].tolist())),
                  "tier_counts_near_crate_spheres": dict(collections.Counter(tiers[near & (kinds == "sphere")].tolist())),
                  "n_near_crate": int(near.sum()), "min_crate_dist_mm": 1000 * float(crate_d.min())}
            sel = near & infine
            if sel.any():
                b2["near_crate_in_window_median_coarse_minus_fine_mm"] = 1000 * float(np.median(dc[sel] - dfv[sel]))
                b2["near_crate_in_window_n_coarse_lt_fine"] = int((dc[sel] < dfv[sel]).sum())
                b2["near_crate_in_window_n"] = int(sel.sum())
            e["query_tiers_reference"] = b2
            lin.set_joint_parameter_path(None); lin.set_attached(None, None)
            out.append(e)
            print(run, c["t"], "apple@field fine", b1["fine"]["value_at_gt_apple_centre_mm"] and round(b1["fine"]["value_at_gt_apple_centre_mm"], 1), b1["fine"]["n_apple_only_voxels"],
                  "coarse", b1["coarse"]["value_at_gt_apple_centre_mm"] and round(b1["coarse"]["value_at_gt_apple_centre_mm"], 1), b1["coarse"]["n_apple_only_voxels"],
                  "tiers", b2["tier_counts_all"], "near", b2["n_near_crate"], b2["tier_counts_near_crate_lt50mm"], round(b2["min_crate_dist_mm"], 1), flush=True)
        res["B"][run] = {"window_t": [g, rel], "n_attached_points_reconstructed": None if att is None else len(att.points),
                         "attached_bounding_sphere_radius_mm": None if att is None else 1000 * float(att.primitives[0].dimensions[0]),
                         "attached_n_primitives": None if att is None else len(att.primitives), "chunks": out}
        (OUT / "d3_records.json").write_text(json.dumps(res, indent=1, default=float))
    res["n_invalid_geomdistance"] = P.N_INVALID[0]
    (OUT / "d3_records.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
