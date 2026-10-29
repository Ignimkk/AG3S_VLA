"""T35-diag D — what the field has at the reference worst rows of the chunks where collision drives the TO shift.

usage: MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} <py> d_probe.py -> d/d_probe.json
For each (run, t) listed: rebuild the chunk's field and scene exactly as b_ablation.py (held spheres for held chunks), take every
reference row with clearance < 0 (and the worst row), and report: link, step, tier, clearance (mm), field distance d(c), the
field nearest-surface estimate p = c - d(c) * grad/|grad| (base), GT surface distance per body at p (P1-2 probe) -> nearest GT
body at p and its distance (mm), height of p above the GT apple top (apple centre z + 34 mm) and above the table plane
(support_mask TLS plane, z > 0.5 m points), and the GT sphere-centre surface distances (apple / banana / table, minus r).
Also the recorded field layers present (esdf_free recorded?) and esdf_extra.held_free (voxels freed).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T35diag'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import b_ablation as B  # noqa
import p12_records as P  # noqa
import AG3S.trajopt.experiments.fixed_input_ablation as T  # noqa
from AG3S.trajopt.linearize import SceneSnapshot  # noqa

V = B.V
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T35diag/d')); OUT.mkdir(parents=True, exist_ok=True)
CASES = [("E3b_ep1800_r1", 144), ("E3b_ep1800_r1", 152), ("E3b_ep1800_r2", 152), ("E3a_ep1800_r1", 144), ("E3b_ep1808_r1", 112),
         ("E3a_ep1800_r1", 136), ("E3b_ep1800_r1", 112)]


def main():
    m, d, probe, objg, _ = P.build()
    res = {"definitions": __doc__, "cases": []}
    shs = {}
    for run, t in CASES:
        cond = run.split("_")[0]
        A = json.loads((V / "p" / f"{cond}_analysis.json").read_text())["runs"][run]
        c = next(x for x in A["chunks"] if x["t"] == t and x.get("server_chunk"))
        run_dir = V / f"{cond}_rec" / run
        rows = {int(x["seq"]): x for x in (json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines() if l.strip())
                if x.get("kind") == "planning"}
        if run not in shs:
            shs[run] = B.setup(cond, run_dir)
        sh = shs[run]
        robot, lin, layout, st = sh["robot"], sh["meter"].lin, sh["layout"], sh["settings"]
        names = [str(n) for n in robot.sphere_link_names]; S = len(names)
        z = np.load(V / f"{cond}_rec" / c["server_chunk"], allow_pickle=False)
        s = json.loads(str(z["summary_json"]))
        row = rows[c["seq"]]
        q_now = T.q_now_from_qpos(row["qpos"], sh["qadr"])
        field = T.build_field(z, outside_distance=sh["outside"])
        tp = z["target_points"] if "target_points" in z.files else None
        gr = s.get("grasp") or {}
        held = (gr.get("attached") or {}).get("held") if gr.get("attached_in_constraints") else None
        fj = s.get("finger_joints") or {}
        refc = np.asarray(z["reference_chunk"], np.float64)
        Pp, _ = sh["fmap"].path(fj.get("opening_norm") or {}, {"left": refc[:, 7], "right": refc[:, 15]}, lin.horizon, default=robot.param_vector())
        if held:
            link = held["frame_link"]
            cl_ = np.array([x["centre_m"] for x in held["spheres_parent_frame"]], float)
            rr_ = np.array([x["radius_mm"] / 1000.0 for x in held["spheres_parent_frame"]], float)
            Tq = np.asarray(robot.link_pose(q_now, link), float)
            margin, _, _, _, _ = T.contact_policy(st, robot.sphere_link_names, has_target=True, has_free=False, attached=False)
            scene = SceneSnapshot(candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0), candidate_active=np.zeros(0, bool),
                                  d_safe=np.zeros((S, 0)), plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0),
                                  plane_active=np.zeros(0, bool), robot_radii=lin.robot_radii, candidate_ids=np.zeros(0, np.int64),
                                  esdf=field, esdf_margin=float(st.esdf_margin), manipulated_spheres=(Tq[:3, :3] @ cl_.T).T + Tq[:3, 3],
                                  manipulated_sphere_radii=rr_, manipulated_link_margin=margin,
                                  attached_points=cl_, attached_parent_link=link, attached_radii=rr_)
            lin.set_attached(cl_, link, rr_)
        else:
            margin, mask, _, _, _ = T.contact_policy(st, robot.sphere_link_names, has_target=bool(s.get("has_target")) and tp is not None,
                                                     has_free=bool(field.target_free_layers), attached=False)
            scene = T.make_scene(field, lin.robot_radii, st.esdf_margin, target_points=tp, margin=margin, mask=mask)
            lin.set_attached(None, None)
        lin.set_joint_parameter_path(Pp)
        traj = layout.chunk_to_trajectory(refc)[:, :lin.horizon]
        cl = lin.esdf_clearance(traj, q_now, scene)[..., 0]
        centres = lin.sphere_states(traj, q_now)[0]
        radii = np.asarray(lin.query_radii if hasattr(lin, "query_radii") else lin.robot_radii, float)
        tiers = np.asarray(field.answer_tier(centres.reshape(-1, 3))).reshape(cl.shape)
        pts = np.asarray(z["raw_cloud"], float)[np.asarray(z["support_mask"], bool)]
        tab = pts[pts[:, 2] > 0.5]; c0 = tab.mean(0); nrm = np.linalg.svd(tab - c0, full_matrices=False)[2][-1]
        nrm = nrm if nrm[2] > 0 else -nrm; off = float(nrm @ c0)
        P.set_q(m, d, row["qpos"])
        apple_w = np.asarray(row["qpos"][38:41], float)
        neg = sorted([tuple(x) for x in np.argwhere(cl < 0).tolist()], key=lambda kq: cl[kq])
        if not neg:
            k0 = np.unravel_index(np.argmin(cl), cl.shape); neg = [tuple(int(v) for v in k0)]
        out = []
        for (k, q) in neg[:10]:
            cc = centres[k, q]
            dc = float(np.asarray(field.distance(cc[None]), float)[0]); gv = np.asarray(field.gradient(cc[None]), float)[0]
            gn = float(np.linalg.norm(gv))
            r_q = float(radii[q]) if q < len(radii) else float("nan")
            P.set_q(m, d, row["qpos"])
            _, _, per_c = P.nearest(m, d, probe, objg, P.base_to_world(m, d, cc))
            e = {"step": int(k), "query": int(q), "link": names[q] if q < S else f"attached[{q - S}]", "tier": str(tiers[k, q]),
                 "clearance_mm": 1000 * float(cl[k, q]), "field_d_mm": 1000 * dc, "radius_mm": 1000 * r_q,
                 "centre_gt_gap_mm": {b: 1000 * (v - (r_q if np.isfinite(r_q) else 0)) for b, v in per_c.items() if b in ("apple", "banana", "table", "orange", "crate")}}
            if gn > 1e-9:
                p = cc - dc * gv / gn
                pw = P.base_to_world(m, d, p)
                P.set_q(m, d, row["qpos"])
                _, _, per = P.nearest(m, d, probe, objg, pw)
                nb = min(per, key=per.get)
                e.update({"surface_gt_nearest": nb, "surface_gt_nearest_mm": 1000 * per[nb], "surface_gt_apple_mm": 1000 * per["apple"],
                          "surface_gt_banana_mm": 1000 * per.get("banana", np.inf), "surface_gt_table_mm": 1000 * per.get("table", np.inf),
                          "surface_above_apple_top_mm": 1000 * float(pw[2] - (apple_w[2] + 0.034)),
                          "surface_above_table_plane_mm": 1000 * float(nrm @ p - off)})
            out.append(e)
        lin.set_joint_parameter_path(None); lin.set_attached(None, None)
        ex = (c.get("esdf_extra") or {})
        rec = {"run": run, "t": t, "held": bool(held), "layers": {"coarse": "esdf_coarse_distance" in z.files, "fine": True, "free": "esdf_free_distance" in z.files},
               "held_free": ex.get("held_free"), "exclusion": c.get("exclusion"), "n_neg_rows": int((cl < 0).sum()), "rows": out}
        res["cases"].append(rec)
        print(run, t, "held", rec["held"], "nneg", rec["n_neg_rows"], [(r["link"], r["step"], r["tier"], round(r["clearance_mm"], 1), r.get("surface_gt_nearest"),
                                                                        round(r.get("surface_gt_nearest_mm", np.nan), 1), round(r.get("surface_above_apple_top_mm", np.nan), 1),
                                                                        round(r.get("surface_above_table_plane_mm", np.nan), 1)) for r in out[:4]], flush=True)
    (OUT / "d_probe.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
