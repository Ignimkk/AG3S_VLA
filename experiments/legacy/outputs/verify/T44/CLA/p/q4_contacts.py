"""T44dA verifier Q4: every hand-fruit_other contact event (CL/replay/<ARM>_R_<NAME>.json, contact_events) of C0 / C1a / C1b.
Per event: rows, max force, MuJoCo body of the hand geom (subtree based; the MJCF/URDF left finger name swap is reported, not used),
the fruit, and for the planning chunk that issued the first contact row (seq = replay seq at t_first):
  * field: was the distractor in the served target-free fine layer? f_tf (recorded esdf_free grid) at the distractor's collision-mesh
    vertices (GT pose of that planning row) -> fraction of vertices with f < 7.5 mm (surface present) ; same for main fine (f_main).
  * target_exclusion of that chunk: n_kept_nontarget, n_ambiguous, n_unowned_removed, excl_mode ; ball centre / radius vs the distractor.
  * TO prediction: min over the active hand's authorized spheres of (target-free distance - r) along the REFINED plan rows that cover the
    contact rows (field.py ON formula, as served), and the GT hull gap of those spheres to the distractor at the planned rows.
  * execution: GT hull gap of the active hand spheres to the distractor at the ACTUAL qpos of the first contact row (control row qpos) and
    |FK(actual) - FK(planned row)| max over the hand spheres (servo / tracking error).
-> CLA/q4_contacts.json   (run from ${AG3S_PROJECT_ROOT}-t44cl)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import os, sys, json, glob, pathlib
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/CLA/p'))
import numpy as np, mujoco
import fkmodel as F, field as FD
from extract_chunks import hull_sd, FRUITS, FV, _quat_R, _m
CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL')); C = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CLA'))
sc, qm = F.model("gripper"); links = np.asarray(qm.sphere_link_names); rad = np.asarray(qm.radii)
out = []
for arm in ("C0", "C1a", "C1b"):
    for jf in sorted(glob.glob(str(CL / "replay" / f"{arm}_R_*.json"))):
        j = json.load(open(jf)); ev = j["contact_events"].get("hand-fruit_other") or []
        if not ev: continue
        name = pathlib.Path(jf).stem.split("_R_")[1]
        a = json.load(open(CL / "analysis" / f"{name}.json")); side = a["used_arm"]; s1 = "l" if side == "left" else "r"; tgt = a["target_fruit"]
        rows = [json.loads(l) for l in open(pathlib.Path(_ag3s_workspace_path('')) / a["run_dir"] / "frames.jsonl")]
        P = {r["seq"]: r for r in rows if r["kind"] == "planning"}; CT = {r["t_step"]: r for r in rows if r["kind"] == "control"}
        files = {c["seq"]: c["server_file"] for c in a["server_chunks"]}
        rz = np.load(str(jf)[:-5] + ".npz")
        act = np.array([l in (f"ee_finger_{s1}1", f"ee_finger_{s1}2", f"ee_{side}") for l in links])
        for e in ev:
            t0 = int(e["t_first"]); seq = int(rz["seq"][t0]); p = P.get(seq); z = np.load(files[seq], allow_pickle=True); s = json.loads(str(z["summary_json"]))
            fruit = [b for g, b in [(pp[0], pp[1]) for pp in e["pairs"]] if b.split("_")[0] in FRUITS] or [pp[0] for pp in e["pairs"]]
            fr = [b.split("_")[0] for pp in e["pairs"] for b in pp if b.split("_")[0] in FRUITS and b.split("_")[0] != tgt][0]
            hg = [b for pp in e["pairs"] for b in pp if b.startswith("geom#")]
            hbody = [mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_BODY, int(_m.geom_bodyid[int(g[5:])])) for g in hg]
            op = p["object_poses"]; V = FV[fr] @ _quat_R(op[fr]["quat"]).T + np.asarray(op[fr]["pos"])
            on = FD.build(z, True)
            ftf = np.asarray(on.target_free_distance(V)); fmain = np.asarray(on.distance(V))
            te = s.get("target_exclusion") or {}
            tf = s.get("esdf_stats", {}).get("target_free"); tf = eval(tf) if isinstance(tf, str) else (tf or {}); ball = (tf or {}).get("ball") or {}
            bc = np.asarray(ball.get("centre_m") or [np.nan] * 3); br = (ball.get("radius_mm") or np.nan) / 1e3
            k = t0 - p["t_step"]  # row of the chunk
            mask = np.asarray((s.get("target_permission") or {}).get("target_field_exclude") or [], bool)
            pr = F.params_from_qpos(sc, qm, p["qpos"])
            Qf = F.chunk_q(sc, qm, p["qpos"], np.asarray(z["refined_chunk"], np.float64)[:16])
            rows_k = list(range(max(0, k), min(16, k + max(1, int(e["t_last"]) - t0 + 1))))
            Sf = np.stack([F.spheres(qm, Qf[i], pr) for i in rows_k])          # (n, S, 3)
            sel = act & mask if mask.size == act.size else act
            X = Sf[:, sel].reshape(-1, 3); r = np.tile(rad[sel], len(rows_k))
            dtf = np.asarray(on.target_free_distance(X)) - r
            gap_plan = hull_sd(op[fr], fr, X) - r
            ctl = CT.get(t0); qa = ctl["qpos"] if ctl else None
            if qa is not None:
                Sa = F.spheres(qm, F.q_from_qpos(sc, qm, qa), F.params_from_qpos(sc, qm, qa))
                gap_act = float((hull_sd(op[fr], fr, Sa[act]) - rad[act]).min() * 1e3)
                servo = float(np.linalg.norm(Sa[act] - Sf[0][act], axis=1).max() * 1e3)
            else:
                gap_act = servo = None
            rec = {"arm": arm, "name": name, "t_first": t0, "t_last": int(e["t_last"]), "substeps": e["substeps"], "max_force_N": e["max_force_N"],
                   "min_dist_mm": 1e3 * e["min_dist_m"], "hand_bodies_mjcf": hbody, "fruit": fr, "seq": seq, "row_in_chunk": k, "hold_rows": e["n_hold_rows"],
                   "grasp_state": (s.get("grasp") or {}).get("state"), "excl_mode": te.get("mode"), "excl_method": te.get("exclusion_method"),
                   "n_kept_nontarget": te.get("n_kept_nontarget"), "n_ambiguous": te.get("n_ambiguous"), "n_unowned_removed": te.get("n_unowned_removed"),
                   "n_removed": te.get("n_removed"), "n_seeds_in_ball": te.get("n_seeds_in_ball"),
                   "distractor_centre_to_ball_centre_mm": float(np.linalg.norm(np.asarray(op[fr]["pos"]) - bc) * 1e3), "ball_r_mm": ball.get("radius_mm"),
                   "frac_distractor_verts_f_tf_lt_7p5mm": float(np.mean(ftf < 0.0075)), "frac_distractor_verts_f_main_lt_7p5mm": float(np.mean(fmain < 0.0075)),
                   "frac_distractor_verts_inside_ball": float(np.mean(np.linalg.norm(V - bc, axis=1) <= br)),
                   "to_pred_min_tf_clearance_mm": float(dtf.min() * 1e3), "plan_gt_gap_to_distractor_mm": float(gap_plan.min() * 1e3),
                   "actual_gt_gap_to_distractor_mm": gap_act, "servo_err_hand_mm": servo, "to_status": (s.get("to") or {}).get("status")}
            out.append(rec); print(json.dumps(rec))
json.dump(out, open(C / "q4_contacts.json", "w"), indent=1)
