"""T43 HM — offline survey of the held (apple) ESDF rows on the recorded V7g / V8 runs (implementer, CPU only, no new runs).

usage: T43W_SNAP=<client snapshot> PYTHONPATH=${AG3S_PROJECT_ROOT} hm_survey.py ARM OUT_JSON [KEY ...]
  ARM = V7g | V8 ; index = analysis json (V7g CT/raw/analysis_v7g.json, V8 V8/analysis/analysis.json); stages H and B.

Per attached server chunk (held query set present; DX rule `dx_common.held_sets`) and per refined row k = 0..7 (plus, for the
executed control row of that (seq, k), the servo-predicted arm joints from HX raw `pred_q14`, T43/HX/raw/hx_<ARM>.json):
  held query spheres = server held set (body_parent_frame / lifted_parent_frame) placed on the EE body of FK(planning qpos,
  arms = refined row k) -- the HX `to_rows` rule.  P = centre in the field (base) frame.
  d = composed field (FIA.build_field: coarse -> fine; held rows read the main layer, target-free mask is padded False for held
  queries, linearize.py), g = field gradient at P, S = P - d * g/|g| = the nearest field surface point (base frame).
  features:  d_minus_r ; s_plane = n.P - off (held.support.plane of the record) and support = |s_plane - d| <= 2 voxel (the
  linearize `_obstacle_rows` rule; plane = the table plane AG3S recorded at attach -- SceneSnapshot.support_planes is the frame's
  support_surfaces, not recorded) ; self_cur = signed distance of S to the union of this row's held spheres ;
  self_cap = signed distance of S to the attach-time fit spheres (record `held.fit_spheres`, base frame, capture pose) ;
  self_now = signed distance of S to the union of the held query spheres at the observation state q_now (planning qpos) --
  where the field saw the held object ; self_filt = same for the self-filter (fit) spheres `filter_spheres_parent_frame` ;
  GT identity of S: nearest MuJoCo collision geom group at the observation state (planning qpos -- the field is a snapshot of
  it): hurdle = active obstacle-manager geoms, table, crate, apple, other fruit, shelf, floor/walls, robot (every other
  collision geom) -- box / capsule / cylinder signed, mesh = nearest vertex.
Phase of a chunk: lift = first 3 attached chunks; crate = apple (GT, planning state) centre within 80 mm of the crate geoms;
  carry = the rest.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys, time
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p'))
import numpy as np, mujoco  # noqa: E402
import hold_w as HW  # noqa: E402
import dx_common as C  # noqa: E402
from AG3S.trajopt.experiments import fixed_input_ablation as FIA  # noqa: E402

T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
ANALYSIS = {"V7g": T43 / "CT/raw/analysis_v7g.json", "V8": T43 / "V8/analysis/analysis.json"}
HXRAW = {"V7g": T43 / "HX/raw/hx_V7g.json", "V8": T43 / "HX/raw/hx_V8.json"}
GROUPS = ("hurdle", "table", "crate", "apple", "fruit", "shelf", "room", "robot")


def main():
    arm, out_path = sys.argv[1], pathlib.Path(sys.argv[2])
    runs = json.load(open(ANALYSIS[arm]))["runs"]
    keys = sys.argv[3:] or sorted(runs)
    hx = json.load(open(HXRAW[arm]))["runs"]
    cl = HW.load_client()
    res = {"definitions": __doc__, "arm": arm, "client_snapshot": str(HW.HY), "runs": {}}
    for key in keys:
        t0w = time.time(); a = runs[key]; rd = pathlib.Path(a["run_dir"])
        if a.get("grasp_t") in (None, "None"):
            res["runs"][key] = {"key": key, "grasp_t": None, "chunks": []}; continue
        man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = int(man["scene"]["episode_index"])
        z = np.load(rd / "traj.npz", allow_pickle=True); seed = int(z["policy_seed"])
        stt = rd.parent / f"{rd.name}.start"
        cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines()
                    if l.startswith("obstacle=") and l != "obstacle=:"), None) if stt.exists() else None
        L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; mg = L["obstacle_manager"]
        hurdle = set(int(g) for g in mg.collision_geom_ids) if mg is not None else set()
        side = "right" if ep == 1834 else "left"; SD = side[0].upper()
        qL = np.asarray(L["idx"]["left_q"][:7], int); qR = np.asarray(L["idx"]["right_q"][:7], int)
        apple_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple"); apple_q = int(m.jnt_qposadr[m.body_jntadr[apple_b]])
        apple_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom")
        # geom groups
        ggroup = {}
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            root = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.body_rootid[int(m.geom_bodyid[g])])) or ""
            if g in hurdle:
                ggroup[g] = "hurdle"
            elif root == "table":
                ggroup[g] = "table"
            elif root == "crate":
                ggroup[g] = "crate"
            elif g == apple_g:
                ggroup[g] = "apple"
            elif root in ("banana", "orange", "pear"):
                ggroup[g] = "fruit"
            elif root == "shelf":
                ggroup[g] = "shelf"
            elif root in ("world", "office"):
                ggroup[g] = "room"
            elif not root.startswith("pick_obstacle"):
                ggroup[g] = "robot"
        crate_geoms = [g for g, v in ggroup.items() if v == "crate"]

        def setq(q):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d)

        def mesh_verts(g):
            mid = int(m.geom_dataid[g]); a0 = int(m.mesh_vertadr[mid]); n = int(m.mesh_vertnum[mid])
            V = m.mesh_vert[a0:a0 + n].astype(float)
            return V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]

        def group_dist(Pw):
            """(N, len(GROUPS)) min distance (m) of world points to each group (signed for primitives, mesh = nearest vertex)."""
            out = np.full((len(Pw), len(GROUPS)), np.inf)
            for g, gr in ggroup.items():
                t = int(m.geom_type[g])
                if t == 0:   # plane (floor): z distance
                    dd = Pw[:, 2] - d.geom_xpos[g][2]
                elif t == 7:
                    V = mesh_verts(g); dd = np.linalg.norm(Pw[:, None, :] - V[None], axis=2).min(1)
                else:
                    try:
                        dd = C.geom_point_sd(m, d, g, Pw)
                    except ValueError:
                        continue
                j = GROUPS.index(gr); out[:, j] = np.minimum(out[:, j], dd)
            return out

        def with_arms(qfull, a14, move_apple):
            q = np.asarray(qfull, float).copy()
            if move_apple:
                setq(q); T0 = C.body_T(m, d, f"EE_BODY_{SD}")
            q[qL] = a14[0:7]; q[qR] = a14[7:14]
            if move_apple:
                setq(q); T1 = C.body_T(m, d, f"EE_BODY_{SD}")
                Ta = np.eye(4); Ta[:3, :3] = C.qmat(q[apple_q + 3:apple_q + 7]); Ta[:3, 3] = q[apple_q:apple_q + 3]
                Tn = T1 @ np.linalg.inv(T0) @ Ta; qa = np.zeros(4); mujoco.mju_mat2Quat(qa, Tn[:3, :3].reshape(-1))
                q[apple_q:apple_q + 3] = Tn[:3, 3]; q[apple_q + 3:apple_q + 7] = qa
            return q

        rows_f = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
        ctrl = {r["t_step"]: r for r in rows_f if r["kind"] == "control"}
        plan = {r["seq"]: r for r in rows_f if r["kind"] == "planning"}
        by_seqk = {(int(r["chunk_seq"]), int(r["step_in_chunk"])): t for t, r in ctrl.items()}
        seg = C.lit(a.get("server_segment")); chunks = C.segment_chunks(seg) if seg and seg != "None" else {}
        st, outside, om, logp = C.server_settings(seg, rd / "manifest.json")
        hxrows = {r["t"]: r for r in (hx.get(key) or {}).get("rows", [])}
        runrec = {"key": key, "side": side, "episode": ep, "seed": seed, "grasp_t": a.get("grasp_t"), "place_t": a.get("place_t"),
                  "esdf_margin_mm": 1e3 * float(st.esdf_margin), "obstacle_margin_mm_log": om, "chunks": []}
        n_att = 0
        for seq in sorted(chunks):
            if seq not in plan:
                continue
            zz = np.load(chunks[seq], allow_pickle=True); s = json.loads(str(zz["summary_json"]))
            used, sets, flk, bc = C.held_sets(s)
            held = sets.get(used) if used else None
            if not held:
                continue
            n_att += 1
            field = FIA.build_field(zz, outside); vox = float(field.grid.voxel_size)
            ref = np.asarray(zz["refined_chunk"], float); qcap = np.asarray(plan[seq]["qpos"], float)
            g_ = s.get("grasp") or {}; att = C.lit(g_.get("attached")) or {}; hrec = C.lit(att.get("held")) or {}
            sup = (hrec.get("support") or {}).get("plane") or {}
            plane = None
            if sup.get("normal") is not None and sup.get("offset_m") is not None:
                plane = (np.asarray(sup["normal"], float), float(sup["offset_m"]))
            fit = [(np.asarray(x["centre_m"], float), 1e-3 * float(x["radius_mm"])) for x in (hrec.get("fit_spheres") or [])]
            Fc = np.array([c for c, _ in fit]) if fit else None; Fr = np.array([r for _, r in fit]) if fit else None
            to = s.get("to") or {}; met = to.get("metrics") or {}; mv = C.lit(met.get("max_violation_pair")) or {}
            verdict = [(x.get("kind"), x.get("action")) for x in ((s.get("verdict") or {}).get("reasons") or [])]
            gate = plan[seq].get("gate"); gate = gate.get("action") if isinstance(gate, dict) else gate
            setq(qcap); Tw = C.T_world_base(m, d)
            apple_w = d.xpos[apple_b].copy()
            crate_d = min(float(C.geom_point_sd(m, d, g, apple_w[None])[0]) for g in crate_geoms) if crate_geoms else np.inf
            phase = "lift" if n_att <= 3 else ("crate" if crate_d <= 0.080 else "carry")
            crec = {"seq": seq, "t0": int(plan[seq]["t_step"]), "phase": phase, "n_att": n_att, "held_set": used,
                    "n_held": len(held), "S": int(zz["sphere_radii"].shape[0]), "voxel_m": vox,
                    "plane": None if plane is None else [*plane[0].tolist(), plane[1]],
                    "fit_spheres": None if Fc is None else [[*c.tolist(), r] for c, r in zip(Fc, Fr)],
                    "apple_crate_mm": 1e3 * crate_d, "to_status": to.get("status"),
                    "to_clearance_mm": None if met.get("clearance_m") is None else 1e3 * float(met["clearance_m"]),
                    "worst": {"link": mv.get("link"), "query": mv.get("query"), "step": mv.get("step"), "class": mv.get("obstacle_margin_class"),
                              "obstacle": mv.get("obstacle"), "clearance_mm": None if mv.get("clearance_m") is None else 1e3 * float(mv["clearance_m"])},
                    "verdict": verdict, "gate": gate, "rows": []}

            # the held object where the field saw it: query spheres and filter (fit) spheres at the observation state q_now
            setq(qcap); Tb0 = C.T_world_base(m, d); Te0 = C.body_T(m, d, f"EE_BODY_{SD}")
            Hn_c, Hn_r = C.place(Te0, held); Hn_b = C.to_base(Tb0, Hn_c)
            filt = sets.get("filter_spheres_parent_frame")
            if filt:
                Fn_c, Fn_r = C.place(Te0, filt); Fn_b = C.to_base(Tb0, Fn_c)
            else:
                Fn_b = Fn_r = None
            crec["held_now"] = [[*c.tolist(), float(r)] for c, r in zip(Hn_b, Hn_r)]

            def held_rows(a14, qbase):
                q = with_arms(qbase, a14, True); setq(q); Tb = C.T_world_base(m, d)
                Te = C.body_T(m, d, f"EE_BODY_{SD}"); Hc, Hr = C.place(Te, held)
                Pb = C.to_base(Tb, Hc); dd = np.asarray(field.distance(Pb), float)
                gg = np.asarray(field.gradient(Pb), float); nrm = np.linalg.norm(gg, axis=1, keepdims=True)
                ghat = np.where(nrm > 1e-12, gg / np.maximum(nrm, 1e-12), 0.0)
                Sb = Pb - dd[:, None] * ghat
                Sw = Sb @ Tb[:3, :3].T + Tb[:3, 3]
                Hb = C.to_base(Tb, Hc)
                out = {"d_minus_r_mm": (1e3 * (dd - Hr)).tolist(), "r_mm": (1e3 * Hr).tolist(), "grad_norm": nrm[:, 0].tolist(),
                       "self_cur_mm": (1e3 * C.union_sd(Sb, Hb, Hr)).tolist(),
                       "self_cap_mm": None if Fc is None else (1e3 * C.union_sd(Sb, Fc, Fr)).tolist(),
                       "self_now_mm": (1e3 * C.union_sd(Sb, Hn_b, Hn_r)).tolist(),
                       "self_filt_mm": None if Fn_b is None else (1e3 * C.union_sd(Sb, Fn_b, Fn_r)).tolist(),
                       "P_base": Pb.tolist(), "S_base": Sb.tolist()}
                if plane is not None:
                    sp_ = Pb @ plane[0] - plane[1]; out["s_plane_mm"] = (1e3 * sp_).tolist()
                    out["support"] = (np.abs(sp_ - dd) <= 2.0 * vox).tolist()
                setq(qcap)          # the field is a snapshot of the observation state: identify S against that scene
                gd = group_dist(Sw); j = gd.argmin(1)
                setq(q)
                out["gt_group"] = [GROUPS[int(i)] for i in j]; out["gt_dist_mm"] = (1e3 * gd[np.arange(len(j)), j]).tolist()
                out["gt_all_mm"] = (1e3 * gd).round(2).tolist()
                # centre-level truth for the hurdle (sphere model): signed distance of the held sphere to hurdle geoms minus r
                if hurdle:
                    out["true_hurdle_mm"] = (1e3 * (C.hurdle_sd(m, d, sorted(hurdle), Hc) - Hr)).tolist()
                return out

            for k in range(min(8, ref.shape[0])):
                a14 = np.r_[ref[k, 0:7], ref[k, 8:15]]
                rr = {"k": k, "plan": held_rows(a14, qcap)}
                t = by_seqk.get((seq, k))
                rr["t"] = t; rr["executed"] = None if t is None else bool(ctrl[t].get("executed", True))
                h = hxrows.get(t) if t is not None else None
                if h is not None and h.get("pred_q14") is not None:
                    rr["servo"] = held_rows(np.asarray(h["pred_q14"], float), qcap)
                crec["rows"].append(rr)
            runrec["chunks"].append(crec)
        runrec["sec"] = round(time.time() - t0w, 1)
        res["runs"][key] = runrec
        print(key, "attached chunks", n_att, "sec", runrec["sec"], flush=True)
        out_path.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    out_path.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
