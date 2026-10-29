"""T43 DX-1 — apple-hurdle contacts during carry (CT: 25 carry records): held query spheres, margin rule, recorded TO clearance,
recorded ESDF at the contact. (verifier, CPU only, recorded data only; MUJOCO_GL=osmesa JAX_PLATFORMS=cpu)

usage: T43W_SNAP=<snapshot> dx1_apple.py OUT_JSON KEY [KEY ...]     KEY = <ARM>/H/<name> (ct.json runs key)

Events: ct.json runs[KEY].contacts with kind 'apple' and phase 'carry' (first_t..last_t).
Rows of an event: every control row t in [first_t, last_t]; t* = the row with the smallest apple-hurdle MuJoCo contact dist
  (CT replay json obstacle_contact_rows, apple geoms). States: pre = control row t qpos, post = control row t+1 qpos
  (= replay post state, CT qpos_mismatch_rows 0). The reported state of a row = the one with the smaller apple-hurdle distance
  (mj_geomDistance(apple_geom, hurdle geoms), fromto -> apple point / hurdle point).
Hurdle: client reset (hold_w.reset, mocap pose) of the run's episode / seed / obstacle config, collision geoms of the manager.
Executing chunk = server chunk with t_step == control row chunk_seq; k = step_in_chunk (refined row index).
Held query spheres (server rule, server_y.py): body_cover.lifted_parent_frame if body_cover.frame.near_support else
  body_cover.body_parent_frame; without body_cover (V1): held.spheres_parent_frame. Frame = frame_link (ee_left == MJCF EE_BODY_L).
  actual  : placed with the MuJoCo EE body pose of the reported state.
  planned : placed with FK of the chunk's planning-row qpos whose active-arm joints = refined_chunk row k (what the TO checked).
  fidelity: on every chunk of the run whose recorded worst row is a held row ('attached:*'): |my centre at step - point_m| and
            |my clearance - recorded clearance| (composed field, margin = esdf_margin).
Apple: body 'apple' pose, collision mesh apple_geom vertices (world).
  slip_mm / slip_deg  : apple pose in the EE body frame at the row vs at the attach chunk's planning row (translation norm, rotation angle).
  cover_frac          : fraction of apple mesh vertices inside the union of the held query spheres; cover_max_out_mm = max vertex
                        distance outside the union; contact_pt_sd_mm = signed distance of the apple contact point (fromto apple side)
                        to the union (< 0 inside).
  held_true_bar_mm    : min over held spheres of (signed distance centre -> hurdle geoms) - r  [true geometry], actual / planned.
  apple_true_bar_mm   : mj_geomDistance(apple_geom, hurdle) (actual); apple_planned_bar_mm = apple vertices moved rigidly with the
                        hand from the actual to the planned EE pose, min vertex signed distance to the hurdle geoms.
  ee_plan_vs_actual_mm: |EE origin planned (row k) - actual|.
Margin of held rows (linearize._esdf_clearance): margin = esdf_margin (server log) ; destination_margin only where the field label
  is 'destination' (not recorded); obstacle_margin (T41 a) only for class 'obstacle'; _obstacle_rows sets class 'held' for
  every query >= n_spheres. Recorded per chunk: max_violation_pair.obstacle_margin_class of the worst row.
  implied_margin_mm (held-worst chunks) = d(point_m, composed field) - r(query) - recorded clearance.
TO held rows (recomputed): clearance = d(composed field, full layer: held rows never ask the target-free layer, mask padded False)
  - r - esdf_margin, per refined row 0..7; held_clear_row_k_mm (min over spheres at row k), held_clear_min_mm (rows 0..7).
ESDF at the contact (recorded field of the executing chunk): fine grid (esdf_*) and coarse grid (esdf_coarse_*) trilinear, NaN
  outside; composed = FIA.build_field distance + answer_tier. Points: bar surface point (fromto hurdle side), bar axis point
  (closest axis point of that hurdle geom), apple contact point, held sphere nearest to the bar (d - r).
  bar_axis_min_*: min over 21 axis points within +-50 mm of the axis point.
Chunk record: to.status, to.metrics.geometry_certified, clearance_m, max_violation_pair (link, class, step, tier, clearance), verdict
  reasons (kind, action), executed rows of that chunk (control rows), esdf_stats label_names, target_free n_layers / mode.
Carry chunks of the run (attach chunk .. last chunk before detach/placed): same record + recomputed held min clearance +
  ee_plan_vs_actual_post_mm per control row [t, k, |EE(post state of row t = qpos of row t+1) - EE(planned refined row k)|]."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys, time
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p'))
import numpy as np, mujoco  # noqa: E402
import hold_w as HW  # noqa: E402
import dx_common as C  # noqa: E402
from AG3S.trajopt.experiments import fixed_input_ablation as FIA  # noqa: E402

CT = json.load(open(C.T43 / "CT/raw/ct.json"))["runs"]


def main():
    out_path = pathlib.Path(sys.argv[1]); keys = sys.argv[2:]
    cl = HW.load_client(); res = {"definitions": __doc__, "snapshot": str(HW.HY), "events": [], "runs": {}}
    for key in keys:
        t0 = time.time(); a = C.run_entry(key); rd = pathlib.Path(a["run_dir"]); side = a["side"]
        ctrl, plan = C.frames(rd); seg = a["server_segment"]; chunks = C.segment_chunks(seg)
        st, outside, om, logp = C.server_settings(seg, rd / "manifest.json"); margin = float(st.esdf_margin)
        man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = man["scene"]["episode_index"]
        seed = int(np.load(rd / "traj.npz")["policy_seed"])
        stt = rd.parent / f"{rd.name}.start"
        cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
        L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; hurdle = sorted(L["obstacle_manager"].collision_geom_ids)
        SD = "L" if side == "left" else "R"; ee_name = f"EE_BODY_{SD}"; qi, c0 = (C.LEFT_Q, 0) if side == "left" else (C.RIGHT_Q, 8)
        ga = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom"); mid = int(m.geom_dataid[ga])
        av = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        ft = np.zeros(6)
        rzj = json.load(open(C.T43 / "CT/replay" / (key.replace("/", "_") + ".json")))
        ob_rows = {r["t"]: r["geoms"] for r in rzj["obstacle_contact_rows"]}
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"

        def setq(q):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d)

        def apple_V():
            return d.geom_xpos[ga] + av @ d.geom_xmat[ga].reshape(3, 3).T

        def apple_bar():
            best = (np.inf, None, None, None)
            for h in hurdle:
                v = float(mujoco.mj_geomDistance(m, d, ga, h, 0.3, ft))
                if v < best[0]:
                    best = (v, ft[:3].copy(), ft[3:].copy(), h)
            return best
        zc = {}

        def chunk(seq):
            if seq not in zc:
                zc.clear()
                z = np.load(chunks[seq], allow_pickle=True); s = json.loads(str(z["summary_json"]))
                zc[seq] = (z, s, FIA.build_field(z, outside))
            return zc[seq]

        def planned_q(seq, k):
            z, s, _ = chunk(seq); q = np.asarray(plan[seq]["qpos"], float).copy()
            q[qi] = np.asarray(z["refined_chunk"], float)[k, c0:c0 + 7]; return q

        def held_world(seq, q):
            z, s, _ = chunk(seq); used, sets, fl, bc = C.held_sets(s)
            if not used or used not in sets:
                return None
            setq(q); Te = C.body_T(m, d, ee_name); Cw, r = C.place(Te, sets[used]); return used, Cw, r, Te, bc

        def held_rows(seq):
            """recomputed TO held rows, rows 0..7: (H, n_held) clearance mm (composed field), + centres base."""
            z, s, field = chunk(seq); out = []; cents = []
            for k in range(8):
                hw = held_world(seq, planned_q(seq, k))
                if hw is None:
                    return None, None
                used, Cw, r, _, _ = hw; Tb = C.T_world_base(m, d); Pb = C.to_base(Tb, Cw)
                out.append(1e3 * (np.asarray(field.distance(Pb), float) - r - margin)); cents.append(Pb)
            return np.asarray(out), cents

        def rec_chunk(seq):
            z, s, field = chunk(seq); to = s.get("to") or {}; met = to.get("metrics") or {}; mv = C.lit(met.get("max_violation_pair")) or {}
            es = s.get("esdf_stats") or {}; tf = C.lit(es.get("target_free")) or {}
            rows = [r for r in ctrl.values() if r["chunk_seq"] == seq]
            g = s.get("grasp") or {}
            return {"seq": seq, "t": plan[seq]["t_step"], "to_status": to.get("status"), "geometry_certified": met.get("geometry_certified"),
                    "to_clearance_mm": None if met.get("clearance_m") is None else 1e3 * met["clearance_m"],
                    "worst": {"link": mv.get("link"), "query": mv.get("query"), "class": mv.get("obstacle_margin_class"), "step": mv.get("step"),
                              "tier": mv.get("tier"), "clearance_mm": None if mv.get("clearance_m") is None else 1e3 * mv["clearance_m"], "point_m": mv.get("point_m")},
                    "verdict": [(x.get("kind"), x.get("action")) for x in ((s.get("verdict") or {}).get("reasons") or [])],
                    "n_rows": len(rows), "n_executed": sum(1 for r in rows if r.get("executed", True)),
                    "grasp_state": g.get("state"), "attach": g.get("attach"), "detach": g.get("detach"),
                    "label_names": es.get("label_names"), "target_free_layers": tf.get("n_layers") if isinstance(tf, dict) else None,
                    "target_free_mode": tf.get("mode") if isinstance(tf, dict) else None,
                    "n_robot_spheres": int(z["sphere_radii"].shape[0]), "field_frame": (lambda c: (c or {}).get("frame_index"))(plan[seq].get("field"))}

        def fidelity(seq):
            z, s, field = chunk(seq); mv = C.lit(((s.get("to") or {}).get("metrics") or {}).get("max_violation_pair")) or {}
            if not (isinstance(mv.get("link"), str) and mv["link"].startswith("attached:")):
                return None
            q_i = int(mv["query"]) - int(z["sphere_radii"].shape[0]); k = int(mv["step"])
            hw = held_world(seq, planned_q(seq, k))
            if hw is None:
                return None
            used, Cw, r, _, _ = hw; Tb = C.T_world_base(m, d); Pb = C.to_base(Tb, Cw)
            dpt = float(np.asarray(field.distance(np.asarray([mv["point_m"]], float)))[0])
            mine = 1e3 * (float(np.asarray(field.distance(Pb[q_i:q_i + 1]))[0]) - r[q_i] - margin)
            return {"seq": seq, "set": used, "centre_vs_point_mm": 1e3 * float(np.linalg.norm(Pb[q_i] - np.asarray(mv["point_m"]))),
                    "clearance_mine_mm": mine, "clearance_rec_mm": 1e3 * mv["clearance_m"], "abs_diff_mm": abs(mine - 1e3 * mv["clearance_m"]),
                    "implied_margin_mm": 1e3 * (dpt - r[q_i] - mv["clearance_m"]), "class": mv.get("obstacle_margin_class"), "radius_mm": 1e3 * r[q_i]}

        seqs = sorted(chunks)
        att_seq = next((sq for sq in seqs if (chunk(sq)[1].get("grasp") or {}).get("attach")), None)
        end_seq = None
        if att_seq is not None:
            for sq in seqs:
                if sq > att_seq:
                    g = chunk(sq)[1].get("grasp") or {}
                    if g.get("detach") or g.get("state") in ("placed", "latched", "released", "open"):
                        end_seq = sq; break
        carry = [sq for sq in seqs if att_seq is not None and sq >= att_seq and (end_seq is None or sq < end_seq)]
        runrec = {"key": key, "side": side, "esdf_margin_mm": 1e3 * margin, "obstacle_margin_mm_log": om, "server_log": logp,
                  "attach_seq": att_seq, "attach_t": None if att_seq is None else plan[att_seq]["t_step"], "end_seq": end_seq,
                  "carry_chunks": [], "fidelity": []}
        for sq in seqs:
            f = fidelity(sq)
            if f:
                runrec["fidelity"].append(f)
        for sq in carry:
            rc = rec_chunk(sq); hr, _ = held_rows(sq)
            rc["held_min_clear_mm"] = None if hr is None else float(hr.min())
            rc["held_min_clear_by_row_mm"] = None if hr is None else hr.min(1).tolist()
            pa_ = []
            for r_ in sorted((r for r in ctrl.values() if r["chunk_seq"] == sq), key=lambda r: r["t_step"]):
                nx = ctrl.get(r_["t_step"] + 1)
                if nx is None or r_["step_in_chunk"] >= 8:
                    continue
                setq(planned_q(sq, int(r_["step_in_chunk"]))); Tp_ = C.body_T(m, d, ee_name)
                setq(np.asarray(nx["qpos"], float)); Ta_ = C.body_T(m, d, ee_name)
                pa_.append([r_["t_step"], int(r_["step_in_chunk"]), 1e3 * float(np.linalg.norm(Ta_[:3, 3] - Tp_[:3, 3]))])
            rc["ee_plan_vs_actual_post_mm"] = pa_
            runrec["carry_chunks"].append(rc)
        # apple in EE at attach planning row
        Ta_att = None
        if att_seq is not None:
            setq(np.asarray(plan[att_seq]["qpos"], float)); Te = C.body_T(m, d, ee_name); Tap = C.body_T(m, d, "apple")
            Ta_att = np.linalg.inv(Te) @ Tap
            hw = held_world(att_seq, np.asarray(plan[att_seq]["qpos"], float))
            if hw:
                used, Cw, r, _, _ = hw; setq(np.asarray(plan[att_seq]["qpos"], float)); V = apple_V(); sd = C.union_sd(V, Cw, r)
                runrec["attach_cover"] = {"set": used, "cover_frac": float((sd <= 0).mean()), "cover_max_out_mm": 1e3 * float(max(sd.max(), 0.0)),
                                          "n_spheres": int(len(r)), "radii_mm": (1e3 * r).tolist()}
        for c in CT[key]["contacts"]:
            if c["kind"] != "apple" or c["phase"] != "carry":
                continue
            ev = {"key": key, "first_t": c["first_t"], "last_t": c["last_t"], "n_rows": c["n_rows"], "ct_max_pen_mm": c["max_pen_mm"],
                  "ct_max_force_N": c["max_force_N"], "rows": []}
            for t in range(int(c["first_t"]), int(c["last_t"]) + 1):
                if t not in ctrl:
                    continue
                orow = ob_rows.get(t, {}); ap = [v for v in orow.values() if v[2] == "apple"]
                row = {"t": t, "ct_min_dist_mm": None if not ap else 1e3 * min(v[4] for v in ap),
                       "ct_hurdle_pairs": sorted({v[0] for v in ap})}
                best = None
                for which, q in (("pre", ctrl[t]["qpos"]), ("post", (ctrl.get(t + 1) or {}).get("qpos"))):
                    if q is None:
                        continue
                    setq(np.asarray(q, float)); dist, pa, ph, h = apple_bar()
                    if best is None or dist < best[1]:
                        best = (which, dist, np.asarray(q, float), pa, ph, h)
                which, dist, q, pa, ph, h = best
                seq = int(ctrl[t]["chunk_seq"]); k = int(ctrl[t]["step_in_chunk"]); z, s, field = chunk(seq)
                setq(q); Te = C.body_T(m, d, ee_name); Tap = C.body_T(m, d, "apple"); V = apple_V(); Tb = C.T_world_base(m, d)
                row.update({"state": which, "apple_true_bar_mm": 1e3 * dist, "hurdle_geom": gname(h), "chunk_seq": seq, "step_in_chunk": k,
                            "executed": bool(ctrl[t].get("executed", True)), "field_age_ms": (ctrl[t].get("field") or {}).get("age_ms"),
                            "field_frame": (ctrl[t].get("field") or {}).get("frame_index")})
                applied = np.asarray(ctrl[t]["applied_ctrl"]["arm"], float)[(0 if side == "left" else 7):(7 if side == "left" else 14)]
                row["applied_vs_refined_rad"] = float(np.abs(applied - np.asarray(z["refined_chunk"], float)[k, c0:c0 + 7]).max())
                if Ta_att is not None:
                    Trel = np.linalg.inv(Te) @ Tap; dT = Trel[:3, 3] - Ta_att[:3, 3]
                    Rr = Ta_att[:3, :3].T @ Trel[:3, :3]
                    row["slip_mm"] = 1e3 * float(np.linalg.norm(dT)); row["slip_vec_ee_mm"] = (1e3 * dT).tolist()
                    row["slip_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))))
                used, sets, fl, bc = C.held_sets(s)
                row["held_set"] = used; row["body_cover_mode"] = None if not bc else bc.get("mode")
                row["near_support"] = None if not bc else bool((bc.get("frame") or {}).get("near_support"))
                if used and used in sets:
                    Cw, r = C.place(Te, sets[used]); sd = C.union_sd(V, Cw, r)
                    row["n_held"] = int(len(r)); row["held_radii_mm"] = (1e3 * r).tolist()
                    row["cover_frac"] = float((sd <= 0).mean()); row["cover_max_out_mm"] = 1e3 * float(max(sd.max(), 0.0))
                    row["contact_pt_sd_mm"] = 1e3 * float(C.union_sd(pa, Cw, r)[0])
                    hs = C.hurdle_sd(m, d, hurdle, Cw) - r; j = int(np.argmin(hs))
                    row["held_true_bar_mm"] = 1e3 * float(hs[j]); row["held_true_bar_idx"] = j
                    row["held_true_bar_all_mm"] = (1e3 * hs).tolist()
                    Pb = C.to_base(Tb, Cw)
                    dcomp = np.asarray(field.distance(Pb), float); tiers = np.asarray(field.answer_tier(Pb, target_free=False)).astype(str)
                    df = C.samp(z["esdf_distance"], z["esdf_origin"], z["esdf_voxel_size"], Pb); dc = C.samp(z["esdf_coarse_distance"], z["esdf_coarse_origin"], z["esdf_coarse_voxel_size"], Pb)
                    row["held_actual_esdf"] = {"d_minus_r_composed_mm": (1e3 * (dcomp - r)).tolist(), "tier": tiers.tolist(),
                                               "d_minus_r_fine_mm": (1e3 * (df - r)).tolist(), "d_minus_r_coarse_mm": (1e3 * (dc - r)).tolist(),
                                               "nearest_bar_sphere": j, "nearest_bar_d_minus_r_composed_mm": 1e3 * float(dcomp[j] - r[j])}
                    # planned (TO) pose at row k
                    hwp = held_world(seq, planned_q(seq, k)); _, Cp, rp, Tep, _ = hwp
                    row["ee_plan_vs_actual_mm"] = 1e3 * float(np.linalg.norm(Tep[:3, 3] - Te[:3, 3]))
                    row["ee_actual_minus_plan_world_mm"] = (1e3 * (Te[:3, 3] - Tep[:3, 3])).tolist()
                    row["arm_q_actual_minus_cmd_deg"] = np.degrees(q[qi] - np.asarray(z["refined_chunk"], float)[k, c0:c0 + 7]).tolist()
                    hsp = C.hurdle_sd(m, d, hurdle, Cp) - rp; row["held_planned_true_bar_mm"] = 1e3 * float(hsp.min())
                    Vp = (Tep @ np.linalg.inv(Te) @ np.c_[V, np.ones(len(V))].T).T[:, :3]
                    row["apple_planned_bar_mm"] = 1e3 * float(C.hurdle_sd(m, d, hurdle, Vp).min())
                    row["apple_actual_bar_vertex_mm"] = 1e3 * float(C.hurdle_sd(m, d, hurdle, V).min())
                    setq(q)
                # recorded TO and recomputed held rows
                hr, cents = held_rows(seq); setq(q)
                if hr is not None:
                    row["held_clear_row_k_mm"] = float(hr[k].min()) if k < 8 else None
                    row["held_clear_min_mm"] = float(hr.min()); row["held_clear_by_row_mm"] = hr.min(1).tolist()
                row["chunk"] = rec_chunk(seq)
                # ESDF at the contact
                pax = C.geom_axis_point(m, d, h, ph); bar_r = float(m.geom_size[h][0])
                R = d.geom_xmat[h].reshape(3, 3); axs = np.array([pax + u * R[:, 2] for u in np.linspace(-0.05, 0.05, 21)])
                pts = {"bar_surface": ph, "bar_axis": pax, "apple_contact": pa}
                e = {"bar_radius_mm": 1e3 * bar_r, "fine_voxel_m": float(z["esdf_voxel_size"]), "coarse_voxel_m": float(z["esdf_coarse_voxel_size"])}
                for nm, p in pts.items():
                    pb = C.to_base(Tb, p)
                    e[nm] = {"fine_mm": 1e3 * float(C.samp(z["esdf_distance"], z["esdf_origin"], z["esdf_voxel_size"], pb)[0]),
                             "coarse_mm": 1e3 * float(C.samp(z["esdf_coarse_distance"], z["esdf_coarse_origin"], z["esdf_coarse_voxel_size"], pb)[0]),
                             "composed_mm": 1e3 * float(np.asarray(field.distance(pb))[0]),
                             "tier": str(np.asarray(field.answer_tier(pb, target_free=False)).reshape(-1)[0]),
                             "in_fine_grid": C.grid_index(z["esdf_origin"], z["esdf_voxel_size"], z["esdf_shape"], pb[0])[1]}
                    if "esdf_free_distance" in z.files:
                        e[nm]["target_free_fine_mm"] = 1e3 * float(C.samp(z["esdf_free_distance"], z["esdf_free_origin"], z["esdf_free_voxel_size"], pb)[0])
                ab = C.to_base(Tb, axs)
                e["bar_axis_min_fine_mm"] = 1e3 * float(np.nanmin(np.r_[C.samp(z["esdf_distance"], z["esdf_origin"], z["esdf_voxel_size"], ab), np.inf]))
                e["bar_axis_min_coarse_mm"] = 1e3 * float(np.nanmin(np.r_[C.samp(z["esdf_coarse_distance"], z["esdf_coarse_origin"], z["esdf_coarse_voxel_size"], ab), np.inf]))
                e["bar_axis_min_composed_mm"] = 1e3 * float(np.min(field.distance(ab)))
                row["esdf"] = e
                row["world_points"] = {"bar_surface": ph.tolist(), "bar_axis": pax.tolist(), "apple_contact": pa.tolist(), "ee": Te[:3, 3].tolist()}
                ev["rows"].append(row)
            okr = [r for r in ev["rows"] if r.get("ct_min_dist_mm") is not None]
            ev["t_star"] = min(okr, key=lambda r: r["ct_min_dist_mm"])["t"] if okr else ev["rows"][0]["t"]
            res["events"].append(ev)
            r0 = next(r for r in ev["rows"] if r["t"] == ev["t_star"])
            print(key, c["first_t"], "t*", ev["t_star"], "slip %.1f" % r0.get("slip_mm", -1), "cover %.2f" % r0.get("cover_frac", -1), "cpt_sd %.1f" % r0.get("contact_pt_sd_mm", 0),
                  "held_true %.1f" % r0.get("held_true_bar_mm", 0), "heldTO_k %.1f" % (r0.get("held_clear_row_k_mm") or 0), "bar_surf comp %.1f %s" % (r0["esdf"]["bar_surface"]["composed_mm"], r0["esdf"]["bar_surface"]["tier"]),
                  "plan-act %.1f" % r0.get("ee_plan_vs_actual_mm", -1), r0["chunk"]["to_status"], r0["chunk"]["geometry_certified"], r0["chunk"]["worst"]["link"], r0["chunk"]["worst"]["class"], flush=True)
        runrec["sec"] = round(time.time() - t0, 1); res["runs"][key] = runrec
        out_path.write_text(json.dumps(res, indent=1, default=float))
    out_path.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
