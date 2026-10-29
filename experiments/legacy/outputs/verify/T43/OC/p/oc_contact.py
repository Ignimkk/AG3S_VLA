"""T43 OC — per post-release finger/arm-bar contact: is the bar in the observation, the clouds and the field, chunk by chunk.
(verifier, CPU only, recorded data only; MUJOCO_GL=osmesa JAX_PLATFORMS=cpu; adapted from T43/DX/p/dx2_finger.py)

usage: T43W_SNAP=<client snapshot> PYTHONPATH=<same snapshot root> oc_contact.py OUT_JSON KEY:T:MJCF_BODY [...]

Contact state: physics replay (DX-2 / CT method) up to row t; the sub-step (or post state) of row t with the smallest
  mj_geomDistance(MJCF body collision geoms, hurdle geoms); fromto -> finger point pf, bar point pb (world).
Bar: hurdle capsule geom (type 3) of the obstacle manager. axial a(P) = (P - c_bar) . axis_bar; a_pb = axial of pb.
  bar surface samples S: axial a_pb + {-30..+30 mm step 5}, 16 angles at the capsule radius (world, static mocap).
Chunk history: every chunk of the run's server segment (analyze_t41 pairing = analysis.json server_segment);
  observation state of chunk seq = planning-row qpos (frames.jsonl kind=planning, seq) = the frozen state the depth was rendered at.
  base frame = MuJoCo body 'base' at that state.
  cameras (zed_left = head, wrist_cam_l, wrist_cam_r; TransportScene.attach(m, d).capture, 480x640, client capture path):
    bar px = pixels whose segmentation body is a hurdle body AND whose back-projected point (PointCloudConfig(range_max=2.0))
             has hurdle signed distance < 10 mm;
    near = bar px with |P - pb| < 30 mm; bins = bar px on the bar capsule (axis distance < r + 10 mm) by axial offset from a_pb,
           edges [-150, -90, -30, 30, 90, 150] mm (centre bin = the contact segment);
    robot_filtered = self_filter_mask (grounding_report.build_robot_model(scene) = serve_safe filter_robot, fingers from scene,
           code-default inflation); held_filtered = inside the record's held.filter_spheres_parent_frame (+ cap slip not added)
           placed at the hold-link body (ee_left = MJCF EE_BODY_L, ee_right = EE_BODY_R) when the chunk is attached;
    vis = near & ~robot_filtered & ~held_filtered (px that reach the TSDF).
    pb_proj: pb projected; occluded = rendered depth at that pixel < z - 5 mm; body at that pixel.
  clouds (chunk npz raw_cloud / filtered_cloud, base frame): n_*_near = points with hurdle sd < 10 mm and |P - pb| < 30 mm.
  ESDF grids (recorded float16, own trilinear, NaN outside): fine full d(pb), min over S; coarse d(pb); target-free d(pb).
  carve-out: esdf_stats.target_free (mode, label, ball centre / radius / z_min); pb_minus_ball_mm = |pb - c| - R (< 0 inside);
    n_S_in_ball = samples with |S - c| <= R and z >= z_min.
  held_free (TSDF voxels set free each held frame, pipeline._held_free_points): held.spheres_parent_frame placed at the hold-link
    body at the observation state; heldfree_S_sd_min_mm = min over S of (|S - c| - r); n_S_in_heldfree = samples inside;
    also first_free_spheres (as recorded, base frame) on the chunks where the record lists them; esdf_stats.held_free counts.
  target identity: manipulated id / state / centroid, subtask label / gate / blocked / released / placed_seen, grasp state,
    exclusion source; target_points: n and n with hurdle sd < 20 mm.
Exec chunk (control row t chunk_seq): composed TO field FIA.build_field distance at pb / pf (full and target-free), answer tier,
  grid inside flags, recorded per-camera n_masked_px vs repro.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys, time
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p'))
import numpy as np, mujoco  # noqa: E402
import hold_w as HW  # noqa: E402
import dx_common as C  # noqa: E402
from AG3S.trajopt.experiments import fixed_input_ablation as FIA  # noqa: E402
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, set_finger_joints_from_scene  # noqa: E402
from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model  # noqa: E402
from AG3S.ag3s.stages.robot_filter import self_filter_mask  # noqa: E402
from AG3S.ag3s.stages.reconstruction import backproject  # noqa: E402
from AG3S.ag3s.config import PointCloudConfig  # noqa: E402

C.ANALYSES["V8"] = C.T43 / "V8/analysis/analysis.json"; C.ARM_SRC["V8"] = "V8"
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
REC_CAM = {"zed_left": "head", "wrist_cam_l": "left_wrist", "wrist_cam_r": "right_wrist"}
PCFG = PointCloudConfig(range_max=2.0)
BINS = np.array([-0.150, -0.090, -0.030, 0.030, 0.090, 0.150])
HOLD_BODY = {"ee_left": "EE_BODY_L", "ee_right": "EE_BODY_R"}


def spheres_world(T, L):
    if not L:
        return None, None
    c = np.array([T[:3, :3] @ np.asarray(x["centre_m"], float) + T[:3, 3] for x in L]); r = np.array([1e-3 * float(x["radius_mm"]) for x in L])
    return c, r


def main():
    out_path = pathlib.Path(sys.argv[1]); specs = [s.rsplit(":", 2) for s in sys.argv[2:]]
    cl = HW.load_client(); res = {"definitions": __doc__, "snapshot": str(HW.HY), "events": []}
    for key, t, body in specs:
        t = int(t); t0 = time.time(); a = C.run_entry(key); rd = pathlib.Path(a["run_dir"])
        ctrl, plan = C.frames(rd); seg = a["server_segment"]; chunks = C.segment_chunks(seg)
        st, outside, om, logp = C.server_settings(seg, rd / "manifest.json")
        man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = man["scene"]["episode_index"]
        seed = int(np.load(rd / "traj.npz")["policy_seed"])
        stt = rd.parent / f"{rd.name}.start"
        cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
        L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; hurdle = sorted(L["obstacle_manager"].collision_geom_ids)
        hbodies = {int(m.geom_bodyid[g]) for g in hurdle}
        coll = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        fgeoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid and coll(g)]
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"
        bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or f"body#{int(b)}"
        ft = np.zeros(6)

        def setq(q):
            d.qpos[:] = q; mujoco.mj_forward(m, d)

        act = L["act"]; spa = max(1, int(round(1.0 / (cl.CTRL_HZ * m.opt.timestep)))); n7 = 7
        for _ in range(1000):
            mujoco.mj_step(m, d)
        mism = 0; best = None
        for tt in sorted(ctrl):
            if tt > t:
                break
            r = ctrl[tt]
            if np.max(np.abs(np.asarray(r["qpos"]) - d.qpos)) != 0.0:
                mism += 1
            ac = r["applied_ctrl"]
            for i in range(n7):
                d.ctrl[act["left_a"][i]] = ac["arm"][i]; d.ctrl[act["right_a"][i]] = ac["arm"][n7 + i]
            d.ctrl[act["left_grip_a"]] = ac["gripper"][0] * cl.RBY1_GRIPPER_OPEN; d.ctrl[act["right_grip_a"]] = ac["gripper"][1] * cl.RBY1_GRIPPER_OPEN
            for ss in range(spa):
                if tt == t:
                    for fg_ in fgeoms:
                        for h in hurdle:
                            v = float(mujoco.mj_geomDistance(m, d, fg_, h, 0.3, ft))
                            if best is None or v < best[1]:
                                best = (f"substep{ss}", v, d.qpos.copy(), ft[:3].copy(), ft[3:].copy(), fg_, h)
                mujoco.mj_step(m, d)
            if tt == t:
                for fg_ in fgeoms:
                    for h in hurdle:
                        v = float(mujoco.mj_geomDistance(m, d, fg_, h, 0.3, ft))
                        if v < best[1]:
                            best = ("post", v, d.qpos.copy(), ft[:3].copy(), ft[3:].copy(), fg_, h)
        which, dist, q_c, pf, pb, fg, hg = best
        # bar capsule
        bar = [g for g in hurdle if int(m.geom_type[g]) == 3][0]
        cbar = d.geom_xpos[bar].copy(); ax = d.geom_xmat[bar].reshape(3, 3)[:, 2].copy(); rbar = float(m.geom_size[bar][0]); hl = float(m.geom_size[bar][1])
        a_pb = float((pb - cbar) @ ax)
        e1 = np.cross(ax, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
        offs = np.arange(-0.030, 0.0301, 0.005); ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        S_w = np.array([cbar + (a_pb + o) * ax + rbar * (np.cos(th) * e1 + np.sin(th) * e2) for o in offs for th in ang])
        seq_x = int(ctrl[t]["chunk_seq"])
        ev = {"key": key, "t": t, "mjcf_body": body, "state": which, "replay_qpos_mismatch_rows": mism, "finger_bar_mesh_mm": 1e3 * dist,
              "finger_geom": gname(fg), "hurdle_geom": gname(hg), "pf_world": pf.tolist(), "pb_world": pb.tolist(),
              "bar": {"geom": gname(bar), "centre_w": cbar.tolist(), "axis_w": ax.tolist(), "radius_mm": 1e3 * rbar, "half_len_mm": 1e3 * hl, "a_pb_mm": 1e3 * a_pb},
              "exec_seq": seq_x, "step_in_chunk": ctrl[t].get("step_in_chunk"), "grasp_t": a.get("grasp_t"), "place_t": a.get("place_t")}
        # scene + filter model (once), reused per chunk
        setq(q_c)
        scene = TransportScene.attach(m, d); rm = build_robot_model(scene)
        hist = []
        for sq in sorted(chunks):
            if sq not in plan:
                continue
            q_obs = np.asarray(plan[sq]["qpos"], float); setq(q_obs)
            set_finger_joints_from_scene(scene, rm); rs = scene.robot_state()
            Tb = C.T_world_base(m, d); pbb = C.to_base(Tb, pb)[0]; Sb = C.to_base(Tb, S_w)
            with np.load(chunks[sq], allow_pickle=True) as z:
                s = json.loads(str(z["summary_json"])); es = s.get("esdf_stats") or {}
                rec = {"seq": sq, "t": plan[sq]["t_step"]}
                g = s.get("grasp") or {}; att = C.lit(g.get("attached")); sub = s.get("subtask") or {}; mp = s.get("manipulated") or {}
                rec.update(grasp_state=g.get("state"), attached=isinstance(att, dict), subtask=sub.get("label"), gate=sub.get("gate"),
                           blocked=sub.get("blocked"), released=sub.get("released"), placed_seen=sub.get("placed_seen"),
                           manip_id=mp.get("id"), manip_state=mp.get("state"), manip_centroid=mp.get("centroid"), manip_switched_from=mp.get("switched_from"),
                           excl_source=(s.get("exclusion") or {}).get("source"), excl_mech=(s.get("exclusion") or {}).get("mechanism"),
                           has_target=s.get("has_target"), label_names=es.get("label_names"))
                # held filter / held free spheres
                hf_c = hf_r = hq_c = hq_r = None
                if isinstance(att, dict):
                    held = C.lit(att.get("held")) or {}; fl = held.get("frame_link") or att.get("parent_link")
                    Te = C.body_T(m, d, HOLD_BODY.get(fl, "EE_BODY_L"))
                    hf_c, hf_r = spheres_world(Te, held.get("filter_spheres_parent_frame"))
                    hq_c, hq_r = spheres_world(Te, held.get("spheres_parent_frame"))
                    if hq_c is not None:
                        sdS = C.union_sd(S_w, hq_c, hq_r); rec["heldfree_S_sd_min_mm"] = 1e3 * float(sdS.min()); rec["n_S_in_heldfree"] = int((sdS <= 0).sum())
                        rec["heldfree_pb_sd_mm"] = 1e3 * float(C.union_sd(pb[None], hq_c, hq_r)[0])
                    ff = held.get("first_free_spheres")
                    if ff:
                        try:
                            fc = np.array([np.asarray(x["centre_m"], float) for x in ff]); fr_ = np.array([1e-3 * float(x["radius_mm"]) for x in ff])
                            sdF = C.union_sd(Sb, fc, fr_); rec["firstfree_S_sd_min_mm_baseframe"] = 1e3 * float(sdF.min()); rec["n_first_free"] = len(ff)
                        except Exception as ex:  # noqa: BLE001
                            rec["firstfree_err"] = repr(ex)[:120]
                hfr = C.lit(es.get("held_free")) or {}
                rec["held_free_n_points"] = hfr.get("n_points") if isinstance(hfr, dict) else None
                rec["held_free_n_voxels"] = hfr.get("n_voxels_freed") if isinstance(hfr, dict) else None
                # cameras
                cams = {}
                for cam in CAMS:
                    fr = scene.capture(cam); K = fr.camera_intrinsics; T = fr.T_base_cam; Tc = np.linalg.inv(T)
                    pc_ = Tc[:3, :3] @ pbb + Tc[:3, 3]; e = {"pb_z_m": float(pc_[2]), "pb_in_image": False}
                    if pc_[2] > 0:
                        u = K[0, 0] * pc_[0] / pc_[2] + K[0, 2]; v = K[1, 1] * pc_[1] / pc_[2] + K[1, 2]; H_, W_ = fr.depth.shape
                        if 0 <= u < W_ and 0 <= v < H_:
                            dpx = float(fr.depth[int(v), int(u)]); g_ = int(fr.geom_ids[int(v), int(u)])
                            e.update(pb_in_image=True, pb_u=float(u), pb_v=float(v), pb_occluded=bool(dpx < pc_[2] - 0.005),
                                     pb_body_at_px=None if g_ < 0 else bname(m.geom_bodyid[g_]))
                    cloud = backproject(fr.depth, K, T, PCFG); uv = cloud.uv; P = cloud.points
                    isbar = np.isin(fr.body_ids[uv[:, 1], uv[:, 0]], list(hbodies))
                    Pb = P[isbar]; Pw = Pb @ Tb[:3, :3].T + Tb[:3, 3]
                    hs = C.hurdle_sd(m, d, hurdle, Pw) if len(Pw) else np.zeros(0)
                    keep = hs < 0.010; Pb, Pw = Pb[keep], Pw[keep]
                    near = np.linalg.norm(Pw - pb, axis=1) < 0.030
                    rad = np.linalg.norm(np.cross(Pw - cbar, ax), axis=1); onbar = rad < rbar + 0.010
                    axo = (Pw - cbar) @ ax - a_pb
                    rf = self_filter_mask(Pb, rm, rs, PCFG)[0] if len(Pb) else np.zeros(0, bool)
                    hfm = (C.union_sd(Pw, hf_c, hf_r) <= 0) if (hf_c is not None and len(Pw)) else np.zeros(len(Pw), bool)
                    vis = ~rf & ~hfm
                    e.update(n_bar_px=int(len(Pb)), n_bar_px_vis=int(vis.sum()), n_near=int(near.sum()), n_near_robot_filtered=int((near & rf).sum()),
                             n_near_held_filtered=int((near & hfm & ~rf).sum()), n_near_vis=int((near & vis).sum()),
                             bins_all=np.histogram(axo[onbar], BINS)[0].tolist(), bins_vis=np.histogram(axo[onbar & vis], BINS)[0].tolist())
                    if sq == seq_x:
                        e["n_masked_px_record"] = ((C.lit(es.get("per_camera")) or {}).get(REC_CAM[cam]) or {}).get("n_masked_px")
                    cams[cam] = e
                rec["cams"] = cams
                rec["vis_near_total"] = int(sum(c["n_near_vis"] for c in cams.values()))
                rec["near_total"] = int(sum(c["n_near"] for c in cams.values()))
                # clouds
                for nm in ("raw_cloud", "filtered_cloud"):
                    Pc = np.asarray(z[nm], float)
                    if len(Pc):
                        dd = np.linalg.norm(Pc - pbb, axis=1); sel = dd < 0.060
                        hs = np.full(len(Pc), 1.0); hs[sel] = C.hurdle_sd(m, d, hurdle, Pc[sel] @ Tb[:3, :3].T + Tb[:3, 3])
                        rec[f"n_{nm.split('_')[0]}_near"] = int(((hs < 0.010) & (dd < 0.030)).sum())
                    else:
                        rec[f"n_{nm.split('_')[0]}_near"] = 0
                rec["raw_eq_filtered"] = bool(z["raw_cloud"].shape == z["filtered_cloud"].shape and np.array_equal(z["raw_cloud"], z["filtered_cloud"]))
                # grids
                for nm, pre in (("fine", "esdf"), ("coarse", "esdf_coarse"), ("free", "esdf_free")):
                    if f"{pre}_distance" not in z.files:
                        rec[f"{nm}_pb_mm"] = None; continue
                    o, vx = z[f"{pre}_origin"], float(z[f"{pre}_voxel_size"])
                    rec[f"{nm}_pb_mm"] = 1e3 * float(C.samp(z[f"{pre}_distance"], o, vx, pbb)[0])
                    if nm in ("fine", "free"):
                        vS = C.samp(z[f"{pre}_distance"], o, vx, Sb)
                        rec[f"{nm}_S_min_mm"] = None if np.all(np.isnan(vS)) else 1e3 * float(np.nanmin(vS))
                        rec[f"{nm}_S_n_inside_grid"] = int((~np.isnan(vS)).sum())
                    rec[f"{nm}_pb_inside"] = C.grid_index(o, vx, z[f"{pre}_shape"], pbb)[1]
                # carve-out ball
                tf = C.lit(es.get("target_free")) or {}; ball = tf.get("ball") if isinstance(tf, dict) else None
                rec["tf_mode"] = tf.get("mode") if isinstance(tf, dict) else None; rec["tf_n_layers"] = tf.get("n_layers") if isinstance(tf, dict) else None
                if isinstance(ball, dict) and ball.get("centre_m") is not None:
                    bc = np.asarray(ball["centre_m"], float); R = 1e-3 * float(ball["radius_mm"]); zmin = ball.get("z_min_m")
                    inS = (np.linalg.norm(Sb - bc, axis=1) <= R) & (Sb[:, 2] >= (zmin if zmin is not None else -9))
                    rec.update(ball_radius_mm=float(ball["radius_mm"]), ball_centre=bc.tolist(), ball_z_min=zmin,
                               pb_minus_ball_mm=1e3 * float(np.linalg.norm(pbb - bc)) - float(ball["radius_mm"]), n_S_in_ball=int(inS.sum()))
                tp = np.asarray(z["target_points"], float) if "target_points" in z.files else np.zeros((0, 3))
                rec["n_target_points"] = int(len(tp))
                if len(tp):
                    hs = C.hurdle_sd(m, d, hurdle, tp @ Tb[:3, :3].T + Tb[:3, 3])
                    rec["n_target_within_20mm_hurdle"] = int((hs < 0.020).sum()); rec["target_centroid_base"] = tp.mean(0).tolist()
                if sq == seq_x:
                    field = FIA.build_field(z, outside); pfb = C.to_base(Tb, pf)[0]
                    x = {}
                    for nm, P1 in (("pb", pbb), ("pf", pfb)):
                        P1 = P1[None]
                        x[f"composed_full_{nm}_mm"] = 1e3 * float(np.asarray(field.distance(P1))[0])
                        x[f"composed_full_{nm}_tier"] = str(np.asarray(field.answer_tier(P1, target_free=False)).reshape(-1)[0])
                        if field.target_free_layers:
                            x[f"composed_free_{nm}_mm"] = 1e3 * float(np.asarray(field.target_free_distance(P1))[0])
                            x[f"composed_free_{nm}_tier"] = str(np.asarray(field.answer_tier(P1, target_free=True)).reshape(-1)[0])
                    x["n_target_free_layers"] = len(field.target_free_layers or ())
                    x["tsdf_truncation"] = es.get("tsdf_truncation"); x["decay"] = es.get("decay"); x["frames"] = es.get("frames")
                    x["fine_origin"] = np.asarray(z["esdf_origin"]).tolist(); x["fine_voxel"] = float(z["esdf_voxel_size"]); x["fine_shape"] = np.asarray(z["esdf_shape"]).tolist()
                    x["coarse_origin"] = np.asarray(z["esdf_coarse_origin"]).tolist(); x["coarse_voxel"] = float(z["esdf_coarse_voxel_size"])
                    x["window"] = (lambda w: {k: w.get(k) for k in ("placement", "lower_m", "upper_m")} if isinstance(w, dict) else None)(C.lit(es.get("window")))
                    x["pf_base"] = pfb.tolist(); x["pb_base"] = pbb.tolist()
                    # record clearance (safe_policy.py:1266-1275, full field - r - esdf_margin, refined trajectory) at row k, per URDF link;
                    # MJCF->URDF finger map measured by oc_namemap.py (left hand swapped, right hand identity)
                    names = C.lit(s.get("sphere_link_names")) or []; k = int(ctrl[t].get("step_in_chunk") or 0)
                    urdf = {"ee_finger_l1": "ee_finger_l2", "ee_finger_l2": "ee_finger_l1"}.get(body, body)
                    x["urdf_link"] = urdf
                    if names and "clearance" in z.files:
                        Cl = np.asarray(z["clearance"], float)
                        for ln in sorted(set(names)):
                            if ln.startswith("ee_finger") or ln == body or ln == urdf:
                                msk = np.array([n == ln for n in names]); x[f"rec_clear_k_{ln}_mm"] = 1e3 * float(Cl[min(k, len(Cl) - 1)][msk].min())
                    met = C.lit((s.get("to") or {}).get("metrics")) or {}
                    x["to_clearance_mm"] = None if met.get("clearance_m") is None else 1e3 * float(met["clearance_m"])
                    x["to_status"] = (s.get("to") or {}).get("status"); x["verdict"] = s.get("verdict")
                    ev["exec"] = x
            hist.append(rec)
        scene.close()
        ev["history"] = hist; ev["sec"] = round(time.time() - t0, 1)
        res["events"].append(ev)
        hx = next(h for h in hist if h["seq"] == seq_x)
        print(key, t, body, "mesh %.2f" % ev["finger_bar_mesh_mm"], "exec", seq_x, "fine_pb %.1f" % hx["fine_pb_mm"], "vis_near", hx["vis_near_total"],
              "raw_near", hx["n_raw_near"], "tf", hx.get("tf_mode"), "sec", ev["sec"], flush=True)
        out_path.write_text(json.dumps(res, indent=1, default=float))
    out_path.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
