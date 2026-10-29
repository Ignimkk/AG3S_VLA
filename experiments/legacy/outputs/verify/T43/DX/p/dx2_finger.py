"""T43 DX-2 — the 8 finger-hurdle contacts where the recorded field had no bar (FC offline (a): sees_old and sees_new false, field
present; d - r +4…+56 mm). (verifier, CPU only, recorded data only; MUJOCO_GL=osmesa JAX_PLATFORMS=cpu)

usage: T43W_SNAP=<snapshot> dx2_finger.py OUT_JSON KEY:T:MJCF_BODY [KEY:T:MJCF_BODY ...]

Contact: physics replay of the run up to row t (CT replay_ct method; replay_qpos_mismatch_rows = rows whose recorded qpos differs
  from the replay pre-step qpos); the sub-step of row t (or its post state) with the smallest mj_geomDistance(finger collision
  geoms of MJCF body, hurdle geoms) is the contact state; fromto -> finger point pf, bar point pb (world).
Hurdle: client reset (hold_w.reset) of the run, manager collision geoms (bar capsule + 2 leg cylinders, mocap pose).
Chunks: exec = server chunk with t_step == control row chunk_seq (the executing chunk, its field = control row 'field');
  fc = latest planning row with t_step <= t (fc_offline rule). Observation state = planning-row qpos of the exec chunk.
Cloud (exec chunk npz, base frame = MuJoCo body 'base'): raw_cloud, filtered_cloud, support_mask.
  n_raw_r{10,20,30,50}: raw points within r mm of pb; n_raw_bar_near: raw points with hurdle signed distance < 10 mm and within
  60 mm of pb; n_raw_bar_all: raw points with hurdle signed distance < 10 mm anywhere; same for filtered; raw_eq_filtered:
  arrays identical; n_support_near: support_mask among the r30 points.
Cameras (zed_left = head, wrist_cam_l, wrist_cam_r), rendered with TransportScene.attach(m, d).capture (480x640, the client's
  capture path) at the observation state and at the contact state:
  proj: pb in that camera (OpenCV frame): z, pixel (u, v), in_image; depth at that pixel; occluded = depth < z - 5 mm;
        geom / body at that pixel (segmentation).
  n_bar_px_near: pixels whose segmentation body is a hurdle body (visual geoms included) and whose back-projected point (benchmark backproject,
        PointCloudConfig(range_max=2.0) = server pointcloud) lies within 30 mm of pb;
  n_bar_px_near_selffiltered: of those, inside the self-filter (robot_filter.self_filter_mask, model =
        grounding_report.build_robot_model(scene) = serve_safe filter_robot, finger joints from the scene, PointCloudConfig code
        default inflation; nothing held is added -> exact only when the chunk is not attached);
  n_masked_px_repro: total self-filtered pixels of that camera (fidelity vs esdf_stats.per_camera.n_masked_px of the record).
ESDF of the exec chunk: grids fine (esdf_*), coarse (esdf_coarse_*), target-free fine (esdf_free_*): voxel, origin, shape,
  grid index of pb and pf and inside flag; trilinear values at pb / pf; composed field (FIA.build_field) distance and
  target_free_distance with answer_tier. TSDF: esdf_stats.tsdf_truncation, tiers.
  target-free layer: esdf_stats.target_free (mode, label, ball centre / radius, n_ball_seeds_excluded); |pb - ball centre| - radius
  (< 0: pb inside the carve-out ball); target_points: count with hurdle signed distance < 20 mm, min hurdle signed distance;
  exclusion (source, authorized links), grasp state, manipulated id / state.
History: for every chunk of the run, trilinear fine full / fine target-free / coarse values at pb (bar point at the contact)
  and the target-free ball containment -> raw history list."""
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

CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
PCFG = PointCloudConfig(range_max=2.0)


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
        hset = set(hurdle); hbodies = {int(m.geom_bodyid[g]) for g in hurdle}
        coll = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        fgeoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid and coll(g)]
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"
        bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or f"body#{int(b)}"
        ft = np.zeros(6)

        def setq(q):
            d.qpos[:] = q; mujoco.mj_forward(m, d)

        # physics replay (CT replay_ct method: warm-up 1000 mj_steps, every control row's applied_ctrl for spa sub-steps) up to row t;
        # contact state = the sub-step of row t with the smallest finger-hurdle distance (full qpos saved).
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
        replay_mism = mism
        which, dist, q_c, pf, pb, fg, hg = best
        seq = int(ctrl[t]["chunk_seq"]); fc_seq = max(sq for sq, p in plan.items() if p["t_step"] <= t)
        z = np.load(chunks[seq], allow_pickle=True); s = json.loads(str(z["summary_json"])); field = FIA.build_field(z, outside)
        q_obs = np.asarray(plan[seq]["qpos"], float)
        ev = {"key": key, "t": t, "mjcf_body": body, "urdf_link": {"ee_finger_l2": "ee_finger_l1", "ee_finger_l1": "ee_finger_l2"}.get(body, body),
              "state": which, "replay_qpos_mismatch_rows": replay_mism, "finger_bar_mesh_mm": 1e3 * dist, "finger_geom": gname(fg), "hurdle_geom": gname(hg),
              "pf_world": pf.tolist(), "pb_world": pb.tolist(), "exec_seq": seq, "fc_seq": fc_seq, "exec_t_step": plan[seq]["t_step"],
              "step_in_chunk": ctrl[t]["step_in_chunk"], "control_field": ctrl[t].get("field"), "plan_field_frame": (plan[seq].get("field") or {}).get("frame_index")}
        setq(q_c); Tb_c = C.T_world_base(m, d)
        setq(q_obs); Tb = C.T_world_base(m, d)
        ev["base_moved_mm"] = 1e3 * float(np.linalg.norm(Tb[:3, 3] - Tb_c[:3, 3]))
        pbb = C.to_base(Tb, pb)[0]; pfb = C.to_base(Tb, pf)[0]
        # ---- cloud
        R_ = np.asarray(z["raw_cloud"], float); F_ = np.asarray(z["filtered_cloud"], float); S_ = np.asarray(z["support_mask"], bool)
        Tw = Tb  # cloud base -> world with the observation base pose
        cl_ = {"n_raw": int(len(R_)), "n_filtered": int(len(F_)), "raw_eq_filtered": bool(R_.shape == F_.shape and np.array_equal(R_, F_))}
        for nm, P in (("raw", R_), ("filtered", F_)):
            dd = np.linalg.norm(P - pbb, axis=1); Pw = P @ Tw[:3, :3].T + Tw[:3, 3]; hs = C.hurdle_sd(m, d, hurdle, Pw)
            for r_ in (10, 20, 30, 50):
                cl_[f"n_{nm}_r{r_}"] = int((dd < r_ * 1e-3).sum())
            cl_[f"n_{nm}_bar_near"] = int(((hs < 0.010) & (dd < 0.060)).sum()); cl_[f"n_{nm}_bar_all"] = int((hs < 0.010).sum())
            cl_[f"min_dist_to_pb_{nm}_mm"] = 1e3 * float(dd.min()) if len(dd) else None
            if nm == "raw":
                cl_["n_support_r30"] = int((S_[dd < 0.030]).sum()) if len(S_) == len(P) else None
        ev["cloud"] = cl_
        # ---- cameras
        es = s.get("esdf_stats") or {}; pc = C.lit(es.get("per_camera")) or {}
        cams = {}
        for tag, qq in (("obs", q_obs), ("contact", q_c)):
            setq(qq); scene = TransportScene.attach(m, d); rm = build_robot_model(scene); set_finger_joints_from_scene(scene, rm); rs = scene.robot_state()
            Tb_ = C.T_world_base(m, d); pb_ = C.to_base(Tb_, pb)[0]
            for cam in CAMS:
                fr = scene.capture(cam); K = fr.camera_intrinsics; T = fr.T_base_cam; Tc = np.linalg.inv(T)
                pc_ = Tc[:3, :3] @ pb_ + Tc[:3, 3]; e = {"z_m": float(pc_[2])}
                if pc_[2] > 0:
                    u = K[0, 0] * pc_[0] / pc_[2] + K[0, 2]; v = K[1, 1] * pc_[1] / pc_[2] + K[1, 2]
                    e.update(u=float(u), v=float(v)); H_, W_ = fr.depth.shape
                    inimg = 0 <= u < W_ and 0 <= v < H_; e["in_image"] = bool(inimg)
                    if inimg:
                        dpx = float(fr.depth[int(v), int(u)]); g_ = int(fr.geom_ids[int(v), int(u)])
                        e.update(depth_at_px_m=dpx, occluded=bool(dpx < pc_[2] - 0.005), geom_at_px=None if g_ < 0 else gname(g_),
                                 body_at_px=None if g_ < 0 else bname(m.geom_bodyid[g_]), px_is_hurdle=bool(g_ >= 0 and int(m.geom_bodyid[g_]) in hbodies))
                else:
                    e["in_image"] = False
                cloud = backproject(fr.depth, K, T, PCFG); uv = cloud.uv; P = cloud.points
                bpx = fr.body_ids[uv[:, 1], uv[:, 0]]; isbar = np.isin(bpx, list(hbodies)); near = isbar & (np.linalg.norm(P - pb_, axis=1) < 0.030)
                inside, _ = self_filter_mask(P, rm, rs, PCFG)
                e.update(n_bar_px=int(isbar.sum()), n_bar_px_near=int(near.sum()), n_bar_px_near_selffiltered=int((near & inside).sum()),
                         n_bar_px_selffiltered=int((isbar & inside).sum()), n_masked_px_repro=int(inside.sum()))
                if tag == "obs":
                    e["n_masked_px_record"] = (pc.get({"zed_left": "head", "wrist_cam_l": "left_wrist", "wrist_cam_r": "right_wrist"}[cam]) or {}).get("n_masked_px")
                cams[f"{tag}:{cam}"] = e
            scene.close()
        ev["cameras"] = cams
        # ---- ESDF
        e = {"tsdf_truncation": es.get("tsdf_truncation"), "tiers": es.get("tiers"), "frames": es.get("frames"), "label_names": es.get("label_names"),
             "held_free": es.get("held_free"), "decay": es.get("decay"), "window": (lambda w: {k: w.get(k) for k in ("placement", "lower_m", "upper_m", "voxel_size")} if isinstance(w, dict) else None)(C.lit(es.get("window")))}
        for nm, pre in (("fine", "esdf"), ("coarse", "esdf_coarse"), ("free", "esdf_free")):
            if f"{pre}_distance" not in z.files:
                continue
            o, v, sh = z[f"{pre}_origin"], float(z[f"{pre}_voxel_size"]), z[f"{pre}_shape"]
            gi_b, in_b = C.grid_index(o, v, sh, pbb); gi_f, in_f = C.grid_index(o, v, sh, pfb)
            e[nm] = {"voxel_m": v, "origin_m": np.asarray(o).tolist(), "shape": np.asarray(sh).tolist(),
                     "upper_m": (np.asarray(o) + v * (np.asarray(sh) - 1)).tolist(),
                     "pb_index": gi_b, "pb_inside": in_b, "pf_index": gi_f, "pf_inside": in_f,
                     "d_pb_mm": 1e3 * float(C.samp(z[f"{pre}_distance"], o, v, pbb)[0]), "d_pf_mm": 1e3 * float(C.samp(z[f"{pre}_distance"], o, v, pfb)[0])}
        for nm, P in (("pb", pbb), ("pf", pfb)):
            P1 = P[None]
            e[f"composed_full_{nm}_mm"] = 1e3 * float(np.asarray(field.distance(P1))[0])
            e[f"composed_full_{nm}_tier"] = str(np.asarray(field.answer_tier(P1, target_free=False)).reshape(-1)[0])
            if field.target_free_layers:
                e[f"composed_free_{nm}_mm"] = 1e3 * float(np.asarray(field.target_free_distance(P1))[0])
                e[f"composed_free_{nm}_tier"] = str(np.asarray(field.answer_tier(P1, target_free=True)).reshape(-1)[0])
        tf = C.lit(es.get("target_free")) or {}
        ball = tf.get("ball") if isinstance(tf, dict) else None
        e["target_free"] = {k: tf.get(k) for k in ("n_layers", "mode", "label", "n_target_points", "n_ball_seeds_excluded")} if isinstance(tf, dict) else None
        if isinstance(ball, dict) and ball.get("centre_m") is not None:
            bc = np.asarray(ball["centre_m"], float)
            e["ball"] = {"centre_m": bc.tolist(), "radius_mm": ball.get("radius_mm"), "radius_raw_mm": ball.get("radius_raw_mm"), "centre_method": ball.get("centre_method"),
                         "fit_reason": (ball.get("fit") or {}).get("reason"), "pb_dist_minus_radius_mm": 1e3 * float(np.linalg.norm(pbb - bc)) - float(ball["radius_mm"]),
                         "pf_dist_minus_radius_mm": 1e3 * float(np.linalg.norm(pfb - bc)) - float(ball["radius_mm"]), "z_min_m": ball.get("z_min_m")}
        tp = np.asarray(z["target_points"], float) if "target_points" in z.files else np.zeros((0, 3))
        if len(tp):
            hs = C.hurdle_sd(m, d, hurdle, tp @ Tb[:3, :3].T + Tb[:3, 3])
            e["target_points"] = {"n": int(len(tp)), "n_within_20mm_of_hurdle": int((hs < 0.020).sum()), "min_hurdle_sd_mm": 1e3 * float(hs.min()),
                                  "centroid_m": tp.mean(0).tolist(), "extent_mm": (1e3 * (tp.max(0) - tp.min(0))).tolist()}
        mp = s.get("manipulated") or {}
        e["manipulated"] = {"id": mp.get("id"), "state": mp.get("state"), "centroid": mp.get("centroid"), "radius": mp.get("radius"), "n_points": mp.get("n_points"),
                            "extents_mm": (mp.get("admissibility") or {}).get("extents_mm")}
        e["exclusion"] = s.get("exclusion"); e["grasp_state"] = (s.get("grasp") or {}).get("state"); e["has_target"] = s.get("has_target")
        e["subtask"] = (s.get("subtask") or {}).get("label")
        ev["esdf"] = e
        # ---- history at pb over all chunks of the run
        hist = []
        for sq in sorted(chunks):
            with np.load(chunks[sq], allow_pickle=True) as zz:
                ss = json.loads(str(zz["summary_json"])); tf_ = C.lit((ss.get("esdf_stats") or {}).get("target_free")) or {}
                b_ = tf_.get("ball") if isinstance(tf_, dict) else None
                rec = {"seq": sq, "t": plan[sq]["t_step"] if sq in plan else None,
                       "fine_mm": 1e3 * float(C.samp(zz["esdf_distance"], zz["esdf_origin"], zz["esdf_voxel_size"], pbb)[0]),
                       "coarse_mm": 1e3 * float(C.samp(zz["esdf_coarse_distance"], zz["esdf_coarse_origin"], zz["esdf_coarse_voxel_size"], pbb)[0]),
                       "free_mm": 1e3 * float(C.samp(zz["esdf_free_distance"], zz["esdf_free_origin"], zz["esdf_free_voxel_size"], pbb)[0]) if "esdf_free_distance" in zz.files else None,
                       "tf_mode": tf_.get("mode") if isinstance(tf_, dict) else None,
                       "pb_in_ball": (None if not (isinstance(b_, dict) and b_.get("centre_m") is not None) else bool(1e3 * np.linalg.norm(pbb - np.asarray(b_["centre_m"])) < float(b_["radius_mm"]))),
                       "ball_radius_mm": None if not isinstance(b_, dict) else b_.get("radius_mm"), "grasp_state": (ss.get("grasp") or {}).get("state"),
                       "manipulated_id": (ss.get("manipulated") or {}).get("id")}
                hist.append(rec)
        ev["history"] = hist; ev["sec"] = round(time.time() - t0, 1)
        res["events"].append(ev)
        o = cams.get("obs:zed_left", {})
        print(key, t, body, "mesh %.2f" % ev["finger_bar_mesh_mm"], "exec/fc", seq, fc_seq, "raw r30", cl_["n_raw_r30"], "bar_near", cl_["n_raw_bar_near"],
              "head vis", o.get("in_image"), o.get("occluded"), "barpx", {k: v["n_bar_px_near"] for k, v in cams.items()},
              "fine in", e.get("fine", {}).get("pb_inside"), "full %.1f free %.1f" % (e["composed_full_pb_mm"], e.get("composed_free_pb_mm", np.nan)), "ball", (e.get("ball") or {}).get("pb_dist_minus_radius_mm"), flush=True)
        out_path.write_text(json.dumps(res, indent=1, default=float))
    out_path.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
