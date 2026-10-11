"""T43 OC — V8_ep1807_s18073 chunk seq 22 (rows t 168-175; TK apple-bar contact t 174 k 6): TO clearance -0.9 mm, executed.
(verifier, CPU, recorded data). usage: T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8 oc_m09.py OUT_JSON
At the observation state (planning-row qpos of seq 22, world <- base from MuJoCo body 'base'):
  worst row (verdict evidence): point_m (base frame, centre of URDF ee_finger_l1 sphere, step 0 of the refined trajectory), radius r;
  geometric signed distances from that point (world) to the hurdle collision geoms, to the apple collision geoms, to the table
  (all minus r -> sphere surface); fine / coarse recorded ESDF trilinear at the point and the composed field (FIA.build_field);
  d_obj = distance to the union of held spheres (held.spheres_parent_frame at MJCF EE_BODY_L = scene.manipulated_spheres rule,
  linearize._manipulated_distance) and of held.body_cover lifted/body sets; record clearance[0, query] (= d_full - r - esdf_margin 0);
  implied margin = d_full - r - to_clearance.
  rule check (safe_policy._classify_esdf_rows): |d_row - d_obj| <= voxel(fine 5 mm) -> on_object, tier fine -> allowed."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p'))
import numpy as np, mujoco
import hold_w as HW, dx_common as C
from AG3S.trajopt.experiments import fixed_input_ablation as FIA
C.ANALYSES["V8"] = C.T43 / "V8/analysis/analysis.json"; C.ARM_SRC["V8"] = "V8"
out = pathlib.Path(sys.argv[1]); key = "V8/H/V8_ep1807_s18073"; SEQ = 22
cl = HW.load_client(); a = C.run_entry(key); rd = pathlib.Path(a["run_dir"]); ctrl, plan = C.frames(rd); chunks = C.segment_chunks(a["server_segment"])
st, outside, om, logp = C.server_settings(a["server_segment"], rd / "manifest.json")
man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = man["scene"]["episode_index"]
seed = int(np.load(rd / "traj.npz")["policy_seed"]); stt = rd.parent / f"{rd.name}.start"
cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; hurdle = sorted(L["obstacle_manager"].collision_geom_ids)
z = np.load(chunks[SEQ], allow_pickle=True); s = json.loads(str(z["summary_json"]))
v = s["verdict"]; row = v["reasons"][0]["evidence"]["row"]; q = int(row["query"]); P_b = np.asarray(row["point_m"], float)
names = C.lit(s["sphere_link_names"]); r = float(z["sphere_radii"][q])
d.qpos[:] = np.asarray(plan[SEQ]["qpos"], float); mujoco.mj_forward(m, d); Tb = C.T_world_base(m, d); P_w = Tb[:3, :3] @ P_b + Tb[:3, 3]
apple_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
ag = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == apple_b and (m.geom_contype[g] or m.geom_conaffinity[g])]
def geom_sd_any(g, p):
    try:
        return float(C.geom_point_sd(m, d, g, p[None])[0])
    except ValueError:  # mesh: use mj_geomDistance with a tiny sphere is not available -> distance to geom AABB centre minus rbound
        return float(np.linalg.norm(p - d.geom_xpos[g]) - m.geom_rbound[g])
res = {"definitions": __doc__, "key": key, "seq": SEQ, "t_rows": [plan[SEQ]["t_step"], plan[SEQ]["t_step"] + 7],
       "verdict": v, "to_status": s["to"]["status"], "to_notes": s["to"].get("notes"),
       "to_clearance_mm": 1e3 * float(C.lit(s["to"]["metrics"])["clearance_m"]) if isinstance(s["to"].get("metrics"), dict) else None,
       "query": q, "link": names[q], "radius_mm": 1e3 * r, "point_base": P_b.tolist(), "point_world": P_w.tolist(),
       "apple_geom_types": [int(m.geom_type[g]) for g in ag]}
res["hurdle_sd_minus_r_mm"] = 1e3 * (float(C.hurdle_sd(m, d, hurdle, P_w[None])[0]) - r)
res["apple_sd_minus_r_mm"] = 1e3 * (min(geom_sd_any(g, P_w) for g in ag) - r) if ag else None
res["apple_centre_dist_mm"] = 1e3 * float(np.linalg.norm(P_w - d.xpos[apple_b]))
ft = np.zeros(6); fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_finger_l2")
fg = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == fb and (m.geom_contype[g] or m.geom_conaffinity[g])]
res["mjcf_ee_finger_l2_to_hurdle_mm_obs"] = 1e3 * min(float(mujoco.mj_geomDistance(m, d, g, h, 0.3, ft)) for g in fg for h in hurdle)
res["mjcf_ee_finger_l2_to_apple_mm_obs"] = 1e3 * min(float(mujoco.mj_geomDistance(m, d, g, h, 0.3, ft)) for g in fg for h in ag) if ag else None
res["apple_to_hurdle_mm_obs"] = 1e3 * min(float(mujoco.mj_geomDistance(m, d, g, h, 0.3, ft)) for g in ag for h in hurdle) if ag else None
for nm, pre in (("fine", "esdf"), ("coarse", "esdf_coarse")):
    res[f"{nm}_d_mm"] = 1e3 * float(C.samp(z[f"{pre}_distance"], z[f"{pre}_origin"], float(z[f"{pre}_voxel_size"]), P_b)[0])
field = FIA.build_field(z, outside); res["composed_d_mm"] = 1e3 * float(np.asarray(field.distance(P_b[None]))[0])
res["composed_tier"] = str(np.asarray(field.answer_tier(P_b[None], target_free=False)).reshape(-1)[0])
res["rec_clearance_step0_mm"] = 1e3 * float(z["clearance"][0, q])
res["implied_margin_mm"] = res["composed_d_mm"] - res["radius_mm"] - res["to_clearance_mm"]
held = C.lit(s["grasp"]["attached"]["held"]); Te = C.body_T(m, d, "EE_BODY_L")
for nm, Ls in (("spheres_parent_frame", held.get("spheres_parent_frame")), ("body_parent_frame", (held.get("body_cover") or {}).get("body_parent_frame")),
               ("lifted_parent_frame", (held.get("body_cover") or {}).get("lifted_parent_frame"))):
    if Ls:
        Cw = np.array([Te[:3, :3] @ np.asarray(x["centre_m"], float) + Te[:3, 3] for x in Ls]); rr = np.array([1e-3 * x["radius_mm"] for x in Ls])
        res[f"d_obj_{nm}_mm"] = 1e3 * float(C.union_sd(P_w[None], Cw, rr)[0])
        res[f"held_{nm}_to_hurdle_min_mm"] = 1e3 * float(min(C.hurdle_sd(m, d, hurdle, Cw[i][None])[0] - rr[i] for i in range(len(rr))))
res["d_obj_record_mm"] = 1e3 * float(row["object_distance_m"])
res["abs_drow_minus_dobj_mm"] = abs(res["composed_d_mm"] - res["d_obj_record_mm"]); res["voxel_mm"] = 1e3 * float(z["esdf_voxel_size"])
res["tolerance_m"] = v["classification"]["tolerance_m"]; res["obstacle_margin_mm_log"] = om
res["planning_qpos_vs_refined_step0_note"] = "point_m is refined step 0 (anchor |Q0 - q_now| <= max_step, problem.py:256-257)"
out.write_text(json.dumps(res, indent=1, default=float)); print(json.dumps({k: v_ for k, v_ in res.items() if k not in ("definitions", "verdict")}, indent=1, default=float))
