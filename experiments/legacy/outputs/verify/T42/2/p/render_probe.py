"""T42-2 invisible-obstacle diagnostic, offline part (verifier, CPU, MUJOCO_GL=osmesa). Snapshot ${AG3S_PROJECT_ROOT}-t42c client.
usage: render_probe.py EP CONFIG PLACE_PROFILE INVIS_PROFILE OUT_JSON   (CONFIG also holds <ep>_parked: obstacle model, no slot)
For each profile in (clear, PLACE_PROFILE, INVIS_PROFILE): the snapshot pi05_infer.main() up to load_remote_policy (stubbed, no
server; --obstacle-stop-distance 0) -> the client's (m, d) at the first observation (after reset, settle, obstacle activate).
Then, at that state:
  policy images : build_obs(m, d, renderer_pol) with the client's own renderer (299x224 -> 224, reflections off) for the three
                  policy cameras; pixel comparison vs clear (number of differing uint8 values, max |diff|)
  AG3S depth    : TransportScene.attach(m, d).capture(cam) (the --safe-remote client's depth / segmentation path) for
                  zed_left, wrist_cam_l, wrist_cam_r; comparison vs clear (pixels with |depth diff| > 1e-9 m)
  state         : qpos / qvel bit equality vs clear; d.ncon and contacts involving obstacle geoms
  model         : stat.extent / center, vis.map.znear / zfar, nbody / ngeom
  active check  : (invisible profile only) collision geoms of the slot: contype / conaffinity, rgba of the visual geoms; then the
                  slot moved onto the target fruit centre (mocap only) + mj_forward -> contacts between the slot's collision geoms
                  and the fruit (MuJoCo contact list) and manager.object_clearance()."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, inspect, io, contextlib, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/2/p'))
import plib as L
P = L._client()
import mujoco
ep, cfg, prof_v, prof_i, outp = sys.argv[1:6]


class Reached(Exception):
    pass


def capture(prof):
    cap = {}

    def stub(remote):
        cap.update(inspect.currentframe().f_back.f_locals); raise Reached
    P.load_remote_policy = stub
    sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1", "--episode-index", ep, "--headless",
                "--warmup-steps", "1000"] + ([] if prof == "clear" else ["--obstacle-config", cfg, "--obstacle-profile", prof, "--obstacle-stop-distance", "0"])
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            P.main()
    except Reached:
        pass
    return cap


def images(cap):
    m, d = cap["m"], cap["d"]
    obs = P.build_obs(cap["mcfg"]["obs_format"], m, d, cap["renderer_pol"], cap["idx"], "x")
    im = {k: np.asarray(v) for k, v in obs["images"].items()} if "images" in obs else {k: np.asarray(v) for k, v in obs.items() if "image" in k}
    sys.path.insert(0, "${AG3S_PROJECT_ROOT}-t42c")
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    sc = TransportScene.attach(m, d)
    dep = {c: np.asarray(sc.capture(c).depth, np.float64) for c in ("zed_left", "wrist_cam_l", "wrist_cam_r")}
    return im, dep


res = {"definitions": __doc__, "episode": int(ep), "config": cfg, "profiles": {}}
caps = {}; data = {}
for prof in ("clear", prof_v, prof_i):
    c = capture(prof); caps[prof] = c; m, d = c["m"], c["d"]
    im, dep = images(c); data[prof] = (im, dep, d.qpos.copy(), d.qvel.copy())
    mg = c.get("obstacle_manager")
    og = set(mg.collision_geom_ids) if mg else set()
    cons = [(int(d.contact[i].geom1), int(d.contact[i].geom2)) for i in range(d.ncon)]
    res["profiles"][prof] = {"model_xml": c["model_xml"], "nbody": int(m.nbody), "ngeom": int(m.ngeom), "stat_extent": float(m.stat.extent),
                             "stat_center": m.stat.center.tolist(), "znear": float(m.vis.map.znear), "zfar": float(m.vis.map.zfar),
                             "ncon": int(d.ncon), "n_contacts_with_obstacle": sum(1 for a, b in cons if a in og or b in og),
                             "active_slots": list(mg.active_slots) if mg else []}
im0, dep0, q0, v0 = data["clear"]
for prof in (prof_v, prof_i):
    im, dep, q, v = data[prof]
    r = res["profiles"][prof]
    r["qpos_equal_clear"] = bool(np.array_equal(q, q0)); r["qvel_equal_clear"] = bool(np.array_equal(v, v0))
    r["policy_images_vs_clear"] = {k: {"n_values_diff": int((im[k] != im0[k]).sum()), "max_abs": int(np.abs(im[k].astype(int) - im0[k].astype(int)).max())} for k in im0}
    r["depth_vs_clear"] = {k: {"n_pixels_diff_gt_1e-9": int((np.abs(dep[k] - dep0[k]) > 1e-9).sum()), "max_abs_m": float(np.abs(dep[k] - dep0[k]).max()),
                               "n_pixels": int(dep[k].size)} for k in dep0}
# parked profile (obstacle model, no active slot) vs clear and vs invisible
pp = prof_i.replace("_place_invis", "_parked")
cp = capture(pp); imp, depp = images(cp)
res["parked_profile"] = {"profile": pp, "model_xml": cp["model_xml"], "stat_extent": float(cp["m"].stat.extent),
                         "qpos_equal_clear": bool(np.array_equal(cp["d"].qpos, q0)),
                         "policy_images_vs_clear": {k: int((imp[k] != im0[k]).sum()) for k in im0},
                         "policy_images_vs_invisible": {k: int((imp[k] != data[prof_i][0][k]).sum()) for k in im0},
                         "depth_vs_invisible_n_gt_1e-9": {k: int((np.abs(depp[k] - data[prof_i][1][k]) > 1e-9).sum()) for k in depp}}
# invisible: also vs visible
im_v, dep_v = data[prof_v][0], data[prof_v][1]; im_i, dep_i = data[prof_i][0], data[prof_i][1]
res["invisible_vs_visible"] = {"policy_images": {k: int((im_i[k] != im_v[k]).sum()) for k in im_v},
                               "depth": {k: int((np.abs(dep_i[k] - dep_v[k]) > 1e-9).sum()) for k in dep_v}}
# active check on the invisible capture
c = caps[prof_i]; m, d = c["m"], c["d"]; mg = c["obstacle_manager"]; slot = mg.active_slots[0]
cg = sorted(mg._collision_geoms[slot]); vg = sorted(mg._visual_geoms[slot])
ac = {"slot": slot, "collision_geoms": [{"geom": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), "contype": int(m.geom_contype[g]),
                                         "conaffinity": int(m.geom_conaffinity[g]), "group": int(m.geom_group[g]), "rgba": m.geom_rgba[g].tolist()} for g in cg],
      "visual_geoms_rgba": [m.geom_rgba[g].tolist() for g in vg]}
fruit = P.load_randomized_episode(int(ep))["target_fruit"]
fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, fruit)
fg = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == fb}
mid = mg._mocap_ids[slot]; old = d.mocap_pos[mid].copy()
d.mocap_pos[mid] = d.xpos[fb] - np.array([0, 0, 0.05]) if slot != "hurdle_0" else d.mocap_pos[mid]
if slot == "hurdle_0":   # move the whole slot so the bar passes through the fruit
    ob = dict(mg.config["profiles"][prof_i]["obstacles"][0]); h = ob.get("bar_height", 0.26)
    mg._place(slot, [float(d.xpos[fb][0]), float(d.xpos[fb][1]), 0.82], float(ob["yaw"]), sizes={"bar_height": float(d.xpos[fb][2] - 0.82)})
mujoco.mj_forward(m, d)
cons = [(int(d.contact[i].geom1), int(d.contact[i].geom2)) for i in range(d.ncon)]
ac["moved_onto_fruit"] = {"fruit": fruit, "n_contacts_obstacle_fruit": sum(1 for a, b in cons if (a in cg and b in fg) or (b in cg and a in fg)),
                          "object_clearance_m": float(mg.object_clearance())}
res["active_check_invisible"] = ac
pathlib.Path(outp).write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: v for k, v in res.items() if k not in ("definitions",)}, default=float)[:3000])
