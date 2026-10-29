"""T42-2 smoke (implementer): run pi05_infer.main() for an episode with an obstacle profile up to the point where the
policy would be contacted (no server, no GPU), then render a third-person osmesa image of the scene main() built.
For every profile the reset checks of main() ran (robot clearance > 0.02, object clearance > 0); this script also
recomputes both clearances with the independent numpy GJK of tests/sim/_gjk.py.
usage: t42_2_smoke.py CONFIG OUT_DIR EPISODE:PROFILE[:VIEW] ..."""
from common.workspace import workspace_path as _ag3s_workspace_path
import inspect, json, os, pathlib, sys
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, _ag3s_workspace_path('integration/rby1/bringup')); sys.path.insert(0, _ag3s_workspace_path(''))
import mujoco, numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import pi05_infer as P  # noqa: E402
from tests.sim._gjk import gjk_dist, make_support  # noqa: E402

cfg, out = sys.argv[1], pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)


class Reached(Exception):
    pass


def run(episode, profile):
    cap = {}

    def stub(remote):
        cap.update(inspect.currentframe().f_back.f_locals); raise Reached

    P.load_remote_policy = stub
    sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1",
                "--episode-index", str(episode), "--headless", "--obstacle-config", cfg, "--obstacle-profile", profile]
    try:
        P.main()
    except Reached:
        pass
    except SystemExit as exc:  # main()'s reset clearance checks reject the placement
        cap["_exit"] = str(exc)
    return cap


def gjk_min(m, d, obstacles, others, cap=0.25):
    best = cap
    for h in obstacles:
        for o in others:
            if np.linalg.norm(d.geom_xpos[h] - d.geom_xpos[o]) - m.geom_rbound[h] - m.geom_rbound[o] > cap:
                continue
            best = min(best, gjk_dist(make_support(m, d, h), make_support(m, d, o), d.geom_xpos[o] - d.geom_xpos[h] + 1e-9))
    return best


results = {}
for spec in sys.argv[3:]:
    episode, profile, *view = spec.split(":")
    cap = run(int(episode), profile)
    if "_exit" in cap:
        print(json.dumps({"profile": profile, "rejected_by_main": cap["_exit"]}), flush=True)
        results[profile] = {"episode": int(episode), "profile": profile, "rejected_by_main": cap["_exit"]}
        continue
    m, d, mgr = cap["m"], cap["d"], cap["obstacle_manager"]
    ep = cap["randomized_episode"]
    obstacles = sorted(mgr.collision_geom_ids)
    target = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ep["target_fruit"])].copy()
    crate = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")].copy()
    summary = {
        "episode": int(episode), "profile": profile, "model_xml": cap["model_xml"],
        "target_fruit": ep["target_fruit"], "used_arm": ep["used_arm"],
        "active_slots": list(mgr.active_slots),
        "entries": mgr.config["profiles"][profile]["obstacles"],
        "robot_clearance_m": mgr.robot_clearance(), "object_clearance_m": mgr.object_clearance(),
        "robot_clearance_gjk_m": gjk_min(m, d, obstacles, mgr.robot_geom_ids),
        "object_clearance_gjk_m": gjk_min(m, d, obstacles, mgr.payload_geom_ids),
        "target_world_m": target.tolist(), "crate_world_m": crate.tolist(),
        "obstacle_geoms": {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g): {
            "xpos": d.geom_xpos[g].round(5).tolist(), "size": m.geom_size[g][:2].round(5).tolist()} for g in obstacles},
    }
    first = np.mean([d.geom_xpos[g] for g in obstacles], axis=0)
    lookat = 0.5 * (first + (crate if "crate" in profile else target))
    azimuth = float(view[0]) if view else (150.0 if ep["used_arm"] == "left" else 210.0)
    r = mujoco.Renderer(m, height=600, width=800)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [lookat[0], lookat[1], 0.92]; cam.distance = 0.85; cam.azimuth = azimuth; cam.elevation = -30.0
    r.update_scene(d, camera=cam)
    name = f"t42_2_smoke_ep{episode}_{profile.split('_', 1)[1]}.png"
    Image.fromarray(r.render()).save(out / name)
    r.close()
    summary["image"] = str(out / name)
    results[profile] = summary
    print(json.dumps({k: summary[k] for k in ("profile", "active_slots", "robot_clearance_m", "robot_clearance_gjk_m",
                                               "object_clearance_m", "object_clearance_gjk_m", "image")}), flush=True)
(out / "t42_2_smoke_summary.json").write_text(json.dumps(results, indent=1))
