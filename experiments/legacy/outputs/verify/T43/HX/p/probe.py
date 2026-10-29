from common.workspace import workspace_path as _ag3s_workspace_path
import os, sys, json, pathlib
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import numpy as np, mujoco
import hold_w as HW
A = json.load(open(_ag3s_workspace_path('outputs/verify/T43/V8/analysis/analysis.json')))["runs"]
key = "V8/H/V8_ep1807_s18073"; a = A[key]; rd = pathlib.Path(a["run_dir"])
man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = int(man["scene"]["episode_index"])
z = np.load(rd / "traj.npz", allow_pickle=True); seed = int(z["policy_seed"])
stt = rd.parent / f"{rd.name}.start"
cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
print(prof, cfg)
cl = HW.load_client(); L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; mg = L["obstacle_manager"]
mujoco.mj_forward(m, d)
for g in sorted(mg.collision_geom_ids):
    print(g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), int(m.geom_type[g]), m.geom_size[g], d.geom_xpos[g], d.geom_xmat[g].reshape(3,3)[:,2])
for nm in ["base", "EE_BODY_L", "ee_left", "ee_finger_l1", "ee_finger_l2", "apple"]:
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm); print(nm, b, None if b < 0 else d.xpos[b])
    if b >= 0:
        print("  geoms", [(g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), int(m.geom_type[g]), int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in range(m.ngeom) if m.geom_bodyid[g] == b])
print([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody) if "EE" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "") or "finger" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "") or "ee_" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "")])
rows = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
ctrl = {r["t_step"]: r for r in rows if r["kind"] == "control"}
print(sorted(ctrl[0].keys())); pl = [r for r in rows if r["kind"] == "planning"][0]; print(sorted(pl.keys()))
g_t, p_t = int(a["grasp_t"]), int(a["place_t"]); ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "EE_BODY_L")
for t in range(g_t, p_t + 1, 4):
    d.qpos[:] = ctrl[t]["qpos"]; mujoco.mj_kinematics(m, d); print(t, np.round(d.xpos[ee], 3), ctrl[t].get("chunk_seq"), ctrl[t].get("step_in_chunk"), ctrl[t].get("executed"))
print(z.files)
