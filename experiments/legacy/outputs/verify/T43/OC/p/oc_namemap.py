"""T43 OC — MJCF finger body <-> URDF constraint link name check (verifier, CPU, recorded data).
usage: T43W_SNAP=<snap> PYTHONPATH=<snap> oc_namemap.py OUT_JSON KEY [KEY ...]
For each run, chunks 1..N: record sphere_centres (refined step 0, base frame) of URDF links ee_finger_l1 / ee_finger_l2 /
(r1, r2) -> world with the planning-row qpos base pose; distance from each URDF link's sphere-centre mean to each MJCF finger
body's collision-geom centre mean (mm) at the same state (planning-row qpos with the arm set to... the recorded qpos itself).
Note: refined step 0 differs from q_now by <= max_step, so distances carry that offset; the comparison is between the two pairings."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p'))
import numpy as np, mujoco
import hold_w as HW, dx_common as C
C.ANALYSES["V8"] = C.T43 / "V8/analysis/analysis.json"; C.ARM_SRC["V8"] = "V8"
out = pathlib.Path(sys.argv[1]); cl = HW.load_client(); res = {"definitions": __doc__, "runs": {}}
for key in sys.argv[2:]:
    a = C.run_entry(key); rd = pathlib.Path(a["run_dir"]); ctrl, plan = C.frames(rd); chunks = C.segment_chunks(a["server_segment"])
    man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = man["scene"]["episode_index"]
    seed = int(np.load(rd / "traj.npz")["policy_seed"]); stt = rd.parent / f"{rd.name}.start"
    cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
    L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]
    side = "r" if int(ep) == 1834 else "l"
    bodies = [f"ee_finger_{side}1", f"ee_finger_{side}2"]
    gid = {b: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) and (m.geom_contype[g] or m.geom_conaffinity[g])] for b in bodies}
    rows = []
    for sq in sorted(chunks):
        if sq not in plan: continue
        with np.load(chunks[sq], allow_pickle=True) as z:
            s = json.loads(str(z["summary_json"])); names = C.lit(s.get("sphere_link_names")) or []
            if "sphere_centres" not in z.files or not names: continue
            Cc = np.asarray(z["sphere_centres"], float)
        d.qpos[:] = np.asarray(plan[sq]["qpos"], float); mujoco.mj_forward(m, d); Tb = C.T_world_base(m, d)
        Cw = Cc @ Tb[:3, :3].T + Tb[:3, 3]
        r = {"seq": sq}
        for ln in bodies:
            msk = np.array([n == ln for n in names])
            if not msk.any(): continue
            mu = Cw[msk].mean(0)
            for b in bodies:
                r[f"urdf:{ln}->mjcf:{b}_mm"] = 1e3 * float(np.linalg.norm(mu - d.geom_xpos[gid[b]].mean(0)))
        rows.append(r)
    agg = {k: float(np.median([r[k] for r in rows if k in r])) for k in rows[0] if k != "seq"}
    res["runs"][key] = {"n_chunks": len(rows), "median_mm": agg, "rows": rows}
    print(key, agg, flush=True)
out.write_text(json.dumps(res, indent=1))
