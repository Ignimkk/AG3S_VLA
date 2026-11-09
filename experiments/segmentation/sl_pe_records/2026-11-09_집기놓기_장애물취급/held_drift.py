"""(offline, 2026-11-09 03:0x dev-4a) held phase: rigid hand-following snapshot vs the real bowl (GT, scoring only), and the
observed bowl points' distance to the hand / finger constraint spheres. Run: SCENE_DIR=<dir> python held_drift.py '<glob>' <stem>"""
import json, pathlib, sys
import mujoco, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-08_PE_경로끝점_identity")
import pe_signal as PS
Q = PS.Q
HAND = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
for run in [p for p in sorted(Q.E.RUNS.glob(sys.argv[1])) if not p.name.startswith("INVALID") and (p / "episodes.jsonl").exists()]:
    eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
    order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
    ch = PS.episode_chunks(run, order)
    for stem in order:
        if sys.argv[2] not in stem: continue
        print("==", run.name[16:60], stem)
        info, m, rg, cache, stc = PS.scene_for(stem); d = mujoco.MjData(m)
        hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
        geom = np.load(run / "episodes" / f"{stem}_geom.npz"); grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
        calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
        snap = None
        for ts, cp in ch[stem]:
            c = calls[ts]; st = (c.get("grasp") or {}).get("state"); z = np.load(cp)
            i = grow[int(c["obs_index"])]; d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]; mujoco.mj_forward(m, d)
            Tw = geom["T_world_base"][i]; Twh = np.eye(4); Twh[:3, :3] = d.xmat[hb].reshape(3, 3); Twh[:3, 3] = d.xpos[hb]
            Tbh = np.linalg.inv(Tw) @ Twh
            S2, G2 = cache.world(d); T = S2[rg[G2] == "target"]; Tb = (T - Tw[:3, 3]) @ Tw[:3, :3]   # GT bowl in base
            tt = cKDTree(Tb)
            if st != "held":
                if "target_points" in z.files and len(z["target_points"]): snap = (z["target_points"].astype(float), Tbh)
                continue
            if snap is None: continue
            Tn = Tbh @ np.linalg.inv(snap[1]); hn = snap[0] @ Tn[:3, :3].T + Tn[:3, 3]
            s = json.loads(str(z["summary_json"])); names = np.array(s["sphere_link_names"]); hm = np.isin(names, HAND)
            C, Rr = z["sphere_centres"][hm].astype(float), z["sphere_radii"][hm].astype(float)
            pts = z["filtered_cloud"].astype(float)[~z["support_mask"].astype(bool)]
            bp = pts[tt.query(pts)[0] <= 0.01]
            dh = (np.linalg.norm(bp[:, None] - C[None], axis=2) - Rr[None]).min(1) * 1e3 if len(bp) else np.array([np.nan])
            dsn = cKDTree(hn).query(bp)[0] * 1e3 if len(bp) else np.array([np.nan])
            bowl_c = Tb.mean(0); hand_rel = np.linalg.inv(Tbh) @ np.r_[bowl_c, 1]
            print(f"  {ts:3d} snap->GT med {np.median(tt.query(hn)[0])*1e3:5.1f} mm | obs bowl pts {len(bp):4d} to snap min {np.nanmin(dsn):5.1f} med {np.nanmedian(dsn):5.1f} | to hand-sph min {np.nanmin(dh):5.1f} med {np.nanmedian(dh):5.1f} | bowl in hand frame {np.round(hand_rel[:3]*1e3).astype(int)}")
