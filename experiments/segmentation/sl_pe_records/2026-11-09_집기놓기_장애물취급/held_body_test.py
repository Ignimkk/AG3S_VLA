"""(offline, 2026-11-09 03:1x dev-4a) candidate SL-PE8 'held body': the union of the manipulated geometry over the CLOSING
requests (object still at rest; identity locked from closing) is fixed in the hand frame at the last closing request, then moved
with the hand while held. Score (GT only for scoring): fraction of the observed held-bowl points within `r` of that body, and
observed points of OTHER objects / fixtures within `r` (would be wrongly removed from the obstacle set).
Hand pose: MuJoCo robot0_right_hand (offline stand-in for the server's FK). Run: SCENE_DIR=<dir> python held_body_test.py '<glob>' [stem]"""
import collections, json, pathlib, sys
import mujoco, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-08_PE_경로끝점_identity")
import pe_signal as PS
Q = PS.Q
RAD = (0.015, 0.020, 0.030)
import os
FILL = os.environ.get('FILL', '1') == '1'
tot = collections.defaultdict(lambda: np.zeros(3))
for run in [p for p in sorted(Q.E.RUNS.glob(sys.argv[1])) if not p.name.startswith("INVALID") and (p / "episodes.jsonl").exists()]:
    eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
    order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
    ch = PS.episode_chunks(run, order)
    for stem in order:
        if len(sys.argv) > 2 and sys.argv[2] not in stem: continue
        if not (Q.SCENE / f"{stem}_scene.xml").exists(): continue
        info, m, rg, cache, stc = PS.scene_for(stem); d = mujoco.MjData(m)
        mujoco.mj_forward(m, d); S1, G1 = stc.world(d); R1 = rg[G1]; k1 = R1 != "robot"; st_tree = cKDTree(S1[k1]); R1k = R1[k1]
        hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
        geom = np.load(run / "episodes" / f"{stem}_geom.npz"); grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
        calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
        body, anchor, closing_pts = None, None, []
        n_held = 0; acc = collections.defaultdict(lambda: np.zeros(3))
        prev = None
        for ts, cp in ch[stem]:
            c = calls[ts]; st = (c.get("grasp") or {}).get("state"); z = np.load(cp)
            i = grow[int(c["obs_index"])]; d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]; mujoco.mj_forward(m, d)
            Tw = geom["T_world_base"][i]; Twh = np.eye(4); Twh[:3, :3] = d.xmat[hb].reshape(3, 3); Twh[:3, 3] = d.xpos[hb]
            Tbh = np.linalg.inv(Tw) @ Twh
            if st == "closing":
                if "target_points" in z.files and len(z["target_points"]): closing_pts.append(z["target_points"].astype(float))
                z_last_closing = z
                anchor = Tbh
            elif st != "held":
                closing_pts = [] if st == "searching" else closing_pts
            if st == "held" and prev != "held" and closing_pts and anchor is not None:
                P = np.concatenate(closing_pts); P = np.unique(np.round(P / 0.005), axis=0) * 0.005
                if FILL:   # fill each column down to the highest support plane under the object (it rested there)
                    pl = np.asarray(z_last_closing["support_planes"], float).reshape(-1, 4)
                    zmin = P[:, 2].min()
                    hz = [(-(pl[k, 3]) / pl[k, 2]) * -1 for k in range(len(pl)) if abs(pl[k, 2]) > 0.9]
                    hz = [h for h in (pl[k, 3] / pl[k, 2] for k in range(len(pl)) if abs(pl[k, 2]) > 0.9) if h <= zmin + 0.01]
                    floor = max(hz) if hz else zmin
                    cols = []
                    for x, y, zz in P:
                        for zc in np.arange(floor + 0.004, zz, 0.005):
                            cols.append((x, y, zc))
                    if cols: P = np.unique(np.round(np.r_[P, np.asarray(cols)] / 0.005), axis=0) * 0.005
                body = (np.linalg.inv(anchor) @ np.c_[P, np.ones(len(P))].T).T[:, :3]   # in the hand frame
            if st != "held": body = None if st == "searching" else body
            prev = st
            if st != "held" or body is None: continue
            n_held += 1
            Bn = body @ Tbh[:3, :3].T + Tbh[:3, 3]
            pts = z["filtered_cloud"].astype(float)[~z["support_mask"].astype(bool)]
            S2, G2 = cache.world(d); R2 = rg[G2]; k2 = R2 != "robot"
            tree = Q.SplitTree(st_tree, R1k, S2[k2], R2[k2])
            dd, jj = tree.query(pts @ Tw[:3, :3].T + Tw[:3, 3]); role = tree.roles[jj]; role[dd > 0.01] = "none"
            db = cKDTree(Bn).query(pts)[0]
            for k, r in enumerate(RAD):
                acc["bowl_in"][k] += ((role == "target") & (db <= r)).sum()
                acc["other_in"][k] += ((role != "target") & (role != "none") & (db <= r)).sum()
                acc["none_in"][k] += ((role == "none") & (db <= r)).sum()
            acc["bowl_all"] += (role == "target").sum()
            for o in np.unique(role[(role != "target") & (role != "none") & (db <= 0.02)]):
                acc["other_roles:" + o][1] += ((role == o) & (db <= 0.02)).sum()
        if n_held:
            b = acc["bowl_all"][0]
            print(f"{run.name[16:58]} {stem.split('spatial_')[-1]:7s} held {n_held:3d} | bowl pts {int(b):5d} covered r15/20/30 "
                  f"{' / '.join(f'{x / max(b, 1):.0%}' for x in acc['bowl_in'])} | other-object/fixture pts removed r15/20/30 "
                  f"{' / '.join(str(int(x)) for x in acc['other_in'])} | unowned {' / '.join(str(int(x)) for x in acc['none_in'])} "
                  f"| {[(k.split(':')[1], int(v[1])) for k, v in acc.items() if k.startswith('other_roles')]}", flush=True)
