"""(offline, 2026-11-09 03:2x dev-4a) candidate SL-PE8 'held object = what moves with the hand'.
While held: A = accumulated held-object points in the HAND frame (starts from the pre-held manipulated snapshot).
Per request k: non-support cloud P_k -> hand frame H_k = T_k^-1 P_k. A point is 'moving with the hand' when it lies within
`match` of A or of the previous request's hand-frame points H_{k-1}, AND a static point there would have drifted >= `drift` in the
hand frame between k-1 and k (|T_k^-1 p - T_{k-1}^-1 p|; without that motion the test cannot tell). Matched points join A.
Removed from the obstacle-object set: points within `match` of A (hand frame). Score with GT (scoring only): covered fraction of
observed held-bowl points; points of other objects / fixtures removed. Run: SCENE_DIR=<dir> python held_motion_test.py '<glob>' [stem]"""
import collections, json, pathlib, sys
import mujoco, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-08_PE_경로끝점_identity")
import pe_signal as PS
Q = PS.Q
MATCH, DRIFT, BOUND = 0.010, 0.025, 0.15
KEEP_ROBOT = True   # score robot pixels the self-filter left as "robot"
REACH_PAD = float(__import__("os").environ.get("REACH_PAD", "0.04"))
res = {}
for run in [p for p in sorted(Q.E.RUNS.glob(sys.argv[1])) if not p.name.startswith("INVALID") and (p / "episodes.jsonl").exists()]:
    eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
    order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
    ch = PS.episode_chunks(run, order)
    for stem in order:
        if len(sys.argv) > 2 and sys.argv[2] not in stem: continue
        if not (Q.SCENE / f"{stem}_scene.xml").exists(): continue
        info, m, rg, cache, stc = PS.scene_for(stem); d = mujoco.MjData(m)
        mujoco.mj_forward(m, d); S1, G1 = stc.world(d); R1 = rg[G1]; k1 = np.ones(len(R1), bool) if KEEP_ROBOT else R1 != "robot"; st_tree = cKDTree(S1[k1]); R1k = R1[k1]
        hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
        geom = np.load(run / "episodes" / f"{stem}_geom.npz"); grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
        calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
        reach = 0.15; AV = np.zeros((0, 3)); A, snap, prevH, prevT, prev, prevP, c0 = None, None, None, None, None, np.zeros((0, 3)), np.zeros(3)
        per = []
        acc = collections.Counter()
        for ts, cp in ch[stem]:
            c = calls[ts]; st = (c.get("grasp") or {}).get("state"); z = np.load(cp)
            i = grow[int(c["obs_index"])]; d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]; mujoco.mj_forward(m, d)
            Tw = geom["T_world_base"][i]; Twh = np.eye(4); Twh[:3, :3] = d.xmat[hb].reshape(3, 3); Twh[:3, 3] = d.xpos[hb]
            T = np.linalg.inv(Tw) @ Twh; Ti = np.linalg.inv(T)
            pts = z["filtered_cloud"].astype(float)[~z["support_mask"].astype(bool)]
            tip = (((c.get("grasp") or {}).get("tip_alignment") or {}).get("now") or {}).get("tip_m")
            tip = None if tip is None else np.asarray(tip, float)
            H = pts @ Ti[:3, :3].T + Ti[:3, 3]
            if st != "held":
                if "target_points" in z.files and len(z["target_points"]):
                    tp = z["target_points"].astype(float)
                    reach = min(float(np.ptp(tp, axis=0).max()) + REACH_PAD, 0.15)
                    if tip is not None:
                        tp = tp[np.linalg.norm(tp - tip, axis=1) <= reach]   # only what can be in the hand
                    snap = tp @ Ti[:3, :3].T + Ti[:3, 3]
                A, prevH, prevT, prev, prevP = None, H, T, st, pts   # the last pre-held request is the first reference
                continue
            if prev != "held":
                prevH = None if prev != "closing" else prevH
                A = snap.copy() if snap is not None else np.zeros((0, 3))
                AV = np.zeros((0, 3))   # verified: moved with the hand (the only points ever removed)
                c0 = A.mean(axis=0) if len(A) else np.zeros(3)
            ref = A if (prevH is None or prev != "held") else np.r_[A, prevH]   # pre-held points are not the held object
            if prevT is not None and len(pts) and len(ref):
                drift = np.linalg.norm((pts @ Ti[:3, :3].T + Ti[:3, 3]) - (pts @ np.linalg.inv(prevT)[:3, :3].T + np.linalg.inv(prevT)[:3, 3]), axis=1)
                near = cKDTree(ref).query(H)[0] <= MATCH
                static = cKDTree(prevP).query(pts)[0] <= MATCH if len(prevP) else np.zeros(len(pts), bool)
                inside = (np.linalg.norm(pts - tip, axis=1) <= reach) if tip is not None else np.zeros(len(pts), bool)
                new = near & (drift >= DRIFT) & ~static & inside
                if new.any():
                    A = np.unique(np.round(np.r_[A, H[new]] / 0.004), axis=0) * 0.004
                    AV = np.unique(np.round(np.r_[AV, H[new]] / 0.004), axis=0) * 0.004
            else:
                drift = np.zeros(len(pts)); static = np.zeros(len(pts), bool)
            removed = cKDTree(AV).query(H)[0] <= MATCH if len(AV) and len(H) else np.zeros(len(H), bool)
            if prevT is not None and len(prevP):
                removed &= ~(static & (drift >= DRIFT))
            removed &= (np.linalg.norm(pts - tip, axis=1) <= reach) if tip is not None else False   # explained by 'at rest' while the hand moved: never removed
            S2, G2 = cache.world(d); R2 = rg[G2]; k2 = np.ones(len(R2), bool) if KEEP_ROBOT else R2 != "robot"
            tree = Q.SplitTree(st_tree, R1k, S2[k2], R2[k2])
            dd, jj = tree.query(pts @ Tw[:3, :3].T + Tw[:3, 3]); role = tree.roles[jj].astype(object); role[dd > 0.01] = "none"
            nb = int((role == "target").sum()); cov = int(((role == "target") & removed).sum())
            gb = np.array([m.body(b).name for b in m.geom_bodyid])
            btree = Q.SplitTree(st_tree, gb[G1][k1], S2[k2], gb[G2][k2])
            body = btree.roles[jj]
            oth = collections.Counter([f"{r}:{b}" for r, b in zip(role[removed & (role != "target")], body[removed & (role != "target")])])
            acc["bowl"] += nb; acc["cov"] += cov
            for k, v in oth.items(): acc["other:" + k] += v
            per.append((ts, nb, cov, dict(oth), len(A)))
            prevH, prevT, prev, prevP = H, T, st, pts
        if per:
            print(f"{run.name[16:58]} {stem.split('spatial_')[-1]:7s} held {len(per):3d} | bowl pts {acc['bowl']:5d} covered {acc['cov'] / max(acc['bowl'], 1):.0%} | "
                  f"removed other {[(k[6:], v) for k, v in acc.items() if k.startswith('other:')]}", flush=True)
            res[f"{run.name[16:58]}/{stem}"] = per
            for p in per: print("    ", p)
json.dump(res, open(pathlib.Path(__file__).parent / f"held_motion_{pathlib.Path(sys.argv[1]).name.strip('*2')[:30]}.json", "w"), default=str)
