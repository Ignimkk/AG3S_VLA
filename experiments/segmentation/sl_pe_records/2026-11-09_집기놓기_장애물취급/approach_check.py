"""(offline, 2026-11-09 03:3x dev-4a) approach-phase refusals: what was the identity (recorded manipulated geometry
`target_points`, GT role majority), and why did the target's own points stay in the obstacle set (their distance to the
identity points, their component vs the identity's)? Run: SCENE_DIR=<dir> python approach_check.py '<glob>' <stem>"""
import collections, json, pathlib, sys
import mujoco, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-08_PE_경로끝점_identity")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pe_signal as PS
from held_obst_replay import components, V
Q = PS.Q
run = next(p for p in sorted(Q.E.RUNS.glob(sys.argv[1])) if not p.name.startswith("INVALID") and (p / "episodes.jsonl").exists())
eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
ch = PS.episode_chunks(run, order)
for stem in order:
    if sys.argv[2] not in stem: continue
    info, m, rg, cache, stc = PS.scene_for(stem); d = mujoco.MjData(m)
    geom = np.load(run / "episodes" / f"{stem}_geom.npz"); grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
    calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
    for ts, cp in ch[stem]:
        c = calls[ts]; st = (c.get("grasp") or {}).get("state")
        pc = (c.get("to") or {}).get("passthrough_clear") or {}
        if st not in ("searching", "latched", "closing") or pc.get("reason") != "obstacle": continue
        z = np.load(cp); i = grow[int(c["obs_index"])]; d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]; mujoco.mj_forward(m, d)
        Tw = geom["T_world_base"][i]
        S1, G1 = stc.world(d); S2, G2 = cache.world(d); S = np.r_[S1, S2]; R = np.r_[rg[G1], rg[G2]]
        k = R != "robot"; tree = cKDTree(S[k]); Rk = R[k]
        role = lambda X: np.where(tree.query(X @ Tw[:3, :3].T + Tw[:3, 3])[0] <= 0.01, Rk[tree.query(X @ Tw[:3, :3].T + Tw[:3, 3])[1]], "none")
        tp = z["target_points"].astype(float) if "target_points" in z.files else np.zeros((0, 3))
        idr = collections.Counter(role(tp).tolist()).most_common(2) if len(tp) else None
        pts = z["filtered_cloud"].astype(float)[~z["support_mask"].astype(bool)]
        comp = components(pts); r = role(pts)
        tgt = r == "target"
        dist = cKDTree(tp).query(pts[tgt])[0] * 1e3 if len(tp) and tgt.any() else np.array([np.nan])
        tcomp = set(np.unique(comp[cKDTree(pts).query_ball_point(tp, V, return_length=False)[0]] if False else [])) if False else None
        # components of target points: which contain identity-near points (dropped) vs not
        near = cKDTree(tp).query(pts)[0] <= V if len(tp) else np.zeros(len(pts), bool)
        dropped = set(np.unique(comp[near]).tolist())
        tc = collections.Counter(comp[tgt].tolist())
        kept = {cc: n for cc, n in tc.items() if cc not in dropped}
        kept_info = []
        for cc, n in sorted(kept.items(), key=lambda x: -x[1])[:3]:
            cm = comp == cc
            kept_info.append((n, int(cm.sum()), collections.Counter(r[cm].tolist()).most_common(2), np.round(np.ptp(pts[cm], axis=0) * 1e3).astype(int).tolist()))
        man = s = json.loads(str(z["summary_json"])).get("manipulated") or {}
        print(f"{ts:3d} {st:9s} id {idr} ({len(tp)} pts, src {(man.get('anchor') or {}).get('geometry_source')}) | target obs pts {int(tgt.sum())}, "
              f"to identity med {np.nanmedian(dist):5.1f} | target pts left in set {sum(kept.values())}: {kept_info}")
