"""(offline, 2026-11-09 02:5x dev-4a) held phase: how far is the recorded manipulated geometry (`target_points`) from the real
bowl, and how far are the observed (non-support, self-filtered) bowl points from the hand / finger spheres?
Run: SCENE_DIR=<dir> python held_geom_offset.py '<run glob>' <stem filter>"""
import json, pathlib, sys
import mujoco, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-08_PE_경로끝점_identity")
import pe_signal as PS
Q = PS.Q
HAND = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
run = next(p for p in sorted(Q.E.RUNS.glob(sys.argv[1])) if not p.name.startswith("INVALID"))
eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
ch = PS.episode_chunks(run, order)
for stem in order:
    if sys.argv[2] not in stem: continue
    info, m, rg, cache, stc = PS.scene_for(stem)
    d = mujoco.MjData(m)
    geom = np.load(run / "episodes" / f"{stem}_geom.npz"); grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
    calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
    for ts, cp in ch[stem]:
        c = calls[ts]; st = (c.get("grasp") or {}).get("state")
        if st not in ("closing", "held"): continue
        z = np.load(cp); s = json.loads(str(z["summary_json"]))
        i = grow[int(c["obs_index"])]; d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]; mujoco.mj_forward(m, d)
        S2, G2 = cache.world(d); T = S2[rg[G2] == "target"]
        Tw = geom["T_world_base"][i]; W = lambda X: np.asarray(X, float).reshape(-1, 3) @ Tw[:3, :3].T + Tw[:3, 3]
        tt = cKDTree(T)
        tp = z["target_points"].astype(float)
        off = np.median(tt.query(W(tp))[0]) * 1e3 if len(tp) else np.nan
        pts = z["filtered_cloud"].astype(float)[~z["support_mask"].astype(bool)]
        on_b = tt.query(W(pts))[0] <= 0.01
        names = np.array(s["sphere_link_names"]); hm = np.isin(names, HAND)
        C, Rr = z["sphere_centres"][hm].astype(float), z["sphere_radii"][hm].astype(float)
        bp = pts[on_b]
        dh = (np.linalg.norm(bp[:, None] - C[None], axis=2) - Rr[None]).min(1) * 1e3 if len(bp) else np.array([np.nan])
        dtp = cKDTree(tp).query(bp)[0] * 1e3 if len(tp) and len(bp) else np.array([np.nan])
        man = s.get("manipulated") or {}
        print(f"{ts:3d} {st:8s} manip {man.get('state')} geomsrc {(man.get('anchor') or {}).get('geometry_source')}/{(man.get('anchor') or {}).get('geometry_frame')} "
              f"tp->bowl med {off:5.1f} mm | bowl pts {len(bp):4d}: to hand-sph min {np.nanmin(dh):5.1f} med {np.nanmedian(dh):5.1f} | to tp min {np.nanmin(dtp):5.1f} med {np.nanmedian(dtp):5.1f}")
