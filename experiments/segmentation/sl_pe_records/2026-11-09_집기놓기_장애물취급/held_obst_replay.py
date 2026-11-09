"""(.venv-ag3s, CPU, offline) 2026-11-09 02:4x (dev-4a). Why does the held bowl stay an obstacle object, and would a
'component touching the fingers / hand while attached = the held object' rule remove it?

For every call whose pass-through was refused by an obstacle-object row (`to.passthrough_clear.reason == 'obstacle'`):
  1. rebuild the scene obstacle-object set as `pipeline._scene_obstacle_points` does (15 mm components of the non-support
     cloud; drop components within 15 mm of the recorded manipulated geometry `target_points`, the 0-30 mm band under its base;
     union of the last 3 requests). Destination points are not recorded -> not dropped here (noted).
  2. nearest obstacle point to the worst row's sphere centre: its age (0 = this request, 1-2 = earlier requests in the union),
     its GT owner (target / destination / fixture / ...), and the row clearance re-measured on points (dist - sphere radius).
  3. candidate rule HF (held fingers): while the latch is `held` (or closing), also drop components having a point within
     `pad` of a hand / finger constraint sphere surface (robot model at that request, recorded `sphere_centres`).
     Components larger than `max_extent` (fixtures) are cut to the points within `crop` of the hand spheres instead.
Run: SCENE_DIR=<scene dir> python held_obst_replay.py '<run glob>' <stem filter> <tag>
"""
import collections
import json
import pathlib
import sys

import mujoco
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-08_PE_경로끝점_identity")
import pe_signal as PS  # noqa: E402

Q = PS.Q
OUT = pathlib.Path(__file__).resolve().parent
V = 0.015
HAND = ("panda_hand", "panda_leftfinger", "panda_rightfinger")


def components(pts):
    lo = pts.min(axis=0)
    ijk = np.floor((pts - lo) / V).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    occ = np.zeros(tuple(int(x) for x in dims), bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3), bool))
    return lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def frame_obstacles(z, hand_rule, pad=0.015, max_extent=0.20, crop=0.06, held=None, held_r=V):
    pts = z["filtered_cloud"].astype(float)
    sm = z["support_mask"].astype(bool)
    pts = pts[~sm]
    if not len(pts):
        return np.zeros((0, 3))
    comp = components(pts)
    tree = cKDTree(pts)
    drop = set()
    tgt = z["target_points"].astype(float) if "target_points" in z.files else np.zeros((0, 3))
    if len(tgt):
        for ids in tree.query_ball_point(tgt, V):
            drop.update(int(comp[i]) for i in ids)
        zmin = float(tgt[:, 2].min())
        under = (pts[:, 2] <= zmin + 0.005) & (pts[:, 2] >= zmin - 0.03)
        base = tgt[tgt[:, 2] <= zmin + 0.01]
        if under.any():
            near = cKDTree(base[:, :2]).query(pts[under, :2])[0] <= 0.01
            drop.update(int(c) for c in np.unique(comp[under][near]))
    if held is not None and len(held):   # SL-PE3 replica: components within held_r of the held snapshot points now
        for ids in tree.query_ball_point(held, held_r):
            drop.update(int(comp[i]) for i in ids)
    keep = ~np.isin(comp, sorted(drop)) & (comp > 0)
    if hand_rule is not None:
        C, Rr = hand_rule
        d_hand = (np.linalg.norm(pts[:, None, :] - C[None], axis=2) - Rr[None]).min(axis=1)
        touch = np.unique(comp[d_hand <= pad])
        for c in touch:
            m = comp == c
            ext = np.ptp(pts[m], axis=0).max()
            if ext <= max_extent:
                keep &= ~m
            else:
                keep &= ~(m & (d_hand <= crop))
    return pts[keep]


def main():
    runs = [p for p in sorted(Q.E.RUNS.glob(sys.argv[1])) if not p.name.startswith("INVALID")
            and (p / "episodes.jsonl").is_file()]
    filt, tag = sys.argv[2], sys.argv[3]
    res = {}
    for run in runs:
        eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
        order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
        succ = {o: bool(e["official_success"]) for o, e in zip(order, eps)}
        ch = PS.episode_chunks(run, order)
        for stem in order:
            if filt not in stem or not (Q.SCENE / f"{stem}_scene.xml").exists():
                continue
            info, m, rg, cache, stc = PS.scene_for(stem)
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            S1, G1 = stc.world(d)
            R1 = rg[G1]
            k1 = R1 != "robot"
            st_tree, R1k = cKDTree(S1[k1]), R1[k1]
            geom = np.load(run / "episodes" / f"{stem}_geom.npz")
            grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
            calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
            hist = {"base": [], "hf": [], "pe3": [], "pe3w": []}
            hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
            snap = None   # (target_points base, T_base_hand) of the last pre-held request with a manipulated geometry
            def T_base_hand(i):
                d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]
                mujoco.mj_forward(m, d)
                Twh = np.eye(4); Twh[:3, :3] = d.xmat[hb].reshape(3, 3); Twh[:3, 3] = d.xpos[hb]
                return np.linalg.inv(geom["T_world_base"][i]) @ Twh
            rows = []
            for ts, cp in ch.get(stem, []):
                z = np.load(cp)
                c = calls.get(ts)
                if c is None:
                    continue
                st = (c.get("grasp") or {}).get("state")
                s = json.loads(str(z["summary_json"]))
                names = np.array(s.get("sphere_link_names") or [])
                hm = np.isin(names, HAND)
                hand_rule = (z["sphere_centres"][hm].astype(float), z["sphere_radii"][hm].astype(float)) \
                    if st in ("held", "closing") and hm.any() else None
                ii = grow.get(int(c["obs_index"]))
                held_now = None
                if st != "held" and "target_points" in z.files and len(z["target_points"]):
                    snap = (z["target_points"].astype(float), T_base_hand(ii))
                elif st == "held" and snap is not None:
                    Tn = T_base_hand(ii) @ np.linalg.inv(snap[1])
                    held_now = snap[0] @ Tn[:3, :3].T + Tn[:3, 3]
                fr = {"base": frame_obstacles(z, None), "hf": frame_obstacles(z, hand_rule),
                      "pe3": frame_obstacles(z, None, held=held_now), "pe3w": frame_obstacles(z, None, held=held_now, held_r=0.04)}
                for k_ in hist:
                    hist[k_] = (hist[k_] + [fr[k_]])[-3:]
                pc = (c.get("to") or {}).get("passthrough_clear") or {}
                w = ((pc.get("obstacle_rows") or {}).get("worst") or {})
                if pc.get("reason") != "obstacle" or w.get("point_m") is None:
                    continue
                p = np.asarray(w["point_m"], float)
                q = w.get("query")
                rad = float(z["sphere_radii"][q]) if q is not None and q < len(z["sphere_radii"]) else 0.0
                i = grow.get(int(c["obs_index"]))
                d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]
                mujoco.mj_forward(m, d)
                S2, G2 = cache.world(d)
                R2 = rg[G2]
                k2 = R2 != "robot"
                tree = Q.SplitTree(st_tree, R1k, S2[k2], R2[k2])
                Tw = geom["T_world_base"][i]
                W = lambda X: np.asarray(X, float).reshape(-1, 3) @ Tw[:3, :3].T + Tw[:3, 3]
                row = {"seq": ts, "state": st, "link": w.get("link"), "rec_clear_mm": round(w["clearance_m"] * 1e3, 1)}
                for key in hist:
                    allp = [(age, P) for age, P in enumerate(reversed(hist[key])) if len(P)]
                    if not allp:
                        row[key] = None
                        continue
                    best = min(((float(np.linalg.norm(P - p, axis=1).min()), age, P[np.argmin(np.linalg.norm(P - p, axis=1))])
                                for age, P in allp), key=lambda x: x[0])
                    dd, jj = tree.query(W(best[2]))
                    row[key] = {"clear_mm": round((best[0] - rad) * 1e3, 1), "age": best[1],
                                "owner": str(tree.roles[jj[0]]), "owner_mm": round(float(dd[0]) * 1e3, 1)}
                rows.append(row)
            res[f"{run.name[16:60]}/{stem}"] = {"success": succ[stem], "rows": rows}
            print("==", run.name[16:60], stem, "ok" if succ[stem] else "FAIL", flush=True)
            for r in rows:
                b, h = r["base"], r["hf"]
                f = lambda x: "-" if x is None else f"{x['clear_mm']:6.1f} age{x['age']} {x['owner']}"
                print(f"  {r['seq']:3d} {r['state']:9s} {r['link'][:22]:22s} rec {r['rec_clear_mm']:6.1f} | base {f(b)} | HF {f(h)} | PE3 {f(r['pe3'])} | PE3w40 {f(r['pe3w'])}")
    (OUT / f"held_replay_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
