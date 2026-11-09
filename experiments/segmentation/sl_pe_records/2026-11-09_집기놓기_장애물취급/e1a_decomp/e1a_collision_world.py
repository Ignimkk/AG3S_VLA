"""(2026-11-09 12:1x dev-4a) E1a collisions: was the SafeLIBERO obstacle in the collision world just before it was hit?
Per collided episode: the last recorded chunk with t_step <= collide_t; ESDF value (esdf_distance = the layer the solve
used; esdf_free = target-free layer) at the obstacle's initial position (SafeLIBERO record, world -> base by T_world_base),
distance of the identity (target_centroid) to the obstacle, and clearance_min_mm of that chunk. GT = obstacle position only
(scoring). usage: python e1a_collision_world.py <tag>"""
import glob
import json
import sys

import numpy as np

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"


def esdf_at(z, key, p):
    if f"{key}_origin" not in z.files:
        return None
    o, v, D = z[f"{key}_origin"], float(z[f"{key}_voxel_size"]), z[f"{key}_distance"]
    ijk = np.floor((p - o) / v).astype(int)
    if np.any(ijk < 0) or np.any(ijk >= np.array(D.shape)):
        return None
    return float(D[tuple(ijk)])


def main(tag):
    rows = []
    for d in sorted(glob.glob(B + f"2*qe-{tag}-e1a-spatial-LI-t*")):
        eps = [json.loads(l) for l in open(d + "/episodes.jsonl")]
        chunks = sorted(glob.glob(d + "/server_constraints/run_0000/chunk_*.npz"))
        # split chunks into episodes by t_step reset
        per, cur, last = [], [], None
        for c in chunks:
            ts = int(np.load(c)["t_step"])
            if last is not None and ts <= last:
                per.append(cur)
                cur = []
            cur.append((ts, c))
            last = ts
        per.append(cur)
        for e, ch in zip(eps, per):
            s = e["safelibero"]
            if not s["collided"]:
                continue
            ct = int(s["collide_t"])
            t, ep = int(e["task_id"]), int(e["episode_idx"])
            g = np.load(f"{d}/episodes/safelibero_spatial_t{t}_e{ep}_geom.npz")
            cs = [json.loads(x) for x in open(f"{d}/episodes/safelibero_spatial_t{t}_e{ep}_calls.jsonl")]
            seq = max((c["seq"] for c in cs if c["obs_index"] <= ct), default=None)
            prev = [x for x in ch if seq is not None and x[0] <= seq]
            if not prev:
                rows.append((t, ep, s["obstacle"], ct, None))
                continue
            ts, cp = prev[-1]
            z = np.load(cp)
            i = int(np.argmin(np.abs(g["obs_index"] - ts)))
            Twb = g["T_world_base"][i]
            pw = np.r_[np.asarray(s["obstacle_initial_pos"], float), 1.0]
            pb = (np.linalg.inv(Twb) @ pw)[:3]
            summ = json.loads(str(z["summary_json"]))
            tc = z["target_centroid"] if "target_centroid" in z.files and len(z["target_centroid"]) else None
            rows.append(dict(task=t, ep=ep, seq=seq, n_ep_chunks=len(ch), obst=s["obstacle"].replace("_obstacle_1", ""), collide_t=ct, chunk_t=ts,
                             disp_mm=round(1000 * s["max_displacement_m"], 1),
                             esdf_used=esdf_at(z, "esdf", pb), esdf_free=esdf_at(z, "esdf_free", pb),
                             id_to_obst_mm=None if tc is None else round(1000 * float(np.linalg.norm(tc - pb)), 0),
                             grasp=(summ.get("grasp") or {}).get("state"), path=summ.get("execution_path"),
                             clearance_min_mm=summ.get("clearance_min_mm"),
                             verdict=(summ.get("verdict") or {}).get("kind") if isinstance(summ.get("verdict"), dict) else summ.get("verdict")))
    for r in rows:
        print(r)
    json.dump(rows, open(f"/mnt/dev/AG3S_VLA/outputs/2026-11-09_집기놓기_장애물취급/e1a_decomp/collision_world_{tag}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1])
