"""(2026-11-09 12:1x dev-4a, offline) E1a: does SL-PE11 (--scene-obstacle-target-cut) keep the SafeLIBERO obstacle in the
scene obstacle-object set where the component rule (SL-GTTO20) dropped it? Replays AG3S._scene_obstacle_points on each
recorded pre-held / held call (filtered_cloud, support_mask, target_points of the chunk; single request, no 3-request union)
with target_cut False / True. Scored with the obstacle position (SafeLIBERO record, GT = scoring only): kept voxel centroids
within 80 mm of it. usage: python e1a_pe11_offline.py <tag> [collided|all]"""
import collections
import glob
import json
import sys
import types

import numpy as np

sys.path.insert(0, "/mnt/dev/AG3S_VLA")
sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-09_집기놓기_장애물취급/e1a_decomp")
from AG3S.ag3s.runtime.pipeline import AG3S  # noqa: E402
from e1a_passthrough_window import B, per_episode_chunks  # noqa: E402

R = 0.08


def obstacle_kept(z, pb, cut):
    st = types.SimpleNamespace(_obstacle_history=[])
    tgt = z["target_points"].astype(float) if "target_points" in z.files and len(z["target_points"]) else None
    out, _ = AG3S._scene_obstacle_points(st, z["filtered_cloud"].astype(float), z["support_mask"].astype(bool), tgt, None,
                                         target_cut=cut)
    if out is None or not len(out):
        return 0
    return int((np.linalg.norm(out - pb, axis=1) <= R).sum())


def main(tag, which):
    tot = collections.Counter()
    res = []
    for d in sorted(glob.glob(B + (tag if "*" in tag else f"2*qe-{tag}-e1a-spatial-LI-t*"))):
        if "INVALID" in d:
            continue
        eps = [json.loads(l) for l in open(d + "/episodes.jsonl")]
        for e, ch in zip(eps, per_episode_chunks(d)):
            s = e["safelibero"]
            if which == "collided" and not s["collided"]:
                continue
            t, ep = int(e["task_id"]), int(e["episode_idx"])
            cs = {json.loads(x)["seq"]: json.loads(x) for x in open(f"{d}/episodes/safelibero_spatial_t{t}_e{ep}_calls.jsonl")}
            g = np.load(f"{d}/episodes/safelibero_spatial_t{t}_e{ep}_geom.npz")
            ct = s["collide_t"]
            last = max((q for q, c in cs.items() if ct is not None and c["obs_index"] <= ct), default=max(cs) if cs else 0)
            line, n_drop_only = [], 0
            for ts, cp in ch:
                if ts > last:
                    break
                c = cs.get(ts)
                if c is None:
                    continue
                z = np.load(cp)
                i = int(np.argmin(np.abs(g["obs_index"] - c["obs_index"])))
                pb = (np.linalg.inv(g["T_world_base"][i]) @ np.r_[np.asarray(s["obstacle_initial_pos"], float), 1])[:3]
                a, b = obstacle_kept(z, pb, False), obstacle_kept(z, pb, True)
                tot["calls"] += 1
                tot["off_zero"] += a == 0
                tot["on_zero"] += b == 0
                tot["rescued"] += (a == 0 and b > 0)
                tot["lost"] += (a > 0 and b == 0)
                n_drop_only += (a == 0 and b > 0)
                line.append(f"{ts}:{a}/{b}")
            res.append(dict(task=t, ep=ep, obst=s["obstacle"], collided=s["collided"], last_seq=last, calls=line))
            print(f"t{t}e{ep:2d} {s['obstacle'][:14]:14s} coll={int(s['collided'])} obstacle voxels off/on ->", " ".join(line[-12:]), flush=True)
    print("TOTAL", dict(tot))
    json.dump(res, open(f"/mnt/dev/AG3S_VLA/outputs/2026-11-09_집기놓기_장애물취급/e1a_decomp/pe11_offline_{tag.replace('*','_')}_{which}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "collided")
