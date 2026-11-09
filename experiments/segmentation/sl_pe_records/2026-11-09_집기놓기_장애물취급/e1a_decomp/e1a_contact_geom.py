"""(2026-11-09 12:1x dev-4a) E1a collisions, the call that executed into the hit (seq): nearest robot collision sphere
(current config) to the SafeLIBERO obstacle position (GT, scoring), its link, the ESDF value the solve used at the obstacle
position and around it (min over a 6 cm cube = is the obstacle occupied in the world), the planned min clearance over the
executed rows, and the identity centroid distance to that sphere. usage: python e1a_contact_geom.py <tag>"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-09_집기놓기_장애물취급/e1a_decomp")
from e1a_passthrough_window import B, per_episode_chunks  # noqa: E402


def esdf_min_cube(z, p, half=0.03):
    o, v, D = z["esdf_origin"], float(z["esdf_voxel_size"]), z["esdf_distance"]
    lo = np.floor((p - half - o) / v).astype(int)
    hi = np.floor((p + half - o) / v).astype(int) + 1
    lo = np.clip(lo, 0, np.array(D.shape))
    hi = np.clip(hi, 0, np.array(D.shape))
    sub = D[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return None if sub.size == 0 else round(1000 * float(sub.min()), 1)


def main(tag):
    rows = []
    for d in sorted(glob.glob(B + f"2*qe-{tag}-e1a-spatial-LI-t*")):
        eps = [json.loads(l) for l in open(d + "/episodes.jsonl")]
        for e, ch in zip(eps, per_episode_chunks(d)):
            s = e["safelibero"]
            if not s["collided"]:
                continue
            t, ep, ct = int(e["task_id"]), int(e["episode_idx"]), int(s["collide_t"])
            cs = [json.loads(x) for x in open(f"{d}/episodes/safelibero_spatial_t{t}_e{ep}_calls.jsonl")]
            seq = max((c["seq"] for c in cs if c["obs_index"] <= ct), default=0)
            c = next(x for x in cs if x["seq"] == seq)
            byts = dict(ch)
            if seq not in byts:
                continue
            z = np.load(byts[seq])
            sm = json.loads(str(z["summary_json"]))
            g = np.load(f"{d}/episodes/safelibero_spatial_t{t}_e{ep}_geom.npz")
            i = int(np.argmin(np.abs(g["obs_index"] - c["obs_index"])))
            pb = (np.linalg.inv(g["T_world_base"][i]) @ np.r_[np.asarray(s["obstacle_initial_pos"], float), 1])[:3]
            sc, sr = z["sphere_centres"], z["sphere_radii"]
            names = sm.get("sphere_link_names") or []
            dd = np.linalg.norm(sc - pb, axis=1) - sr
            k = int(np.argmin(dd))
            clr = z["clearance"]
            ex = int((sm.get("execution_path") or {}).get("execution_length") or 5)
            tc = z["target_centroid"] if "target_centroid" in z.files and len(z["target_centroid"]) else None
            r = dict(task=t, ep=ep, obst=s["obstacle"].replace("_obstacle_1", ""), seq=seq, steps_into=ct - c["obs_index"],
                     grasp=(c.get("grasp") or {}).get("state"),
                     sphere_to_obst_mm=round(1000 * float(dd[k])), link=names[k] if k < len(names) else k,
                     esdf_min_near_obst_mm=esdf_min_cube(z, pb), exec_min_clearance_mm=round(1000 * float(clr[:ex].min()), 1),
                     exec_min_clearance_link=names[int(np.unravel_index(np.argmin(clr[:ex]), clr[:ex].shape)[1])] if names else None,
                     id_to_obst_mm=None if tc is None else round(1000 * float(np.linalg.norm(tc - pb))),
                     disp_mm=round(1000 * s["max_displacement_m"], 1))
            rows.append(r)
            print(r)
    json.dump(rows, open(f"/mnt/dev/AG3S_VLA/outputs/2026-11-09_집기놓기_장애물취급/e1a_decomp/contact_geom_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
