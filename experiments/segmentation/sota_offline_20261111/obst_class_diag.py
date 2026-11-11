"""(2026-11-11 dev-57, offline CPU, GT = scoring only) Why was the row next to the real obstacle not an obstacle-object row?
For each collided E1a episode and the calls hit-1, hit: take the true nearest obstacle point p_o / robot point p_r (from
first_contact_<tag>.json, world -> base with the recorded T_world_base) at the frame of the executed window where the robot
is closest, and read from the server record of that call:
  set_mm   distance p_o -> the scene obstacle-object set (replayed _scene_obstacle_points, 3-request union)
  cloud_n  filtered non-support cloud points within 30 mm of p_o in this request (is the obstacle seen now?)
  esdf_o   field distance at p_o (main / target-free tier; ~0 = the surface is in the world model)
  esdf_r   field distance at p_r vs the true distance (world-model error at the robot's nearest point)
  tgt_mm   distance p_o -> nearest target point;  ball = inside the target-free ball
usage: python obst_class_diag.py <tag> [target_cut 0|1]"""
import glob, json, sys, types
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.runtime.pipeline import AG3S  # noqa: E402

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
OUT = "/mnt/dev/AG3S_VLA/outputs/2026-11-11_SOTA_개선/offline/"


def per_episode_chunks(d):
    per, cur, last = [], [], None
    for c in sorted(glob.glob(d + "/server_constraints/run_*/chunk_*.npz")):
        ts = int(np.load(c)["t_step"])
        if last is not None and ts <= last:
            per.append(cur)
            cur = []
        cur.append((ts, c))
        last = ts
    per.append(cur)
    return per


def field(z, prefix, p):
    if prefix + "origin" not in z.files:
        return None
    o, v, D = z[prefix + "origin"], float(z[prefix + "voxel_size"]), z[prefix + "distance"]
    i = np.floor((np.asarray(p) - o) / v).astype(int)
    if (i < 0).any() or (i >= np.array(D.shape)).any():
        return None
    return round(1000 * float(D[i[0], i[1], i[2]]), 1)


def main(tag, cut):
    fc = {(r["task"], r["init"]): r for r in json.load(open(OUT + f"first_contact_{tag}.json"))}
    rows = []
    for t in range(4):
        d = sorted(glob.glob(B + f"2*qe-{tag}-e1a-spatial-LI-t{t}-*"))[-1]
        eps = [json.loads(l) for l in open(d + "/episodes.jsonl")]
        for e, ch in zip(eps, per_episode_chunks(d)):
            k = (t, e["init_state_index"])
            if k not in fc or fc[k]["collide_t"] is None:
                continue
            r = fc[k]
            ct = r["collide_t"]
            cs = [json.loads(x) for x in open(f"{d}/episodes/safelibero_spatial_t{t}_e{e['episode_idx']}_calls.jsonl")]
            hit = max(c["seq"] for c in cs if c["obs_index"] <= ct)
            g = np.load(f"{d}/episodes/safelibero_spatial_t{t}_e{e['episode_idx']}_geom.npz")
            td = {int(a): b for a, b in r["true_robot_dist_mm"].items()}
            byts = dict(ch)
            st = types.SimpleNamespace(_obstacle_history=[])
            for q in range(max(0, hit - 4), hit + 1):
                if q not in byts:
                    continue
                z = np.load(byts[q])
                sm = json.loads(str(z["summary_json"]))
                tgt = z["target_points"].astype(float) if "target_points" in z.files and len(z["target_points"]) else None
                oop, rec = AG3S._scene_obstacle_points(st, z["filtered_cloud"].astype(float), z["support_mask"].astype(bool),
                                                       tgt, None, target_cut=bool(cut))
                if q < hit - 1:
                    continue
                c = next(x for x in cs if x["seq"] == q)
                f0 = c["obs_index"]
                fr = [f for f in range(f0, f0 + 6) if f in td and td[f][2]]
                if not fr:
                    continue
                f = min(fr, key=lambda x: td[x][0])
                i = int(np.argmin(np.abs(g["obs_index"] - f0)))
                Tbw = np.linalg.inv(g["T_world_base"][i])
                pr = (Tbw @ np.r_[td[f][2][:3], 1])[:3]
                po = (Tbw @ np.r_[td[f][2][3:], 1])[:3]
                cloud = z["filtered_cloud"].astype(float)[~z["support_mask"].astype(bool)]
                tf = (sm.get("esdf_stats") or {}).get("target_free") or {}
                if isinstance(tf, str):
                    import ast
                    try:
                        tf = ast.literal_eval(tf)
                    except Exception:
                        tf = {}
                ball = tf.get("ball") or {}
                inball = None
                if ball.get("centre_m") is not None:
                    inball = bool(np.linalg.norm(po - np.asarray(ball["centre_m"])) <= float(ball["radius_mm"]) / 1000)
                rows.append(dict(
                    task=t, init=k[1], seq=q, hit=(q == hit), f=f, true_mm=td[f][0], by=td[f][1].split("/")[1].replace("_collision", ""),
                    grasp=(sm.get("grasp") or {}).get("state"),
                    set_mm=None if oop is None or not len(oop) else round(1000 * float(cKDTree(oop).query(po)[0]), 1),
                    set_n=0 if oop is None else int(len(oop)), n_this=rec.get("n_this_frame"),
                    cloud_n=int((np.linalg.norm(cloud - po, axis=1) <= 0.03).sum()),
                    esdf_o=field(z, "esdf_", po), free_o=field(z, "esdf_free_", po),
                    esdf_r=field(z, "esdf_", pr), free_r=field(z, "esdf_free_", pr),
                    tgt_mm=None if tgt is None else round(1000 * float(cKDTree(tgt).query(po)[0]), 1), ball=inball,
                    dropped=rec.get("n_dropped_total"), ncomp=rec.get("n_components")))
                x = rows[-1]
                print(f"t{t} e{k[1]:2d} seq{q:3d}{'*' if x['hit'] else ' '} f{f:3d} true {x['true_mm']:6.1f} by {x['by'][:16]:16s} {str(x['grasp'])[:8]:8s}"
                      f" set {x['set_mm']} (n {x['set_n']}, this {x['n_this']})  cloud_n {x['cloud_n']}  esdf_o {x['esdf_o']}/{x['free_o']}"
                      f"  esdf_r {x['esdf_r']}/{x['free_r']}  tgt {x['tgt_mm']} ball {x['ball']}  comp {x['ncomp']} drop {x['dropped']}")
    json.dump(rows, open(OUT + f"obst_class_diag_{tag}_cut{int(cut)}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
