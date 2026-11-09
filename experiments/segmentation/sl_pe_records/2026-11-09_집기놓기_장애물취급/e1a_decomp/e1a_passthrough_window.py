"""(2026-11-09 12:1x dev-4a) E1a collisions: what did the server do in the calls just before the hit?
Per collided episode, calls seq-3 .. seq (seq = last call before collide_t): execution source (reference = pass-through,
else QP / SQP / HOLD), pass-through reason, obstacle-row worst violation vs tau_obs, n obstacle rows, worst point distance to
the SafeLIBERO obstacle (initial pos, GT = scoring only). usage: python e1a_passthrough_window.py <tag>"""
import ast
import collections
import glob
import json
import sys

import numpy as np

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"


def per_episode_chunks(d):
    per, cur, last = [], [], None
    for c in sorted(glob.glob(d + "/server_constraints/run_0000/chunk_*.npz")):
        ts = int(np.load(c)["t_step"])
        if last is not None and ts <= last:
            per.append(cur)
            cur = []
        cur.append((ts, c))
        last = ts
    per.append(cur)
    return per


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}
    return x or {}


def main(tag):
    tot = collections.Counter()
    out = []
    for d in sorted(glob.glob(B + f"2*qe-{tag}-e1a-spatial-LI-t*")):
        eps = [json.loads(l) for l in open(d + "/episodes.jsonl")]
        for e, ch in zip(eps, per_episode_chunks(d)):
            s = e["safelibero"]
            if not s["collided"]:
                continue
            t, ep, ct = int(e["task_id"]), int(e["episode_idx"]), int(s["collide_t"])
            cs = [json.loads(x) for x in open(f"{d}/episodes/safelibero_spatial_t{t}_e{ep}_calls.jsonl")]
            seq = max((c["seq"] for c in cs if c["obs_index"] <= ct), default=0)
            g = np.load(f"{d}/episodes/safelibero_spatial_t{t}_e{ep}_geom.npz")
            byts = dict(ch)
            line = []
            for q in range(max(0, seq - 3), seq + 1):
                c = next((x for x in cs if x["seq"] == q), None)
                if c is None:
                    continue
                if c["hold"]:
                    line.append(f"{q}:HOLD")
                    tot["HOLD"] += 1
                    continue
                cp = byts.get(q)
                if cp is None:
                    line.append(f"{q}:nochunk")
                    continue
                z = np.load(cp)
                sm = json.loads(str(z["summary_json"]))
                src = (sm.get("execution_path") or {}).get("source")
                to = sm.get("to") or {}
                pc = lit((to.get("metrics") or {}).get("passthrough_clear"))
                ob = (pc.get("obstacle_rows") or {})
                w = ob.get("worst") or {}
                i = int(np.argmin(np.abs(g["obs_index"] - c["obs_index"])))
                pb = (np.linalg.inv(g["T_world_base"][i]) @ np.r_[np.asarray(s["obstacle_initial_pos"], float), 1])[:3]
                dist = None if not w.get("point_m") else round(1000 * float(np.linalg.norm(np.asarray(w["point_m"]) - pb)))
                viol = ob.get("violation_m")
                tag_ = ("pass" if pc.get("applied") else "solve") if src == "reference" else str(src)
                if not pc.get("applied"):
                    tag_ += f"[{((to.get('metrics') or {}).get('returned'))}]"
                tot[tag_] += 1
                if pc.get("applied") and viol and viol > 0:
                    tot["pass_with_obstacle_violation"] += 1
                line.append(f"{q}:{tag_}:{pc.get('reason')}:obsviol{None if viol is None else round(1000*viol,1)}mm"
                            f":nobs{w.get('n_obstacle_rows')}:{w.get('link')}:d_obst{dist}mm")
            out.append(dict(task=t, ep=ep, obst=s["obstacle"], seq=seq, calls=line))
            print(f"t{t}e{ep:2d} {s['obstacle'][:14]:14s} seq{seq}: " + " | ".join(line))
    print("TOTAL", dict(tot))
    json.dump(out, open(f"/mnt/dev/AG3S_VLA/outputs/2026-11-09_집기놓기_장애물취급/e1a_decomp/passthrough_window_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
