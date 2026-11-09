"""(2026-11-09 12:3x KST dev-4a) E4-F failure labels (plan §1.7, Tab. S9) for an AG3S E1a arm, automatic part only.
Every episode that is not a safe success gets: outcome (S+C / F+C / F), and
  collided -> mechanism at the call that executed into the hit (seq = last call with obs_index <= collide_t):
     P2-merge  obstacle absent from the scene obstacle-object set at seq (0 voxels within 80 mm of the obstacle) while
               present earlier in the episode  (= merged into the target's 15 mm component; replay of
               AG3S._scene_obstacle_points on the recorded chunk, no 3-request union)
     P1-unseen obstacle absent from the set on every recorded call up to seq (never observed / occluded)
     A-tauobs  pass-through with obstacle-row penetration in (0, tau_obs]
     A3-model  solve (action QP / SQP) returned a chunk the check called clear, the executed motion still hit
     other     none of the above (e.g. arm link rows not classed as obstacle rows while held)
  not collided -> end reason: A1-hold (>= 30 % of calls HOLD, or the last 3 calls HOLD), grasp-none (never 'held'),
     place-fail (held, not placed in time), other.
GT is used only for scoring (SafeLIBERO obstacle position / collision time). usage: python e4f_labels.py <run-id tag> [...]"""
import ast
import collections
import glob
import json
import sys
import types

import numpy as np

sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.runtime.pipeline import AG3S  # noqa: E402

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
OUT = "/mnt/dev/AG3S_VLA/outputs/2026-11-09_최종실험_자료수집/"


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


def obstacle_voxels(z, pb):
    st = types.SimpleNamespace(_obstacle_history=[])
    tgt = z["target_points"].astype(float) if "target_points" in z.files and len(z["target_points"]) else None
    out, _ = AG3S._scene_obstacle_points(st, z["filtered_cloud"].astype(float), z["support_mask"].astype(bool), tgt, None)
    return 0 if out is None or not len(out) else int((np.linalg.norm(out - pb, axis=1) <= 0.08).sum())


def label(d, e, ch):
    s = e["safelibero"]
    t, ep = int(e["task_id"]), int(e["episode_idx"])
    stem = f"safelibero_spatial_t{t}_e{ep}"
    cs = [json.loads(x) for x in open(f"{d}/episodes/{stem}_calls.jsonl")]
    states = [(c.get("grasp") or {}).get("state") for c in cs]
    holds = [bool(c["hold"]) for c in cs]
    succ, coll = bool(e["official_success"]), bool(s["collided"])
    r = dict(task=t, ep=ep, outcome={(True, True): "S+C", (False, True): "F+C", (False, False): "F", (True, False): "SS"}[(succ, coll)],
             obstacle=s["obstacle"].replace("_obstacle_1", ""), disp_mm=round(1000 * s["max_displacement_m"], 1),
             n_calls=len(cs), n_hold=sum(holds), reached_held="held" in states, video=e.get("video"))
    if r["outcome"] == "SS":
        return r
    if coll:
        ct = int(s["collide_t"])
        seq = max((c["seq"] for c in cs if c["obs_index"] <= ct), default=0)
        g = np.load(f"{d}/episodes/{stem}_geom.npz")
        byts = dict(ch)
        r.update(collide_t=ct, seq=seq, phase="held" if states[min(seq, len(states) - 1)] == "held" else "pre-held")
        vox = []
        for q in sorted(k for k in byts if k <= seq):
            c = next((x for x in cs if x["seq"] == q), None)
            if c is None:
                continue
            i = int(np.argmin(np.abs(g["obs_index"] - c["obs_index"])))
            pb = (np.linalg.inv(g["T_world_base"][i]) @ np.r_[np.asarray(s["obstacle_initial_pos"], float), 1])[:3]
            vox.append((q, obstacle_voxels(np.load(byts[q]), pb)))
        at = dict(vox).get(seq)
        hit = next((x for x in cs if x["seq"] == seq), {})
        mech = "other"
        if hit.get("hold"):
            mech = "other"
        elif at is not None and at == 0:
            mech = "P2-merge" if any(v > 0 for _, v in vox) else "P1-unseen"
        elif seq in byts:
            m = ((json.loads(str(np.load(byts[seq])["summary_json"])).get("to") or {}).get("metrics") or {})
            pc = lit(m.get("passthrough_clear"))
            v = float(((pc.get("obstacle_rows") or {}).get("violation_m")) or 0.0)
            if pc.get("applied"):
                mech = "A-tauobs" if v > 0 else "other"
            else:
                mech = "A3-model"
        r.update(mechanism=mech, obstacle_voxels_at_hit=at, obstacle_voxels_trace=vox[-6:])
    else:
        last3 = holds[-3:]
        if sum(holds) >= 0.3 * len(holds) or (len(last3) == 3 and all(last3)):
            reason = "A1-hold"
        elif "held" not in states:
            reason = "grasp-none"
        else:
            reason = "place-fail"
        r.update(end_reason=reason, final_grasp_state=states[-1] if states else None)
    return r


def main(tags):
    allr = {}
    for tag in tags:
        rows = []
        for d in sorted(glob.glob(B + f"2*qe-{tag}-e1a-spatial-LI-t*")):
            if "INVALID" in d:
                continue
            eps = [json.loads(l) for l in open(d + "/episodes.jsonl")]
            for e, ch in zip(eps, per_episode_chunks(d)):
                rows.append(label(d, e, ch))
        allr[tag] = rows
        cnt = collections.Counter(r["outcome"] for r in rows)
        mech = collections.Counter(r.get("mechanism") for r in rows if "mechanism" in r)
        endr = collections.Counter(r.get("end_reason") for r in rows if "end_reason" in r)
        print(f"== {tag} n={len(rows)} {dict(cnt)}\n   collision mechanism {dict(mech)}\n   no-collision failure {dict(endr)}")
        for r in rows:
            if r["outcome"] != "SS":
                print("  ", f"t{r['task']}e{r['ep']:2d}", r["outcome"], r["obstacle"][:12], r.get("mechanism") or r.get("end_reason"),
                      r.get("phase", ""), f"hold {r['n_hold']}/{r['n_calls']}", "held" if r["reached_held"] else "-",
                      r.get("obstacle_voxels_trace", ""))
    json.dump(allr, open(OUT + "e4f_labels_" + "_".join(tags) + ".json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1:])
