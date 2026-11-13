#!/usr/bin/env python3
"""(2026-11-13 sdm-f6) Offline replay of the per-frame target grounding on the recorded pre-grasp chunks of the frozen version
(no identity memory: `ground_target(..., confirm=None)`), under alternative ranking rules, scored against the GT body of the
chosen cluster. Answers: when the server picks a non-task body, is it the per-frame ranking (score = (0.7 mean attention + 0.3
geometry) / 1.0) or the identity memory (first adoption kept)? Rules:
  frozen   - the frozen version's clustering config as built for the server (split radius 12 mm, raw ratio, partition)
  attn     - same, w_geometry = 0 (rank by mean attention only)
  mass     - rank by attention mass (sum) among clusters
Admissibility is mimicked as `narrowest extent <= 78 mm` (T26) for every rule. Recorded = the server's actual target.
usage: .venv-ag3s/bin/python grounding_replay.py OUT.json"""
import ast, glob, json, os, re, sys, collections, dataclasses
import numpy as np
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.adapters.panda import profile as pf
from AG3S.ag3s.stages import target_grounding as tg
from AG3S.ag3s.stages.admissibility import principal_extents
from AG3S.ag3s.types import AttentionPointCloud, PointCloud
B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
FZ = pf.FROZEN_20261112
TTDIRS = ["/tmp/claude-0/-mnt-dev/a2a2d792-941a-400f-8ace-60815fa2c777/scratchpad/common/tt",
          "/tmp/claude-0/-mnt-dev/a2a2d792-941a-400f-8ace-60815fa2c777/scratchpad/tt_tab10"]
OPEN = 0.078


def clustering_config():
    args = pf.parse_v11ab_args(argv=pf.version_server_argv(FZ) + list(pf.AG3S_VERSIONS[FZ]))
    cfg = pf.panda_ag3s_config(pf.v11ab_ag3s_config(pf.v11ab_build_kwargs(args)))
    return cfg


def goals():
    out = {}
    for dd in TTDIRS:
        for f in glob.glob(dd + "/*.log"):
            for l in open(f, errors="ignore"):
                m = re.match(r"safelibero_(\w+?)_t(\d)_e(\d+) goal (\[.*?\])", l)
                if m:
                    out[(m.group(1), int(m.group(2)))] = ast.literal_eval(m.group(4))
    return out


def objs(g):
    names = [str(n) for n in g["model_joint_names"]]; S = g["sim_state"]; adr = 1; out = {}
    for n in names:
        if n.endswith("_joint0"):
            out[n[:-7]] = S[:, adr:adr + 3].copy(); adr += 7
        else:
            adr += 1
    return out


def eps_of(D):
    ch = sorted(glob.glob(D + "/server_constraints/run_0000/chunk_*.npz")); eps = []; cur = []; last = -1
    for f in ch:
        ts = int(np.load(f, allow_pickle=True)["t_step"])
        if ts <= last and cur:
            eps.append(cur); cur = []
        cur.append(f); last = ts
    if cur:
        eps.append(cur)
    return eps


def pick(res, rule):
    cl = [c for c in (res.clusters or []) if c.principal_extents is not None and float(np.min(c.principal_extents)) <= OPEN]
    if not cl:
        return None
    key = {"frozen": lambda c: c.target_score, "attn": lambda c: c.mean_attention, "mass": lambda c: c.attention_mass}[rule]
    return max(cl, key=key)


def main():
    cfg = clustering_config(); G = goals()
    cc = cfg.clustering
    print("clustering:", {k: getattr(cc, k) for k in ("w_attention", "w_geometry", "attention_split_radius", "attention_split_ratio_source", "attention_split_partition", "eps")})
    tot = {r: collections.Counter() for r in ("recorded", "frozen", "attn", "mass")}
    global WHY, MERGED
    WHY, MERGED = collections.Counter(), collections.Counter()
    per_ep = []
    for pat in ("2*_ms-pkgFrssdm-*", "2*_ms-tab10sdm-*-i19-*", "2*_sub-pkgFrssdm-e1a-spatial-LI-t*"):
        for D in sorted(glob.glob(B + pat)):
            if not os.path.isfile(D + "/episodes.jsonl") or os.path.getsize(D + "/episodes.jsonl") == 0:
                continue
            m = re.search(r"_ms-[^-]+-(\w+)-(LI+)-i", D)
            suite, lvl = (m.group(1), m.group(2)) if m else ("spatial", "LI")
            E = [json.loads(l) for l in open(D + "/episodes.jsonl") if l.strip()]
            chunks = eps_of(D)
            if len(chunks) != len(E):
                continue
            for e, files in zip(E, chunks):
                t, i = int(e["task_id"]), int(e["init_state_index"]); goal = G.get((suite, t))
                if goal is None:
                    continue
                s = e.get("safelibero") or {}
                lab = "SS" if e["official_success"] and not s.get("collided") else "notSS"
                cf = glob.glob(D + f"/episodes/*_t{t}_e{i}_calls.jsonl")[0]; cs = [json.loads(x) for x in open(cf)]
                g = np.load(cf.replace("_calls.jsonl", "_geom.npz"), allow_pickle=True); O = objs(g); oi = [int(x) for x in g["obs_index"]]
                T = np.asarray(g["T_world_base"], np.float64).reshape(-1, 4, 4)
                ep = {r: collections.Counter() for r in tot}
                ep["sel_err_why"] = collections.Counter(); ep["merged_ext"] = collections.Counter()
                for f in files:
                    z = np.load(f, allow_pickle=True); sq = int(z["t_step"]); sm = json.loads(str(z["summary_json"]))
                    if ((sm.get("grasp") or {}).get("state")) in ("held", "closing"):
                        break
                    if sq >= len(cs) or cs[sq]["obs_index"] not in oi:
                        continue
                    k = oi.index(cs[sq]["obs_index"]); Tinv = np.linalg.inv(T[min(k, len(T) - 1)])
                    gt = {n: (Tinv @ np.r_[p[k], 1.0])[:3] for n, p in O.items() if p[k, 2] < 2 and np.linalg.norm((Tinv @ np.r_[p[k], 1.0])[:3]) < 1.0}
                    P = np.asarray(z["attention_cloud"], np.float32); A = np.asarray(z["attention_values"], np.float32)
                    AR = np.asarray(z["attention_raw"], np.float32); sup = np.asarray(z["support_mask"], bool)
                    owner = lambda pts: min(gt, key=lambda n: np.linalg.norm(gt[n] - pts.mean(0)))
                    tp = z["target_points"] if "target_points" in z.files else np.zeros((0, 3))
                    if len(tp):
                        ep["recorded"]["task" if owner(np.asarray(tp)) in goal else "other"] += 1
                    cloud = AttentionPointCloud(cloud=PointCloud(points=P), attention=A, raw_attention=AR) if "cloud" in AttentionPointCloud.__dataclass_fields__ else AttentionPointCloud(points=P, attention=A, raw_attention=AR)
                    try:
                        res = tg.ground_target(cloud, cc, exclude_mask=sup)
                    except Exception as ex:
                        ep["frozen"]["error"] += 1; continue
                    # where is the task object in this frame's candidate list? (SL-SOTA19 diagnosis)
                    cl = list(res.clusters or [])
                    own = [owner(P[np.asarray(c.point_indices)]) for c in cl]
                    task_any = [c for c, o in zip(cl, own) if o in goal]
                    task_ok = [c for c in task_any if c.principal_extents is not None and float(np.min(c.principal_extents)) <= OPEN]
                    means = {}
                    for n, q in gt.items():
                        msk = (np.linalg.norm(P - q, axis=1) < 0.06) & ~sup
                        if msk.sum() >= 15:
                            means[n] = float(A[msk].mean())
                    att_task = bool(means) and max(means, key=means.get) in goal
                    rec_task = bool(len(tp)) and owner(np.asarray(tp)) in goal
                    if att_task and not rec_task:
                        why = ("no cluster at all" if not cl else "task object in no cluster (seeds elsewhere)" if not task_any else
                               "task object only in non-graspable clusters" if not task_ok else "graspable task cluster exists, not chosen")
                        ep["sel_err_why"][why] += 1
                        if task_any and not task_ok:
                            ep["merged_ext"][str(sorted(round(float(x) * 1000) for x in task_any[0].principal_extents))] += 1
                    for rule in ("frozen", "attn", "mass"):
                        c = pick(res, rule)
                        if c is None:
                            ep[rule]["none"] += 1
                        else:
                            ep[rule]["task" if owner(P[np.asarray(c.point_indices)]) in goal else "other"] += 1
                for r in tot:
                    tot[r].update(ep[r])
                WHY.update(ep["sel_err_why"]); MERGED.update(ep["merged_ext"])
                per_ep.append({"suite": suite, "level": lvl, "task": t, "init": i, "label": lab, **{r: dict(v) for r, v in ep.items()}})
            print(os.path.basename(D)[16:70], flush=True)
    print("selection errors (attention top = task object, server target = other), why:", dict(WHY))
    print("extents (mm) of the non-graspable cluster holding the task object, most common:", MERGED.most_common(8))
    json.dump({"totals": {r: dict(v) for r, v in tot.items()}, "selection_error_why": dict(WHY), "episodes": per_ep}, open(sys.argv[1], "w"), indent=1)
    for r, v in tot.items():
        n = v["task"] + v["other"]
        print(f"{r:9s} task object {v['task']} / {n} = {100 * v['task'] / max(n, 1):.1f} %  (none {v.get('none', 0)}, error {v.get('error', 0)})")


if __name__ == "__main__":
    main()
