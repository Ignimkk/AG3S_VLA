#!/usr/bin/env python3
"""(2026-11-13 sdm-f6) Is a wrong target the VLA attention's fault or the selection's? For every pre-grasp call of the frozen
version's SafeLIBERO runs: A = the body with the highest MEAN attention (non-support points within 6 cm of each body's GT
position, >= 15 points) is a task object (BDDL obj_of_interest); R = the recognised target (chunk target_points, nearest GT body)
is a task object. Per call: A&R right, A&~R selection error (attention points at the task object, the server picks another),
~A&~R attention error, ~A&R right despite attention. Reports per episode / suite and the episode-level majority.
usage: .venv-ag3s/bin/python attn_vs_target.py OUT.json"""
import ast, glob, json, os, re, sys, collections
import numpy as np
B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
TTDIRS = ["/tmp/claude-0/-mnt-dev/a2a2d792-941a-400f-8ace-60815fa2c777/scratchpad/common/tt",
          "/tmp/claude-0/-mnt-dev/a2a2d792-941a-400f-8ace-60815fa2c777/scratchpad/tt_tab10"]


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


def main():
    G = goals(); res = []
    for pat in ("2*_ms-pkgFrssdm-*", "2*_ms-tab10sdm-*-i19-*", "2*_sub-pkgFrssdm-e1a-spatial-LI-t*"):
        for D in sorted(glob.glob(B + pat)):
            if not os.path.isfile(D + "/episodes.jsonl") or os.path.getsize(D + "/episodes.jsonl") == 0:
                continue
            m = re.search(r"_ms-[^-]+-(\w+)-(LI+)-i", D)
            suite, lvl = (m.group(1), m.group(2)) if m else ("spatial", "LI")
            E = [json.loads(l) for l in open(D + "/episodes.jsonl") if l.strip()]
            chunks = eps_of(D)
            if len(chunks) != len(E):
                print("skip (episode/chunk count)", os.path.basename(D)[:60], len(chunks), len(E)); continue
            for e, files in zip(E, chunks):
                t, i = int(e["task_id"]), int(e["init_state_index"]); goal = G.get((suite, t))
                if goal is None:
                    continue
                s = e.get("safelibero") or {}
                lab = "SS" if e["official_success"] and not s.get("collided") else "S+C" if e["official_success"] else "F+C" if s.get("collided") else "F"
                cf = glob.glob(D + f"/episodes/*_t{t}_e{i}_calls.jsonl")[0]; cs = [json.loads(x) for x in open(cf)]
                g = np.load(cf.replace("_calls.jsonl", "_geom.npz"), allow_pickle=True); O = objs(g); oi = [int(x) for x in g["obs_index"]]
                T = np.asarray(g["T_world_base"], np.float64).reshape(-1, 4, 4)
                cnt = collections.Counter(); top_names = collections.Counter(); rec_names = collections.Counter()
                for f in files:
                    z = np.load(f, allow_pickle=True); sq = int(z["t_step"]); sm = json.loads(str(z["summary_json"]))
                    if ((sm.get("grasp") or {}).get("state")) in ("held", "closing"):
                        break
                    if sq >= len(cs) or cs[sq]["obs_index"] not in oi:
                        continue
                    k = oi.index(cs[sq]["obs_index"]); Tinv = np.linalg.inv(T[min(k, len(T) - 1)])
                    P = np.asarray(z["attention_cloud"], np.float64); A = np.asarray(z["attention_values"], np.float64); sup = np.asarray(z["support_mask"], bool)
                    gt = {n: (Tinv @ np.r_[p[k], 1.0])[:3] for n, p in O.items() if p[k, 2] < 2 and np.linalg.norm((Tinv @ np.r_[p[k], 1.0])[:3]) < 1.0}
                    means = {}
                    for n, q in gt.items():
                        msk = (np.linalg.norm(P - q, axis=1) < 0.06) & ~sup
                        if msk.sum() >= 15:
                            means[n] = float(A[msk].mean())
                    tp = z["target_points"] if "target_points" in z.files else np.zeros((0, 3))
                    if not means or len(tp) == 0:
                        cnt["no target / no attention"] += 1; continue
                    top = max(means, key=means.get); rec = min(gt, key=lambda n: np.linalg.norm(gt[n] - np.asarray(tp).mean(0)))
                    a, r = top in goal, rec in goal
                    cnt["right" if a and r else "selection error" if a else "attention error" if not r else "right despite attention"] += 1
                    top_names[top] += 1; rec_names[rec] += 1
                res.append({"suite": suite, "level": lvl, "task": t, "init": i, "label": lab, "calls": dict(cnt),
                            "attention_top": dict(top_names.most_common(3)), "recognised": dict(rec_names.most_common(3)), "goal": goal})
            print(os.path.basename(D)[16:70], "ok", flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    tot = collections.Counter(); ep = collections.Counter(); epss = collections.defaultdict(lambda: [0, 0])
    for r in res:
        c = r["calls"]; n = sum(v for k, v in c.items() if k != "no target / no attention")
        tot.update(c)
        if n == 0:
            continue
        major = max(("right", "selection error", "attention error", "right despite attention"), key=lambda k: c.get(k, 0))
        if c.get("right", 0) + c.get("right despite attention", 0) >= n / 2:
            major = "right (>= half)"
        ep[major] += 1; epss[major][0] += 1; epss[major][1] += r["label"] == "SS"
    print("calls:", dict(tot))
    print("episodes by majority:", {k: f"{v[0]} ep, SS {v[1]}" for k, v in epss.items()})


if __name__ == "__main__":
    main()
