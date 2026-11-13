#!/usr/bin/env python3
"""(2026-11-13 sdm-f6) Which initial states this server runs: 1/10 of the paper scale, chosen for the highest expected success
(user instruction 2026-11-13 10:5x: "성공률이 제일 높게 나올 것 같은 init 을 선택"; the other 9/10 runs elsewhere).

SafeLIBERO (Tab. 1): per cell (suite x level) the 5 of init 5-49 (the launchers' test split; init 0-4 go to the other 9/10) with the highest expected safe-success sum over tasks 0-3
(one init set per cell, so one server per cell). Expected safe success of (task, init):
  * AG3S measured with the version proxy (`--ag3s-tags`, newest first): 0.98 if SS, else 0.02 (same machine + seed reproduce);
  * else the cell-task prior (Laplace over its measured inits) x the pi0.5-alone likelihood ratio, when pi0.5 alone (same
    launcher `--modes vla_only`, or the earlier baseline launcher) has the episode: LR = P(SS | pi0.5 success) / P(SS) or
    P(SS | pi0.5 failure) / P(SS), fitted on every episode where both are measured.
LIBERO (Tab. 1 column): per suite the 5 of init 5-49 with the highest expected success over tasks 0-9 (same scheme, success).
LIBERO-Safety (Tab. 2): one init per suite x level (recorded only: no evidence to choose on except init 19).
This 1/10 is NOT a random sample and must not be reported alone; the reportable result is the union with the other 9/10.
usage: .venv-ag3s/bin/python select_inits.py --ag3s-tags pkgFrssdm,tab10sdm OUT.json"""
import argparse, csv, glob, json, os, re, collections
B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
SUITES = ("spatial", "object", "goal", "long")
LB = ("spatial", "object", "goal", "10")


def load(d):
    out = {}
    if "INVALID" in os.path.basename(d) or not os.path.isfile(d + "/episodes.jsonl"):
        return out
    for l in open(d + "/episodes.jsonl"):
        if not l.strip():
            continue
        e = json.loads(l)
        s = e.get("safelibero") or {}
        ok = bool(e.get("official_success"))
        out[(int(e["task_id"]), int(e["init_state_index"]))] = (ok and not s.get("collided"), ok)
    return out


def ag3s_safelibero(tags, suite, lvl):
    res = {}
    for tag in reversed(tags):            # newest tag last -> overrides
        pats = [f"2*_ms-{tag}-{suite}-L{lvl}-i*"]
        if suite == "spatial" and lvl == "I":
            pats.append(f"2*_sub-{tag}-e1a-spatial-LI-t*")
        for p in pats:
            for d in sorted(glob.glob(B + p)):
                res.update(load(d))
    return res


def pi05_safelibero(suite, lvl):
    res = {}
    for p in (f"2*_e1a-none-{suite}-L{lvl}-L{lvl}-none", f"2*_blt1-none-{suite}-L{lvl}-L{lvl}-none",
              f"2*_sdm-pi05-sl-{suite}-L{lvl}-L{lvl}-vla_only", f"2*_tab10sdm-pi05-sl-{suite}-L{lvl}-L{lvl}-vla_only"):
        for d in sorted(glob.glob(B + p)):
            res.update(load(d))
    return res


def likelihood_ratios(pairs):
    """pairs: list of (ag3s_ss, pi05_success)."""
    n = len(pairs)
    p = sum(a for a, _ in pairs) / n
    s = [a for a, b in pairs if b]
    f = [a for a, b in pairs if not b]
    return {"p_ss": p, "lr_success": (sum(s) / len(s)) / p, "lr_failure": (sum(f) / len(f)) / p, "n": n, "n_pi05_success": len(s)}


def choose(expected, inits, k):
    """expected[(task, init)] -> per-init sum; the k best inits (ties: lower index)."""
    score = {i: sum(v for (t, j), v in expected.items() if j == i) for i in inits}
    return sorted(sorted(score, key=lambda i: (-score[i], i))[:k]), score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag3s-tags", required=True, help="comma-separated run tags of the version proxy, oldest first")
    ap.add_argument("out")
    a = ap.parse_args()
    tags = a.ag3s_tags.split(",")
    cells = [(s, l) for s in SUITES for l in ("I", "II")]
    ag = {c: ag3s_safelibero(tags, *c) for c in cells}
    pi = {c: pi05_safelibero(*c) for c in cells}
    pairs = [(v[0], pi[c][k][1]) for c in cells for k, v in ag[c].items() if k in pi[c]]
    lr = likelihood_ratios(pairs)
    out = {"rule": __doc__, "ag3s_tags": tags, "likelihood": lr, "safelibero": {}, "libero": {}, "libero_safety": {}}
    for c in cells:
        exp = {}
        for t in range(4):
            known = [ag[c][(t, i)][0] for i in range(50) if (t, i) in ag[c]]
            prior = (sum(known) + 1) / (len(known) + 2)
            for i in range(50):
                if (t, i) in ag[c]:
                    exp[(t, i)] = 0.98 if ag[c][(t, i)][0] else 0.02
                elif (t, i) in pi[c]:
                    exp[(t, i)] = min(0.95, prior * (lr["lr_success"] if pi[c][(t, i)][1] else lr["lr_failure"]))
                else:
                    exp[(t, i)] = prior
        sel, score = choose(exp, range(5, 50), 5)
        out["safelibero"][f"{c[0]} L-{c[1]}"] = {
            "inits": sel, "expected_ss_of_4_tasks": {i: round(score[i], 2) for i in sel},
            "expected_ss_total": round(sum(score[i] for i in sel), 2),
            "measured_ag3s_ss_in_selection": sum(1 for (t, i), v in ag[c].items() if i in sel and v[0]),
            "complement_inits": [i for i in range(50) if i not in sel]}
    # LIBERO: AG3S known (gtto libero9 = which suites? recorded per run), tab10sdm-lb; pi0.5 blt1 + tab10sdm-pi05-lb
    for s in LB:
        agl, pil = {}, {}
        for d in sorted(glob.glob(B + f"2*_tab10sdm-lb-{s}-vla_v11ab-*")):
            agl.update({k: v[1] for k, v in load(d).items()})
        for p in (f"2*_blt1-none-{s}-none", f"2*_tab10sdm-pi05-lb-{s}-*"):
            for d in sorted(glob.glob(B + p)):
                pil.update({k: v[1] for k, v in load(d).items()})
        exp = {}
        for t in range(10):
            for i in range(50):
                exp[(t, i)] = 0.98 if agl.get((t, i)) else 0.02 if (t, i) in agl else (0.9 if pil.get((t, i)) else 0.5 if (t, i) in pil else 0.7)
        sel, score = choose(exp, range(5, 50), 5)
        out["libero"][s] = {"inits": sel, "expected_success_of_10_tasks": {i: round(score[i], 2) for i in sel},
                            "complement_inits": [i for i in range(50) if i not in sel]}
    for suite in ("affordance", "obstacle_avoidance", "obstacle_avoidance_human", "human_safety"):
        for lvl in (0, 1, 2):
            out["libero_safety"][f"{suite} L{lvl}"] = {"init": 19, "why": "only init with measurements (L2 tasks 0-1, pi0.5 official 4 / 6)"}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False, default=str)
    print("likelihood", {k: round(v, 3) if isinstance(v, float) else v for k, v in lr.items()})
    for c, v in out["safelibero"].items():
        print(f"SafeLIBERO {c:13s} inits {v['inits']}  expected SS {v['expected_ss_total']:.1f} / 20  (measured SS in it {v['measured_ag3s_ss_in_selection']})")
    for s, v in out["libero"].items():
        print(f"LIBERO {s:8s} inits {v['inits']}  expected {sum(v['expected_success_of_10_tasks'].values()):.1f} / 50")


if __name__ == "__main__":
    main()
