"""IFG held-out — aggregation of the locked arms (dev rule: frame -> episode mean -> episode paired bootstrap 2000, seed 20261015).

Arms (rows written by ifg_heldout_eval):
  A        rounds/R10b  arm "R6"                          = top1-only-ns+rm:N (adopted extent, no memory switch)
  B        rounds/R10b  arm "Mem-first:prefix_L8:d0.05"   = A + memory switch (adopted method, centered)
  A-fk     rounds/R6    arm "top1-only-ns:N"              = A with the serving FK-sphere robot mask
  G:N      rounds/R6    arm "G:N"                         = R0 reference
Gate: rounds/R11b/conf.jsonl, q_mem_chain:prefix_L8 < LOCK threshold -> alert; wrong choice = r6_iou < 0.1.
Main = episodes of successful rollouts; failures reported separately (appendix).
"""
import argparse, collections, json, pathlib

import numpy as np

ARMS = {"A": ("R10b", "R6"), "B": ("R10b", "Mem-first:prefix_L8:d0.05"), "A-fk": ("R6", "top1-only-ns:N"), "G:N": ("R6", "G:N")}
METRICS = ("iou", "precision", "recall")


def load_rows(run):
    out = collections.defaultdict(dict)
    for rnd in ("R6", "R10b"):
        for f in (run / "rounds" / rnd / "rows").glob("*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                out[(rnd, r["arm_id"])][r["frame_id"]] = r
    return out


def ep_means(rows, frames, metric):
    by = collections.defaultdict(list)
    for fid, fr in frames.items():
        r = rows.get(fid)
        if r is None or not r.get("valid", r.get("missing_reason") is None) or r.get(metric) is None:
            continue
        by[fr["episode_id"]].append(float(r[metric]))
    return {e: float(np.mean(v)) for e, v in by.items()}


def paired(a, b, seed=20261015):
    es = sorted(set(a) & set(b))
    if not es:
        return None
    d = np.array([a[e] - b[e] for e in es])
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return dict(n_episodes=len(es), diff=float(d.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum()))


def block(rows, frames):
    res = {}
    for arm, key in ARMS.items():
        r = rows[key]
        res[arm] = {}
        for m in METRICS:
            em = ep_means(r, frames, m)
            res[arm][m] = dict(episode_mean=float(np.mean(list(em.values()))) if em else None, n_episodes=len(em))
        valid = [fid for fid in frames if fid in r and r[fid].get("valid", r[fid].get("missing_reason") is None) and r[fid].get("iou") is not None]
        res[arm]["n_valid_frames"] = len(valid)
        for k in ("neighbor_fp", "support_fp", "robot_fp", "FP", "TP"):
            res[arm][k + "_sum"] = float(sum(r[f].get(k) or 0 for f in valid))
    res["paired_iou"] = {f"{x} - {y}": paired(ep_means(rows[ARMS[x]], frames, "iou"), ep_means(rows[ARMS[y]], frames, "iou"))
                         for x, y in (("B", "A"), ("A", "A-fk"), ("A", "G:N"), ("B", "G:N"))}
    res["paired_precision"] = {"A - A-fk": paired(ep_means(rows[ARMS["A"]], frames, "precision"), ep_means(rows[ARMS["A-fk"]], frames, "precision"))}
    res["paired_recall"] = {"A - A-fk": paired(ep_means(rows[ARMS["A"]], frames, "recall"), ep_means(rows[ARMS["A-fk"]], frames, "recall"))}
    sw = [rows[ARMS["B"]][f].get("switched") for f in frames if f in rows[ARMS["B"]]]
    res["B_switch_frames"] = int(sum(bool(s) for s in sw))
    return res


def gate(conf, frames, thr):
    C = [c for c in conf if c["frame_id"] in frames and c.get("q_mem_chain:prefix_L8") is not None and c["target_voxels"] > 0]
    wrong = [c for c in C if (c["r6_iou"] or 0) < 0.1]
    alert = [c for c in C if c["q_mem_chain:prefix_L8"] < thr]
    aw = [c for c in alert if (c["r6_iou"] or 0) < 0.1]
    fp_all = sum(c["r6_fp"] or 0 for c in C); tp_all = sum(c["r6_tp"] or 0 for c in C)
    fp_rm = sum(c["r6_fp"] or 0 for c in alert); tp_rm = sum(c["r6_tp"] or 0 for c in alert)
    s = np.array([c["q_mem_chain:prefix_L8"] for c in C]); y = np.array([(c["r6_iou"] or 0) < 0.1 for c in C])
    auroc = None
    if y.any() and (~y).any():  # P(score of a wrong choice < score of a right one)
        auroc = float(np.mean([(a < b) + 0.5 * (a == b) for a in s[y] for b in s[~y]]))
    return dict(threshold=thr, n_frames=len(C), wrong_choices=len(wrong), alerts=len(alert), alerts_on_wrong=len(aw),
                wrong_caught_ratio=(len(aw) / len(wrong)) if wrong else None, false_alarms=len(alert) - len(aw),
                fp_removed_ratio=(fp_rm / fp_all) if fp_all else None, tp_lost_ratio=(tp_rm / tp_all) if tp_all else None, auroc_wrong=auroc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--locked", type=pathlib.Path, required=True)
    ap.add_argument("--lock", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    run = a.locked / "run"
    fr = json.loads((a.locked / "frames.json").read_text())
    frames = {f["frame_id"]: f for f in fr["frames"]}
    rows = load_rows(run)
    conf = [json.loads(l) for l in open(run / "rounds/R11b/conf.jsonl")]
    thr = json.loads(a.lock.read_text())["arms"]["confidence_gate"]["abstain_if_below"]
    sel = lambda pred: {k: v for k, v in frames.items() if pred(v)}
    succ = sel(lambda f: f["kind"] == "success")
    out = dict(rule=__doc__, skipped=fr["skipped"], n_frames_total=len(frames),
               main_success=block(rows, succ), failure_appendix=block(rows, sel(lambda f: f["kind"] == "failure")),
               by_stratum={s: block(rows, sel(lambda f, s=s: f["kind"] == "success" and f["stratum"] == s)) for s in ("clear", "adjacent", "place")},
               by_grasp_state={g: block(rows, sel(lambda f, g=g: f["kind"] == "success" and f["grasp_state"] == g)) for g in ("pre", "held", "placed")},
               gate_success=gate(conf, succ, thr), gate_all=gate(conf, frames, thr))
    a.out.write_text(json.dumps(out, indent=1, default=float))
    m = out["main_success"]
    print("success episodes:", {k: (round(v["iou"]["episode_mean"], 3), v["iou"]["n_episodes"], v["n_valid_frames"]) for k, v in m.items() if k in ARMS})
    for k, v in m["paired_iou"].items():
        print(" ", k, v and (round(v["diff"], 3), [round(x, 3) for x in v["ci95"]], v["better"], v["worse"]))
    print("  B switches", m["B_switch_frames"])
    for s, b in out["by_stratum"].items():
        print(s, {k: round(b[k]["iou"]["episode_mean"] or 0, 3) for k in ARMS}, "n_ep", b["A"]["iou"]["n_episodes"])
    for g, b in out["by_grasp_state"].items():
        print(g, {k: round(b[k]["iou"]["episode_mean"] or 0, 3) for k in ARMS})
    print("gate", out["gate_success"])
    f = out["failure_appendix"]
    print("failures:", {k: (round(f[k]["iou"]["episode_mean"] or 0, 3), f[k]["iou"]["n_episodes"]) for k in ARMS})


if __name__ == "__main__":
    main()
