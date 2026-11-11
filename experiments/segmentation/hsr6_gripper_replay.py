#!/usr/bin/env python3
"""Offline replay (CPU, no GPU): does the hidden-state phase probe (HSR6-Panda, layer-4 KV) + gripper feedback (grasp latch)
read the task phase / grasp failures better than either alone?  (user request 2026-11-10 00:xx KST, dev-38)

Data: the two AG3S closed-loop runs that recorded **both** signals per policy request (`*_calls.jsonl`):
  nb1104-ag3s-final-LI / -LII  (AG3S-final-panda-v3-e0to, SafeLIBERO Spatial task 0-3 x dev init 0-4, seed 7) = 40 episodes.
  per request: `hsr6.p` (probe posterior, 5 Panda phases) and `grasp.state` / `grasp.evidence` (latch, measured opening).
The probe was trained on pi0.5-only LIBERO spatial (nb1104-probe-dev, init 1-4); these SafeLIBERO episodes are not in it.

Simulator truth per frame (fixed before results; SL6-B3 rule with sim-pose proxies because SafeLIBERO has no
phase_export contact track): bowl = BDDL target `akita_black_bowl_1`, destination = `plate_1`, both from `_geom.npz` sim_state;
eef = obs_robot0_eef_pos; gap = finger qpos[0] - qpos[1].
  held(f)    bowl z - z0 >= 2 cm  and  |bowl - eef| <= 8 cm                (lifted and in the hand)
  close run  maximal run of frames whose producing action closes (action[f-1, -1] > 0)
  grasp      frames of a close run that reaches held, from the run start to its first held frame
  lift       held and bowl xy moved < 5 cm from the lift point (bowl xy at the first held frame of the latest run)
  transport  held and moved >= 5 cm
  placement  after the first held frame of the episode, bowl xy within 6 cm of the plate xy (wins over lift / transport)
  approach   everything else (before, failed closes, drops)
Close-run outcome: HELD (reaches held) / EMPTY (never held, min gap <= 3 mm) / PARTIAL (never held, min gap > 3 mm).
(gap thresholds from t0e1 / t1e1: empty 1.5-2.4 mm, bowl rim 4.8-5.0 mm.)

Phase readers per request (fixed before results):
  P  probe argmax (5 phases)
  L  latch only: searching / latched -> approach, closing -> grasp, held -> carry (= lift | transport | placement), placed -> placement
  R  rule fusion -- the latch picks the level, the probe picks inside it:
       searching / latched : approach
       closing             : grasp if p(grasp)+p(lift)+p(transport)+p(placement) >= p(approach) else approach (a failing close)
       held                : approach if p(approach) >= 0.5 (dropped / regrasp) else argmax over lift / transport / placement
       placed              : placement if p(placement) >= p(approach) else approach
  H  HMM forward filter: emission = probe p (alpha 1) x P(latch state | phase) (leave-one-task-out counts, +1 smoothing),
     transitions fixed by hand: stay 0.85, next phase 0.10, back to approach 0.05 (from grasp..placement), placement->transport 0.03.
Events (fixed before results):
  failing-close alarm, fired at the first request inside [run start, run end + 5 frames]:
     Lg  latch evidence checks.not_empty is False (the gripper closed to empty)
     Hp  latch closing and p(approach) >= 0.7 on 2 consecutive requests (the policy itself is not in grasp)
     F   Lg or Hp (earliest)
  destination gate (where a destination may be registered): gL = latch held ; gR = latch held and R in {transport, placement}.

usage: .venv-ag3s/bin/python experiments/segmentation/hsr6_gripper_replay.py --out outputs/2026-11-10_HSR6_gripper_재생
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib

import numpy as np

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
RUNS = {"LI": "20261104T070028_fe3abf215dc4_nb1104-ag3s-final-LI-LI-vla_v11ab-AG3S-final-panda-v3-e0to-C",
        "LII": "20261104T082420_fe3abf215dc4_nb1104-ag3s-final-LII-LII-vla_v11ab-AG3S-final-panda-v3-e0to-C"}
PH = ("approach", "grasp", "lift", "transport", "placement")
TARGET, DEST = "akita_black_bowl_1_joint0", "plate_1_joint0"
RISE, NEAR, TRAVEL, PLACE_R, EMPTY_GAP = 0.02, 0.08, 0.05, 0.06, 0.003
LATCH = ("searching", "latched", "closing", "held", "placed")


def addr(names):
    a, out = 1, {}
    for n in names:
        out[str(n)] = a
        a += 7 if "joint0" in str(n) else 1
    return out


def truth(E, G):
    A = addr(G["model_joint_names"])
    ss = G["sim_state"]
    bowl = ss[:, A[TARGET]:A[TARGET] + 3]
    plate = ss[:, A[DEST]:A[DEST] + 3]
    eef = E["obs_robot0_eef_pos"]
    q = E["obs_robot0_gripper_qpos"]
    gap = q[:, 0] - q[:, 1]
    act = E["action"]
    wait = E["is_wait"].astype(bool)
    N = len(bowl)
    t0 = int(np.nonzero(~wait)[0][0]) if (~wait).any() else 0
    z0 = bowl[t0, 2]
    held = (bowl[:, 2] - z0 >= RISE) & (np.linalg.norm(bowl - eef, axis=1) <= NEAR)
    close = np.zeros(N, bool)
    close[1:] = act[:N - 1, -1] > 0
    lab = np.zeros(N, int)  # approach
    runs = []
    f = 1
    while f < N:
        if close[f]:
            s = f
            while f < N and close[f]:
                f += 1
            runs.append((s, f))  # [s, f)
        else:
            f += 1
    run_out = []
    for s, e in runs:
        hf = np.nonzero(held[s:e])[0]
        if len(hf):
            lab[s:s + hf[0]] = 1
            run_out.append({"start": s, "end": e, "outcome": "HELD", "first_held": int(s + hf[0])})
        else:
            run_out.append({"start": s, "end": e, "outcome": "EMPTY" if gap[s:e].min() <= EMPTY_GAP else "PARTIAL",
                            "min_gap_mm": float(gap[s:e].min() * 1e3)})
    lift_pt, first_held = None, None
    for f in range(N):
        if held[f]:
            if lift_pt is None or not held[f - 1]:
                lift_pt = bowl[f, :2].copy()
            first_held = f if first_held is None else first_held
            lab[f] = 2 if np.linalg.norm(bowl[f, :2] - lift_pt) < TRAVEL else 3
        if first_held is not None and np.linalg.norm(bowl[f, :2] - plate[f, :2]) <= PLACE_R:
            lab[f] = 4
    return lab, run_out, gap


def load():
    eps = []
    for lvl, rd in RUNS.items():
        R = RESULTS / rd
        succ = {}
        for line in open(R / "episodes.jsonl"):
            r = json.loads(line)
            succ[(r["task_id"], r["episode_idx"])] = bool(r["official_success"])
        for cj in sorted(glob.glob(str(R / "episodes" / "*_calls.jsonl"))):
            stem = os.path.basename(cj)[:-len("_calls.jsonl")]
            E = np.load(str(R / "episodes" / (stem + ".npz")))
            G = np.load(str(R / "episodes" / (stem + "_geom.npz")))
            lab, runs, gap = truth(E, G)
            rows = [json.loads(x) for x in open(cj) if x.strip()]
            calls = []
            for r in rows:
                h = r.get("hsr6") or {}
                p = np.array([float((h.get("p") or {}).get(c, np.nan)) for c in PH])
                ev = (r.get("grasp") or {}).get("evidence") or {}
                calls.append({"obs": int(r["obs_index"]), "p": p, "latch": (r.get("grasp") or {}).get("state"),
                              "not_empty": (ev.get("checks") or {}).get("not_empty"),
                              "closing_steps": ev.get("closing_steps")})
            task, ep = int(stem.split("_t")[1].split("_")[0]), int(stem.split("_e")[-1])
            eps.append({"level": lvl, "stem": stem, "task": task, "ep": ep, "success": succ.get((task, ep)),
                        "lab": lab, "runs": runs, "gap": gap, "calls": calls})
    return eps


def read_L(c):
    return {"searching": 0, "latched": 0, "closing": 1, "held": 5, "placed": 4}.get(c["latch"], 0)  # 5 = carry


def read_R(c):
    p, s = c["p"], c["latch"]
    if s in ("searching", "latched") or s is None:
        return 0
    if s == "closing":
        return 1 if p[1:].sum() >= p[0] else 0
    if s == "held":
        return 0 if p[0] >= 0.5 else 2 + int(np.argmax(p[2:]))
    if s == "placed":
        return 4 if p[4] >= p[0] else 0
    return 0


def transitions():
    T = np.zeros((5, 5))
    for i in range(5):
        T[i, i] = 0.85
        if i < 4:
            T[i, i + 1] = 0.10
        if i > 0:
            T[i, 0] += 0.05
    T[4, 3] += 0.03
    return T / T.sum(1, keepdims=True)


def hmm(eps, hold_task):
    cnt = np.ones((5, len(LATCH)))
    for e in eps:
        if e["task"] == hold_task:
            continue
        for c in e["calls"]:
            if c["latch"] in LATCH:
                cnt[e["lab"][c["obs"]], LATCH.index(c["latch"])] += 1
    em = cnt / cnt.sum(1, keepdims=True)
    T = transitions()
    out = {}
    for e in eps:
        if e["task"] != hold_task:
            continue
        b = np.array([1.0, 0, 0, 0, 0])
        seq = []
        for c in e["calls"]:
            pr = b @ T
            lik = np.nan_to_num(c["p"], nan=0.2) + 1e-6
            if c["latch"] in LATCH:
                lik = lik * em[:, LATCH.index(c["latch"])]
            b = pr * lik
            b /= b.sum()
            seq.append(int(np.argmax(b)))
        out[e["stem"] + e["level"]] = seq
    return out


def macro_f1(y, p, K=5):
    f = []
    for k in range(K):
        tp = np.sum((y == k) & (p == k)); fp = np.sum((y != k) & (p == k)); fn = np.sum((y == k) & (p != k))
        if tp + fn == 0:
            continue
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def evaluate(eps):
    H = {}
    for t in sorted({e["task"] for e in eps}):
        H.update(hmm(eps, t))
    Y, P, L, R, HH = [], [], [], [], []
    for e in eps:
        hs = H[e["stem"] + e["level"]]
        for i, c in enumerate(e["calls"]):
            Y.append(e["lab"][c["obs"]]); P.append(int(np.nanargmax(c["p"])) if np.isfinite(c["p"]).any() else 0)
            L.append(read_L(c)); R.append(read_R(c)); HH.append(hs[i])
            c["R"], c["H"], c["y"] = R[-1], hs[i], Y[-1]
    Y, P, L, R, HH = map(np.array, (Y, P, L, R, HH))
    to3 = lambda a: np.where(a >= 2, 2, a)  # approach / grasp / carry
    res = {"n_calls": int(len(Y)), "label_counts": {PH[k]: int((Y == k).sum()) for k in range(5)}, "readers": {}}
    for name, a in (("P probe", P), ("L latch", L), ("R rule fusion", R), ("H HMM fusion", HH)):
        d = {"acc3": float((to3(a) == to3(Y)).mean())}
        if name[0] != "L":
            d["acc5"] = float((a == Y).mean())
            d["macroF1_5"] = macro_f1(Y, a)
            d["recall"] = {PH[k]: float(((a == k) & (Y == k)).sum() / max((Y == k).sum(), 1)) for k in range(5)}
            d["confusion"] = [[int(((Y == i) & (a == j)).sum()) for j in range(5)] for i in range(5)]
        else:
            d["recall3"] = {n: float(((to3(a) == k) & (to3(Y) == k)).sum() / max((to3(Y) == k).sum(), 1))
                            for k, n in enumerate(("approach", "grasp", "carry"))}
        res["readers"][name] = d
    return res


def events(eps):
    rows = []
    for e in eps:
        cs = e["calls"]
        for run in e["runs"]:
            s, en = run["start"], run["end"]
            win = [c for c in cs if s <= c["obs"] < en + 5]
            lg = next((c["obs"] for c in win if c["not_empty"] is False), None)
            hp = None
            for a, b in zip(win, win[1:]):
                if all(x["latch"] == "closing" and x["p"][0] >= 0.7 for x in (a, b)):
                    hp = b["obs"]
                    break
            f = min([x for x in (lg, hp) if x is not None], default=None)
            rows.append({"stem": e["stem"], "level": e["level"], **run, "Lg": lg, "Hp": hp, "F": f})
    summ = {}
    for oc in ("HELD", "EMPTY", "PARTIAL"):
        rs = [r for r in rows if r["outcome"] == oc]
        d = {"runs": len(rs)}
        for k in ("Lg", "Hp", "F"):
            fired = [r for r in rs if r[k] is not None]
            d[k + "_fired"] = len(fired)
            d[k + "_latency_frames_median"] = (float(np.median([r[k] - r["start"] for r in fired])) if fired else None)
            if oc == "HELD":  # alarm before the bowl was held = false alarm
                d[k + "_false_before_held"] = sum(1 for r in fired if r[k] < r["first_held"])
        summ[oc] = d
    return rows, summ


def dest_gate(eps):
    out = {}
    for name, fn in (("gL latch held", lambda c: c["latch"] == "held"),
                     ("gR held and R in transport/placement", lambda c: c["latch"] == "held" and c["R"] in (3, 4)),
                     # added after the first run (00:09, not fixed before results): the HMM reader as the gate
                     ("gH held and H in transport/placement", lambda c: c["latch"] == "held" and c["H"] in (3, 4))):
        open_ = [c for e in eps for c in e["calls"] if fn(c)]
        y = np.array([c["y"] for c in open_]) if open_ else np.array([], int)
        out[name] = {"open_calls": len(open_),
                     "truth_transport_or_placement": int(np.isin(y, (3, 4)).sum()),
                     "truth_lift": int((y == 2).sum()),
                     "truth_approach_or_grasp (bowl not carried)": int(np.isin(y, (0, 1)).sum()),
                     "precision": float(np.isin(y, (3, 4)).mean()) if len(y) else None}
        tp_all = sum(1 for e in eps for c in e["calls"] if c["y"] in (3, 4))
        out[name]["recall_of_truth_transport_placement"] = out[name]["truth_transport_or_placement"] / max(tp_all, 1)
    return out


def figure(eps, path, stems):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#9aa5b1", "#e8a33d", "#5aa469", "#3c78b5", "#a35bb5"]
    sel = [e for s in stems for e in eps if (e["level"], e["stem"]) == s]
    fig, axes = plt.subplots(len(sel), 1, figsize=(11, 1.9 * len(sel)), squeeze=False)
    for ax, e in zip(axes[:, 0], sel):
        cs = e["calls"]
        x = [c["obs"] for c in cs]
        for row, (name, vals) in enumerate((("truth", [c["y"] for c in cs]), ("probe P", [int(np.nanargmax(c["p"])) for c in cs]),
                                            ("latch", [LATCH.index(c["latch"]) if c["latch"] in LATCH else 0 for c in cs]),
                                            ("fusion R", [c["R"] for c in cs]))):
            for xi, v in zip(x, vals):
                color = cols[v] if name != "latch" else ["#cfd6dd", "#9aa5b1", "#e8a33d", "#3c78b5", "#a35bb5"][v]
                ax.add_patch(plt.Rectangle((xi, 3 - row), 5, 0.8, color=color, lw=0))
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(e["gap"])), e["gap"] * 1e3, color="k", lw=0.8)
        ax2.set_ylabel("finger gap mm", fontsize=7)
        ax2.set_ylim(0, 85)
        ax.set_xlim(15, 305); ax.set_ylim(0, 4)
        ax.set_yticks([3.4, 2.4, 1.4, 0.4]); ax.set_yticklabels(["truth", "probe", "latch", "fusion R"], fontsize=7)
        ax.set_title(f"{e['level']} {e['stem']}  success={e['success']}", fontsize=8, loc="left")
    axes[-1, 0].set_xlabel("observation index (env step)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    fig.legend(handles, PH, loc="upper right", ncol=5, fontsize=7)
    fig.text(0.01, 0.005, "latch row: grey=searching/latched, orange=closing, blue=held, purple=placed; black line = finger gap",
             fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = load()
    res = evaluate(eps)
    rows, summ = events(eps)
    res["close_runs"] = summ
    res["destination_gate"] = dest_gate(eps)
    res["latch_vs_truth"] = {s: {PH[k]: sum(1 for e in eps for c in e["calls"] if c["latch"] == s and c["y"] == k)
                                 for k in range(5)} for s in LATCH}
    res["episodes"] = len(eps)
    json.dump(res, open(out / "replay.json", "w"), indent=1)
    json.dump(rows, open(out / "close_runs.json", "w"), indent=1)
    figure(eps, out / "그래프_단계_타임라인.png",
           [("LI", "safelibero_spatial_t0_e1"), ("LI", "safelibero_spatial_t1_e1"),
            ("LII", "safelibero_spatial_t2_e0"), ("LII", "safelibero_spatial_t3_e2")])
    bars(res, out / "그래프_요약.png")
    print(json.dumps(res, indent=1)[:6000])


def bars(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    rd = res["readers"]
    names = ["L latch", "P probe", "R rule fusion", "H HMM fusion"]
    ax[0].bar(range(4), [rd[n]["acc3"] for n in names], color=["#9aa5b1", "#3c78b5", "#e8a33d", "#5aa469"])
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["latch", "probe", "rule R", "HMM H"], fontsize=8)
    ax[0].set_ylim(0.5, 0.95); ax[0].set_title("3-level accuracy (approach/grasp/carry)", fontsize=9)
    for i, n in enumerate(names):
        ax[0].text(i, rd[n]["acc3"] + 0.005, f"{rd[n]['acc3']:.3f}", ha="center", fontsize=8)
    cr = res["close_runs"]
    det = ["Lg", "Hp", "F"]
    x = np.arange(3)
    ax[1].bar(x - 0.2, [cr["EMPTY"][d + "_fired"] / cr["EMPTY"]["runs"] for d in det], 0.4, label="EMPTY runs detected")
    ax[1].bar(x + 0.2, [cr["HELD"][d + "_fired"] / cr["HELD"]["runs"] for d in det], 0.4, label="HELD runs alarmed (false)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["gripper Lg", "hidden Hp", "Lg or Hp"], fontsize=8)
    ax[1].set_title(f"failing-close alarm (EMPTY n={cr['EMPTY']['runs']}, HELD n={cr['HELD']['runs']})", fontsize=9)
    ax[1].legend(fontsize=7)
    g = res["destination_gate"]
    gk = list(g)
    ax[2].bar(x - 0.2, [g[k]["precision"] for k in gk], 0.4, label="precision")
    ax[2].bar(x + 0.2, [g[k]["recall_of_truth_transport_placement"] for k in gk], 0.4, label="recall")
    ax[2].set_xticks(x); ax[2].set_xticklabels(["latch held", "held ∧ R", "held ∧ H"], fontsize=8)
    ax[2].set_title("destination gate vs truth transport/placement", fontsize=9)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
