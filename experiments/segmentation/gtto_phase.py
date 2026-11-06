#!/usr/bin/env python3
"""Per-phase failure analysis of the GT common-TO arms vs VLA alone (user request 2026-11-05 ~09:5x KST:
"쥐기 전 · 쥐는 중 · 쥔 후 로 단계를 나눠 어느 구간에서 문제가 생기는지").

Phases come from the **simulator**, identically for every arm (VLA alone has no latch):
  pre    : from the first policy step to the first executed gripper-close command (action[6] > 0)
  grasp  : from that close command until the task target is lifted >= LIFT_M above its rest height (sim body z)
  post   : from the lift to the end of the episode
Failure phase of an unsuccessful episode = the phase it ended in ("pre" = never closed, "grasp" = closed but never
lifted, "post" = lifted but not done). `dropped` (target back at rest height at the end) is kept in the json only: a placed object
also comes back near its rest height, so it does not mean a drop.

Per phase and arm (GT arms, from `_calls.jsonl` / `_calls.npz`): calls, HOLD share, median `to.reference_deviation`,
C-saturated call share (>= 5/10 rows), mean |executed − policy| xyz action (TO intervention) and steps spent.

usage: python3 experiments/segmentation/gtto_phase.py --out DIR [--arms vla gt2 gt3termsoff-pass gt5zr ...]
"""
import argparse
import collections
import glob
import json
import os
import pathlib
import re

import numpy as np

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
LIFT_M = 0.03
ARM_RE = re.compile(r"_gtto-(?P<arm>vla|gt10ee|gt9it8|gt8hand|gt7ochbc|gt6oc|gt5zr|gt4termsoff-pass-hbc|gt4termsoff-hbc|gt4hbc|gt3termsoff-pass|gt3termsoff|"
                    r"gt3pass|gt2|gt)-(?P<bench>libero9|safelibero11-t\d|ls10)-")
BENCH = {"libero9": "LIBERO", "ls10": "LIBERO-Safety"}
PHASES = ("pre", "grasp", "post")


def _addr(names):
    a, i = {}, 0
    for n in names:
        n = str(n)
        a[n] = i
        i += 7 if "joint0" in n else 1
    return a


def runs():
    out = collections.defaultdict(list)
    for d in sorted(RESULTS.glob("*_gtto-*")):
        m = ARM_RE.search(d.name)
        if m and "gt2rec" not in d.name and "smoke" not in d.name:
            out[m.group("arm")].append(d)
    return out


def episode_success(run_dir):
    res = {}
    p = run_dir / "episodes.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                res[(int(r["task_id"]), int(r["episode_idx"]))] = bool(r.get("official_success"))
    return res


def targets(all_runs):
    """episode stem -> GT target body name (from any GT arm's call record)."""
    t = {}
    for arm, dirs in all_runs.items():
        if arm == "vla":
            continue
        for d in dirs:
            for f in glob.glob(str(d / "episodes" / "*_calls.jsonl")):
                stem = os.path.basename(f).replace("_calls.jsonl", "")
                if stem in t:
                    continue
                for line in open(f):
                    if not line.strip():
                        continue
                    g = ((json.loads(line).get("gt_perception") or {}).get("grounding") or {}).get("target")
                    if g:
                        t[stem] = g
                        break
    return t


def analyse_episode(npz, tgt):
    base = npz[:-4]
    E = np.load(npz)
    G = np.load(base + "_geom.npz", allow_pickle=True)
    act = E["action"]
    wait = E["is_wait"].astype(bool) if "is_wait" in E.files else np.zeros(len(act), bool)
    A = _addr(G["model_joint_names"])
    jn = [k for k in A if tgt and k.startswith(tgt + "_joint")]
    if not jn:
        return None
    ad = 1 + A[jn[0]]
    oi = G["obs_index"] if "obs_index" in G.files else np.arange(len(G["sim_state"]))
    z = G["sim_state"][:, ad + 2]
    steps = np.nonzero(~wait)[0]
    t0 = int(steps[0]) if len(steps) else 0
    rest = float(z[min(t0, len(z) - 1)])
    close = next((int(t) for t in steps if act[t, 6] > 0), None)
    lift = None
    if close is not None:
        lift = next((int(t) for t in range(close, len(z)) if z[t] - rest >= LIFT_M), None)
    dropped = bool(lift is not None and z[-1] - rest < 0.01)
    T = len(act)
    bounds = {"pre": (t0, close if close is not None else T),
              "grasp": (close, lift if lift is not None else T) if close is not None else None,
              "post": (lift, T) if lift is not None else None}
    end = "pre" if close is None else ("grasp" if lift is None else "post")
    return {"bounds": bounds, "end": end, "dropped": dropped, "T": T, "close": close, "lift": lift, "t0": t0}


def call_metrics(base, bounds):
    f = base + "_calls.jsonl"
    if not os.path.exists(f):
        return None
    rows = [json.loads(line) for line in open(f) if line.strip()]
    c = np.load(base + "_calls.npz") if os.path.exists(base + "_calls.npz") else None
    out = {ph: {"calls": 0, "hold": 0, "dev": [], "sat5": 0, "dA": []} for ph in PHASES}
    for k, r in enumerate(rows):
        t = int(r.get("obs_index") or 0)
        ph = next((p for p in PHASES if bounds.get(p) and bounds[p][0] <= t < bounds[p][1]), None)
        if ph is None:
            continue
        o = out[ph]
        o["calls"] += 1
        if r.get("hold"):
            o["hold"] += 1
            continue
        o["dev"].append(float((r.get("to") or {}).get("reference_deviation") or 0.0))
        o["sat5"] += len((r.get("conversion") or {}).get("saturated_rows") or []) >= 5
        if c is not None and k < len(c["actions_policy"]):
            pol = np.asarray(c["actions_policy"][k][:5, :3], float)
            exe = np.asarray(c["actions_executed"][k][:, :3], float)
            n = min(len(pol), len(exe))
            o["dA"].append(float(np.abs(exe[:n] - pol[:n]).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    all_runs = runs()
    tg = targets(all_runs)
    arms = a.arms or sorted(all_runs)
    per_ep, summary = {}, {}
    for arm in arms:
        for d in all_runs.get(arm, []):
            bench = BENCH.get(ARM_RE.search(d.name).group("bench"), "SafeLIBERO")
            succ = episode_success(d)
            for npz in sorted(glob.glob(str(d / "episodes" / "*.npz"))):
                if npz.endswith(("_calls.npz", "_geom.npz")):
                    continue
                stem = os.path.basename(npz)[:-4]
                m = re.search(r"_t(\d+)_e(\d+)$", stem)
                ep = analyse_episode(npz, tg.get(stem))
                if ep is None or m is None:
                    continue
                ok = succ.get((int(m.group(1)), int(m.group(2))))
                cm = call_metrics(npz[:-4], ep["bounds"]) if arm != "vla" else None
                per_ep[f"{arm}/{bench}/{stem}"] = {**{k: v for k, v in ep.items() if k != "bounds"},
                                                   "success": ok,
                                                   "steps": {p: (None if ep["bounds"][p] is None else
                                                                 int(ep["bounds"][p][1] - ep["bounds"][p][0]))
                                                             for p in PHASES},
                                                   "calls": cm}
    # summary per arm x bench
    for key, e in per_ep.items():
        arm, bench, _ = key.split("/")
        s = summary.setdefault(arm, {}).setdefault(bench, {
            "n": 0, "success": 0, "fail_pre": 0, "fail_grasp": 0, "fail_post": 0, "dropped": 0,
            **{f"{p}_{m}": [] for p in PHASES for m in ("steps", "hold", "dev", "sat", "dA")}})
        s["n"] += 1
        if e["success"]:
            s["success"] += 1
        else:
            s[f"fail_{e['end']}"] += 1
            s["dropped"] += bool(e["end"] == "post" and e["dropped"])
        for p in PHASES:
            if e["steps"][p] is not None:
                s[f"{p}_steps"].append(e["steps"][p])
            cm = (e["calls"] or {}).get(p)
            if cm and cm["calls"]:
                s[f"{p}_hold"].append(cm["hold"] / cm["calls"])
                if cm["dev"]:
                    s[f"{p}_dev"].append(float(np.median(cm["dev"])))
                    s[f"{p}_sat"].append(cm["sat5"] / max(1, cm["calls"] - cm["hold"]))
                if cm["dA"]:
                    s[f"{p}_dA"].append(float(np.mean(cm["dA"])))
    table = {arm: {b: {k: (round(float(np.median(v)), 3) if isinstance(v, list) and v else (None if isinstance(v, list)
                                                                                             else v))
                       for k, v in s.items()} for b, s in bs.items()} for arm, bs in summary.items()}
    (a.out / "phase.json").write_text(json.dumps({"lift_m": LIFT_M, "summary": table, "episodes": per_ep},
                                                 indent=1, ensure_ascii=False, default=float))
    lines = ["| 판 | 벤치 | n | 성공 | 실패: 쥐기 전 | 쥐는 중 | 쥔 후 | 쥐기 전 step | 쥐는 중 step | 쥔 후 step |"
             " HOLD 비율 전/중/후 | dev 중앙값 전/중/후 | C 포화 전/중/후 | |Δa| 전/중/후 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
    order = [x for x in ("vla", "gt", "gt2", "gt3termsoff", "gt3pass", "gt3termsoff-pass", "gt4hbc", "gt4termsoff-hbc",
                         "gt4termsoff-pass-hbc", "gt5zr", "gt6oc", "gt7ochbc", "gt8hand", "gt9it8", "gt10ee") if x in table]
    for arm in order:
        for b in ("LIBERO", "SafeLIBERO", "LIBERO-Safety"):
            s = table[arm].get(b)
            if not s:
                continue
            f3 = lambda m: " / ".join("–" if s[f"{p}_{m}"] is None else f"{s[f'{p}_{m}']}" for p in PHASES)
            lines.append(f"| {arm} | {b} | {s['n']} | {s['success']} | {s['fail_pre']} | {s['fail_grasp']} | "
                         f"{s['fail_post']} | {s['pre_steps']} | {s['grasp_steps']} | {s['post_steps']} | "
                         f"{f3('hold')} | {f3('dev')} | {f3('sat')} | {f3('dA')} |")
    (a.out / "단계표.md").write_text("\n".join(lines) + "\n\nstep = 중앙값 (그 단계에 들어간 episode 만). HOLD · dev · C 포화 · "
                                     "|Δa| = episode 별 값의 중앙값. |Δa| = 실행 − 정책 action xyz 평균 (TO 개입 크기).\n")
    print("\n".join(lines))
    _figure(a.out, table, order)


def _figure(out, table, order):
    from AG3S.ag3s.experiments.common import figstyle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ko = figstyle.use_korean()
    benches = ("LIBERO", "SafeLIBERO", "LIBERO-Safety")
    lab = {"pre": "쥐기 전" if ko else "pre-grasp", "grasp": "쥐는 중" if ko else "grasping",
           "post": "쥔 후" if ko else "post-grasp", "ok": "성공" if ko else "success"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = {"ok": "#55A868", "pre": "#C44E52", "grasp": "#DD8452", "post": "#8172B2"}
    for j, b in enumerate(benches):
        ax = axes[0, j]
        arms = [x for x in order if b in table[x]]
        bottom = np.zeros(len(arms))
        for key, col in (("success", "ok"), ("fail_pre", "pre"), ("fail_grasp", "grasp"), ("fail_post", "post")):
            v = np.array([table[x][b][key] for x in arms], float)
            ax.bar(arms, v, bottom=bottom, color=colors[col], label=lab[col])
            bottom += v
        ax.set_title((f"{b}: 결과 / 실패 단계" if ko else f"{b}: outcome / failure phase"), fontsize=10)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)
        ax = axes[1, j]
        arms_gt = [x for x in arms if x != "vla"]
        w = 0.27
        for i, p in enumerate(PHASES):
            v = [table[x][b][f"{p}_dA"] or 0 for x in arms_gt]
            ax.bar(np.arange(len(arms_gt)) + (i - 1) * w, v, w, label=lab[p], color=colors[p])
        ax.set_xticks(np.arange(len(arms_gt)))
        ax.set_xticklabels(arms_gt, rotation=60, fontsize=7)
        ax.set_title(("TO 개입 |실행 − 정책| xyz (단계별)" if ko else "TO intervention |executed − policy| xyz by phase"),
                     fontsize=10)
        if j == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "그래프_단계별.png", dpi=130)


if __name__ == "__main__":
    main()
