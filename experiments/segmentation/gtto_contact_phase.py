#!/usr/bin/env python3
"""Held object · placement contact · task phase diagnosis of the GT common-TO arms (user cautions 2026-11-07 00:1x KST:
(1) the held object may still look like an obstacle -- AG3S should fold it into the robot links and mask it;
(2) at placing, contact between the held object and what it is put on must be allowed;
(3) task phases may be missed).

Simulator truth per observation index t (identical for every arm; same definitions as gtto_phase.py where they overlap):
  approach  before the first executed gripper-close command (action[6] > 0)
  closing   close command .. the task target is >= LIFT_M above its rest height
  carry     lifted .. PLACE_BEFORE steps before the first open command after the lift (action[6] < 0)
  place     [open - PLACE_BEFORE, open + PLACE_AFTER)
  after     the rest
Per call (``*_calls.jsonl``): latch state (``grasp.state``), HOLD, the hold rows (``verdict_reasons`` action == hold:
``why`` / link / step / point_m), geometry age (``grasp.alignment.geometry_age_frames``), grounding status.
Each hold row's point (base frame -> world with ``T_world_base``) is attributed to the nearest free object origin, with the task target split into
``target`` (where it is now) and ``target_ghost`` (where it rested at t0, once the target has moved > GHOST_M):
a ghost is the held object's old surface still in the distance field.

Outputs (DIR): contact_phase.json, 단계접촉표.md, 그래프_단계접촉.png.
usage: python3 experiments/segmentation/gtto_contact_phase.py --out DIR [--arms gt10ee gt5zr]
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
ARM_RE = re.compile(r"_gtto-(?P<arm>gt11hdp|gt10ee|gt9it8|gt8hand|gt7ochbc|gt6oc|gt5zr|gt3termsoff-pass|gt2)-(?P<bench>libero9|safelibero11-t\d|ls10)-")
LIFT_M, PLACE_BEFORE, PLACE_AFTER, GHOST_M = 0.03, 25, 10, 0.05
PHASES = ("approach", "closing", "carry", "place", "after")


def bench_of(tag):
    return "LIBERO" if tag == "libero9" else ("LIBERO-Safety" if tag == "ls10" else "SafeLIBERO")


def free_joints(names):
    a, i = {}, 0
    for n in names:
        n = str(n)
        if "joint0" in n:
            a[n] = i
        i += 7 if "joint0" in n else 1
    return a


def truth(npz, tgt):
    E = np.load(npz)
    G = np.load(npz[:-4] + "_geom.npz", allow_pickle=True)
    act = E["action"]
    wait = E["is_wait"].astype(bool) if "is_wait" in E.files else np.zeros(len(act), bool)
    fj = free_joints(G["model_joint_names"])
    S = G["sim_state"]
    pos = {n[: -len("_joint0")]: S[:, 1 + a:1 + a + 3] for n, a in fj.items()}   # object name -> (T, 3) world origin
    if tgt not in pos:
        return None
    steps = np.nonzero(~wait)[0]
    t0 = int(steps[0]) if len(steps) else 0
    T = len(act)
    z = pos[tgt][:, 2]
    rest_z = float(z[min(t0, len(z) - 1)])
    close = next((int(t) for t in steps if act[t, 6] > 0), None)
    lift = next((int(t) for t in range(close, len(z)) if z[t] - rest_z >= LIFT_M), None) if close is not None else None
    release = next((int(t) for t in range(lift, T) if act[t, 6] < 0), None) if lift is not None else None
    ph = np.array(["approach"] * (T + 1), dtype=object)
    if close is not None:
        ph[close:] = "closing"
    if lift is not None:
        ph[lift:] = "carry"
    if release is not None:
        a, b = max(lift, release - PLACE_BEFORE), min(T + 1, release + PLACE_AFTER)
        ph[a:b] = "place"
        ph[b:] = "after"
    return {"phase": ph, "pos": pos, "rest": pos[tgt][min(t0, len(z) - 1)].copy(), "tgt": tgt, "t0": t0,
            "T_world_base": np.asarray(G["T_world_base"], float),
            "close": close, "lift": lift, "release": release}


def attribute(point, t, tr):
    """``point`` = the hold row's ``point_m`` (robot **base** frame, as the server writes it) -> world, then nearest object."""
    t = min(t, len(tr["pos"][tr["tgt"]]) - 1)
    Tw = tr["T_world_base"][min(t, len(tr["T_world_base"]) - 1)]
    point = Tw[:3, :3] @ point + Tw[:3, 3]
    best, dmin = None, 1e9
    for name, p in tr["pos"].items():
        d = float(np.linalg.norm(p[t] - point))
        if d < dmin:
            best, dmin = (("target" if name == tr["tgt"] else "other:" + name), d)
    moved = float(np.linalg.norm(tr["pos"][tr["tgt"]][t] - tr["rest"]))
    dg = float(np.linalg.norm(tr["rest"] - point))
    if moved > GHOST_M and dg < dmin:
        best, dmin = "target_ghost", dg
    return best, dmin


def target_name(rows):
    for r in rows:
        g = ((r.get("gt_perception") or {}).get("grounding") or {}).get("target")
        if g:
            return g
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["gt10ee", "gt5zr"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    conf = collections.Counter()      # (arm, bench, truth phase, latch state)
    hold = collections.Counter()      # (arm, bench, truth phase, why, step0?)
    owner = collections.Counter()     # (arm, bench, truth phase, why, owner class)
    calls = collections.Counter()     # (arm, bench, truth phase)
    holds = collections.Counter()
    age = collections.defaultdict(list)
    status = collections.Counter()    # (arm, bench, truth phase, grounding status)
    episodes = {}
    for d in sorted(RESULTS.glob("*_gtto-*")):
        m = ARM_RE.search(d.name)
        if not m or m.group("arm") not in a.arms or "smoke" in d.name:
            continue
        arm, bench = m.group("arm"), bench_of(m.group("bench"))
        succ = {}
        if (d / "episodes.jsonl").exists():
            for line in (d / "episodes.jsonl").read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    succ[(int(r["task_id"]), int(r["episode_idx"]))] = bool(r.get("official_success"))
        for f in sorted(glob.glob(str(d / "episodes" / "*_calls.jsonl"))):
            base = f[: -len("_calls.jsonl")]
            rows = [json.loads(x) for x in open(f) if x.strip()]
            tr = truth(base + ".npz", target_name(rows)) if os.path.exists(base + ".npz") else None
            if tr is None:
                continue
            stem = os.path.basename(base)
            te = re.search(r"_t(\d+)_e(\d+)$", stem)
            ep = {"success": succ.get((int(te.group(1)), int(te.group(2)))), "close": tr["close"], "lift": tr["lift"],
                  "release": tr["release"], "latch_seq": [], "hold_by_phase": collections.Counter()}
            for r in rows:
                t = int(r.get("obs_index") or 0)
                ph = str(tr["phase"][min(t, len(tr["phase"]) - 1)])
                g = r.get("grasp") or {}
                st = str(g.get("state"))
                key = (arm, bench, ph)
                calls[key] += 1
                conf[key + (st,)] += 1
                if not ep["latch_seq"] or ep["latch_seq"][-1][1] != st:
                    ep["latch_seq"].append((t, st, ph))
                al = g.get("alignment") or {}
                if al.get("geometry_age_frames") is not None:
                    age[key].append(float(al["geometry_age_frames"]))
                gs = (((r.get("gt_perception") or {}).get("grounding")) or {}).get("status")
                status[key + (str(gs),)] += 1
                if not r.get("hold"):
                    continue
                holds[key] += 1
                ep["hold_by_phase"][ph] += 1
                for rs in r.get("verdict_reasons") or []:
                    if rs.get("action") != "hold":
                        continue
                    row = (rs.get("evidence") or {}).get("row") or {}
                    why = str(row.get("why") or rs.get("kind"))
                    hold[key + (why, "step0" if row.get("step") == 0 else "step>0")] += 1
                    if row.get("point_m") is not None:
                        cls, dist = attribute(np.asarray(row["point_m"], float), t, tr)
                        owner[key + (why, cls if dist < 0.15 else "none<15cm")] += 1
            ep["hold_by_phase"] = dict(ep["hold_by_phase"])
            episodes[f"{arm}/{bench}/{stem}"] = ep
    js = {"definitions": {"lift_m": LIFT_M, "place_before": PLACE_BEFORE, "place_after": PLACE_AFTER, "ghost_m": GHOST_M},
          "calls": {"|".join(k): v for k, v in calls.items()}, "holds": {"|".join(k): v for k, v in holds.items()},
          "latch_vs_truth": {"|".join(k): v for k, v in conf.items()},
          "hold_rows": {"|".join(k): v for k, v in hold.items()},
          "hold_owner": {"|".join(k): v for k, v in owner.items()},
          "geometry_age_median": {"|".join(k): float(np.median(v)) for k, v in age.items()},
          "grounding_status": {"|".join(k): v for k, v in status.items()}, "episodes": episodes}
    (a.out / "contact_phase.json").write_text(json.dumps(js, indent=1, ensure_ascii=False, default=str))
    md = []
    for arm in a.arms:
        for bench in ("LIBERO", "SafeLIBERO", "LIBERO-Safety"):
            ks = [p for p in PHASES if calls[(arm, bench, p)]]
            if not ks:
                continue
            md.append(f"\n### {arm} · {bench}\n")
            md.append("| sim 단계 | call | HOLD | geometry 나이 중앙값 (frame) | latch 상태 (call 수) | grounding status | HOLD 행 (why · step) | HOLD 행 위치 주인 |")
            md.append("|---|---:|---:|---:|---|---|---|---|")
            for p in ks:
                k = (arm, bench, p)
                lat = ", ".join(f"{s} {n}" for (a1, b1, p1, s), n in sorted(conf.items(), key=lambda x: -x[1]) if (a1, b1, p1) == k)
                gst = ", ".join(f"{s} {n}" for (a1, b1, p1, s), n in sorted(status.items(), key=lambda x: -x[1]) if (a1, b1, p1) == k)
                hr = ", ".join(f"{w} {s} {n}" for (a1, b1, p1, w, s), n in sorted(hold.items(), key=lambda x: -x[1]) if (a1, b1, p1) == k)
                ow = ", ".join(f"{w}→{c} {n}" for (a1, b1, p1, w, c), n in sorted(owner.items(), key=lambda x: -x[1]) if (a1, b1, p1) == k)
                ag = f"{np.median(age[k]):.0f}" if age[k] else "–"
                md.append(f"| {p} | {calls[k]} | {holds[k]} | {ag} | {lat} | {gst} | {hr or '–'} | {ow or '–'} |")
    md.append("\n출처: 각 run 의 `episodes/*_calls.jsonl` · `*.npz` · `*_geom.npz` (sim_state). script `experiments/segmentation/gtto_contact_phase.py`.")
    (a.out / "단계접촉표.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    _figure(a.out, a.arms, calls, holds, age)


def _figure(out, arms, calls, holds, age):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(PHASES))
    w = 0.8 / max(1, 3 * len(arms))
    i = 0
    for arm in arms:
        for bench in ("LIBERO", "SafeLIBERO", "LIBERO-Safety"):
            hr = [holds[(arm, bench, p)] / calls[(arm, bench, p)] if calls[(arm, bench, p)] else np.nan for p in PHASES]
            ag = [np.median(age[(arm, bench, p)]) if age[(arm, bench, p)] else np.nan for p in PHASES]
            ax[0].bar(x + i * w, hr, w, label=f"{arm} · {bench}")
            ax[1].bar(x + i * w, ag, w, label=f"{arm} · {bench}")
            i += 1
    for k, (t, yl) in enumerate((("HOLD share per sim phase", "HOLD calls / calls"),
                                  ("geometry age per sim phase", "median geometry age (frames)"))):
        ax[k].set_xticks(x + 0.4 - w / 2)
        ax[k].set_xticklabels(PHASES)
        ax[k].set_title(t, fontsize=10)
        ax[k].set_ylabel(yl)
    ax[0].legend(fontsize=7)
    fig.suptitle("GT common TO: HOLD and geometry age by simulator phase (seed 7, 1 episode per scene)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "그래프_단계접촉.png", dpi=130)


if __name__ == "__main__":
    main()
